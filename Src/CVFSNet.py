# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import math
from typing import Callable, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from fvcore.nn.squeeze_excitation import SqueezeExcitation
from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill
from collections import OrderedDict
from termcolor import cprint
import math
from loguru import logger
from thop import profile


@torch.fx.wrap
def _unsqueeze_dims_fx(tensor: torch.Tensor) -> Tuple[torch.Tensor, int]:
    tensor_dim = tensor.ndim
    if tensor_dim == 4:
        pass
    elif tensor_dim == 3:
        tensor = tensor.unsqueeze(1)
    else:
        raise NotImplementedError(f"Unsupported input dimension {tensor.shape}")
    return tensor, tensor_dim


@torch.jit.script
def _unsqueeze_dims_jit(tensor: torch.Tensor) -> Tuple[torch.Tensor, int]:
    return _unsqueeze_dims_fx(tensor)


@torch.fx.wrap
def _squeeze_dims_fx(tensor: torch.Tensor, tensor_dim: int) -> torch.Tensor:
    if tensor_dim == 4:
        pass
    elif tensor_dim == 3:
        tensor = tensor.squeeze(1)
    else:
        raise NotImplementedError(f"Unsupported input dimension {tensor.shape}")
    return tensor


@torch.jit.script
def _squeeze_dims_jit(tensor: torch.Tensor, tensor_dim: int) -> torch.Tensor:
    return _squeeze_dims_fx(tensor, tensor_dim)


def move_to_cpu(target):
    if isinstance(target, List):
        out = []
        for i in target:
            out.append(i.detach().cpu().numpy())
        return out
    elif isinstance(target, dict):
        out = {}
        for k, v in target.items():
            out[k] = v.detach().cpu().numpy()
        return out
    else:
        return target


class _AttentionPool(torch.nn.Module):
    def __init__(
        self,
        pool: Optional[torch.nn.Module],
        has_cls_embed: bool,
        norm: Optional[torch.nn.Module],
    ) -> None:
        """Apply pool to a flattened input (given pool operation and the unflattened shape).


                                         Input
                                           ↓
                                        Reshape
                                           ↓
                                          Pool
                                           ↓
                                        Reshape
                                           ↓
                                          Norm


        Params:
            pool (Optional[Callable]): Pool operation that is applied to the input tensor.
                If pool is none, return the input tensor.
            has_cls_embed (bool): Whether the input tensor contains cls token. Pool
                operation excludes cls token.
            norm: (Optional[Callable]): Optional normalization operation applied to
            tensor after pool.
        """
        super().__init__()
        self.has_pool = pool is not None
        self.pool = pool if pool is not None else torch.nn.Identity()

        self.has_cls_embed = has_cls_embed
        if norm is not None:
            self.norm_before_pool = isinstance(
                norm, (torch.nn.BatchNorm3d, torch.nn.Identity)
            )
            self.has_norm = True
            self.norm = norm
        else:
            self.norm_before_pool = False
            self.has_norm = False
            self.norm = torch.nn.Identity()

    def forward(
        self, tensor: torch.Tensor, thw_shape: List[int]
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Args:
            tensor (torch.Tensor): Input tensor.
            thw_shape (List): The shape of the input tensor (before flattening).

        Returns:
            tensor (torch.Tensor): Input tensor after pool.
            thw_shape (List[int]): Output tensor shape (before flattening).
        """
        if not self.has_pool:
            return tensor, thw_shape
        tensor_dim = tensor.ndim

        if torch.jit.is_scripting():
            tensor, tensor_dim = _unsqueeze_dims_jit(tensor)
        else:
            tensor, tensor_dim = _unsqueeze_dims_fx(tensor)

        cls_tok: torch.Tensor = torch.tensor(0)  # For typing/torchscriptability
        if self.has_cls_embed:
            cls_tok, tensor = tensor[:, :, :1, :], tensor[:, :, 1:, :]

        B, N, L, C = tensor.shape
        T, H, W = thw_shape
        tensor = tensor.reshape(B * N, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()

        if self.norm_before_pool:
            # If use BN, we apply norm before pooling instead of after pooling.
            tensor = self.norm(tensor)
            # We also empirically find that adding a GELU here is beneficial.
            tensor = torch.nn.functional.gelu(tensor)

        tensor = self.pool(tensor)

        thw_shape = [tensor.shape[2], tensor.shape[3], tensor.shape[4]]
        L_pooled = tensor.shape[2] * tensor.shape[3] * tensor.shape[4]
        tensor = tensor.reshape(B, N, C, L_pooled).transpose(2, 3)
        if self.has_cls_embed:
            tensor = torch.cat((cls_tok, tensor), dim=2)
        if self.has_norm and not self.norm_before_pool:
            tensor = self.norm(tensor)

        if torch.jit.is_scripting():
            tensor = _squeeze_dims_jit(tensor, tensor_dim)
        else:
            tensor = _squeeze_dims_fx(tensor, tensor_dim)

        return tensor, thw_shape


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        # self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        # self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.pool_h = nn.AdaptiveAvgPool3d((None, None, 1))
        self.pool_w = nn.AdaptiveAvgPool3d((None, 1, None))
        self.pool_d = nn.AdaptiveAvgPool3d((1, None, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv3d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm3d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_d = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, d, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        x_d = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_d, x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_d, x_h, x_w = torch.split(y, [d, h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        x_d = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        a_d = self.conv_d(x_d).sigmoid()

        out = identity * a_w * a_h * a_d

        return out


class ResNetBasicStem(nn.Module):
    """
    ResNet basic 3D stem module. Performs spatiotemporal Convolution, BN, and activation
    following by a spatiotemporal pooling.

    ::

                                        Conv3d
                                           ↓
                                     Normalization
                                           ↓
                                       Activation
                                           ↓
                                        Pool3d

    The builder can be found in `create_res_basic_stem`.
    """

    def __init__(
        self,
        *,
        conv: nn.Module = None,
        norm: nn.Module = None,
        activation: nn.Module = None,
        pool: nn.Module = None,
    ) -> None:
        """
        Args:
            conv (torch.nn.modules): convolutional module.
            norm (torch.nn.modules): normalization module.
            activation (torch.nn.modules): activation module.
            pool (torch.nn.modules): pooling module.
        """
        super().__init__()
        set_attributes(self, locals())
        assert self.conv is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.pool is not None:
            x = self.pool(x)
        return x


class BottleneckBlock(nn.Module):
    """
    Bottleneck block: a sequence of spatiotemporal Convolution, Normalization,
    and Activations repeated in the following order:

    ::


                                    Conv3d (conv_a)
                                           ↓
                                 Normalization (norm_a)
                                           ↓
                                   Activation (act_a)
                                           ↓
                                    Conv3d (conv_b)
                                           ↓
                                 Normalization (norm_b)
                                           ↓
                                   Activation (act_b)
                                           ↓
                                    Conv3d (conv_c)
                                           ↓
                                 Normalization (norm_c)

    The builder can be found in `create_bottleneck_block`.
    """

    def __init__(
        self,
        *,
        conv_a: nn.Module = None,
        norm_a: nn.Module = None,
        act_a: nn.Module = None,
        conv_b: nn.Module = None,
        norm_b: nn.Module = None,
        act_b: nn.Module = None,
        conv_c: nn.Module = None,
        norm_c: nn.Module = None,
    ) -> None:
        """
        Args:
            conv_a (torch.nn.modules): convolutional module.
            norm_a (torch.nn.modules): normalization module.
            act_a (torch.nn.modules): activation module.
            conv_b (torch.nn.modules): convolutional module.
            norm_b (torch.nn.modules): normalization module.
            act_b (torch.nn.modules): activation module.
            conv_c (torch.nn.modules): convolutional module.
            norm_c (torch.nn.modules): normalization module.
        """
        super().__init__()
        set_attributes(self, locals())
        assert all(op is not None for op in (self.conv_a, self.conv_b, self.conv_c))
        if self.norm_c is not None:
            # This flag is used for weight initialization.
            self.norm_c.block_final_bn = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Explicitly forward every layer.
        # Branch2a, for example Tx1x1, BN, ReLU.
        x = self.conv_a(x)
        if self.norm_a is not None:
            x = self.norm_a(x)
        if self.act_a is not None:
            x = self.act_a(x)

        # Branch2b, for example 1xHxW, BN, ReLU.
        x = self.conv_b(x)
        if self.norm_b is not None:
            x = self.norm_b(x)
        if self.act_b is not None:
            x = self.act_b(x)

        # Branch2c, for example 1x1x1, BN.
        x = self.conv_c(x)
        if self.norm_c is not None:
            x = self.norm_c(x)
        return x


class ResStage(nn.Module):
    """
    ResStage composes sequential blocks that make up a ResNet. These blocks could be,
    for example, Residual blocks, Non-Local layers, or Squeeze-Excitation layers.

    ::


                                        Input
                                           ↓
                                       ResBlock
                                           ↓
                                           .
                                           .
                                           .
                                           ↓
                                       ResBlock

    The builder can be found in `create_res_stage`.
    """

    def __init__(self, res_blocks: nn.ModuleList) -> nn.Module:
        """
        Args:
            res_blocks (torch.nn.module_list): ResBlock module(s).
        """
        super().__init__()
        self.res_blocks = res_blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _, res_block in enumerate(self.res_blocks):
            x = res_block(x)
        return x


def init_net_weights(
    model: nn.Module,
    init_std: float = 0.01,
    style: str = "resnet",
) -> None:
    """
    Performs weight initialization. Options include ResNet style weight initialization
    and transformer style weight initialization.

    Args:
        model (nn.Module): Model to be initialized.
        init_std (float): The expected standard deviation for initialization.
        style (str): Options include "resnet" and "vit".
    """
    assert style in ["resnet", "vit"]
    if style == "resnet":
        return _init_resnet_weights(model, init_std)
    else:
        raise NotImplementedError


def _init_resnet_weights(model: nn.Module, fc_init_std: float = 0.01) -> None:
    """
    Performs ResNet style weight initialization. That is, recursively initialize the
    given model in the following way for each type:
        Conv - Follow the initialization of kaiming_normal:
            https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_normal_
        BatchNorm - Set weight and bias of last BatchNorm at every residual bottleneck
            to 0.
        Linear - Set weight to 0 mean Gaussian with std deviation fc_init_std and bias
            to 0.
    Args:
        model (nn.Module): Model to be initialized.
        fc_init_std (float): the expected standard deviation for fully-connected layer.
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            """
            Follow the initialization method proposed in:
            {He, Kaiming, et al.
            "Delving deep into rectifiers: Surpassing human-level
            performance on imagenet classification."
            arXiv preprint arXiv:1502.01852 (2015)}
            """
            c2_msra_fill(m)
        elif isinstance(m, nn.modules.batchnorm._NormBase):
            if m.weight is not None:
                if hasattr(m, "block_final_bn") and m.block_final_bn:
                    m.weight.data.fill_(0.0)
                else:
                    m.weight.data.fill_(1.0)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, SpatioTemporalClsPositionalEncoding):
            for weights in m.parameters():
                nn.init.trunc_normal_(weights, std=0.02)
    return model


class Net(nn.Module):
    """
    Build a general Net models with a list of blocks for video recognition.

    ::

                                         Input
                                           ↓
                                         Block 1
                                           ↓
                                           .
                                           .
                                           .
                                           ↓
                                         Block N
                                           ↓

    The ResNet builder can be found in `create_resnet`.
    """

    def __init__(
        self,
        *,
        reserve_index: list = [True, True, True, True, True, True, True],
        dropblock: None,
        blocks: nn.ModuleList,
    ) -> None:
        """
        Args:
            blocks (torch.nn.module_list): the list of block modules.
        """
        super().__init__()
        assert blocks is not None
        self.blocks = blocks
        self.reserve_index = reserve_index
        self.dropblock = dropblock
        init_net_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        middel_results = []
        for index, block in enumerate(self.blocks):
            x = block(x)
            if self.dropblock:
                x = self.dropblock(x)
            if self.reserve_index[index]:
                middel_results.append(x.clone())
        return middel_results


class ResNetBasicHead(nn.Module):
    """
    ResNet basic head. This layer performs an optional pooling operation followed by an
    optional dropout, a fully-connected projection, an optional activation layer and a
    global spatiotemporal averaging.

    ::

                                        Pool3d
                                           ↓
                                        Dropout
                                           ↓
                                       Projection
                                           ↓
                                       Activation
                                           ↓
                                       Averaging

    The builder can be found in `create_res_basic_head`.
    """

    def __init__(
        self,
        pool: nn.Module = None,
        dropout: nn.Module = None,
        proj: nn.Module = None,
        activation: nn.Module = None,
        output_pool: nn.Module = None,
    ) -> None:
        """
        Args:
            pool (torch.nn.modules): pooling module.
            dropout(torch.nn.modules): dropout module.
            proj (torch.nn.modules): project module.
            activation (torch.nn.modules): activation module.
            output_pool (torch.nn.Module): pooling module for output.
        """
        super(ResNetBasicHead, self).__init__()
        set_attributes(self, locals())
        # assert self.proj is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Performs pooling.
        if self.pool is not None:
            x = self.pool(x)
        # Performs dropout.
        if self.dropout is not None:
            x = self.dropout(x)
        # Performs projection.
        if self.proj is not None:
            x = x.permute((0, 2, 3, 4, 1))
            x = self.proj(x)
            x = x.permute((0, 4, 1, 2, 3))
        # Performs activation.
        if self.activation is not None:
            x = self.activation(x)

        if self.output_pool is not None:
            # Performs global averaging.
            x = self.output_pool(x)
            x = x.view(x.shape[0], -1)
        return x


def set_attributes(self, params: List[object] = None) -> None:
    """
    An utility function used in classes to set attributes from the input list of parameters.
    Args:
        params (list): list of parameters.
    """
    if params:
        for k, v in params.items():
            if k != "self":
                setattr(self, k, v)


def round_width(width, multiplier, min_width=8, divisor=8, ceil=False):
    """
    Round width of filters based on width multiplier
    Args:
        width (int): the channel dimensions of the input.
        multiplier (float): the multiplication factor.
        min_width (int): the minimum width after multiplication.
        divisor (int): the new width should be dividable by divisor.
        ceil (bool): If True, use ceiling as the rounding method.
    """
    if not multiplier:
        return width

    width *= multiplier
    min_width = min_width or divisor
    if ceil:
        width_out = max(min_width, int(math.ceil(width / divisor)) * divisor)
    else:
        width_out = max(min_width, int(width + divisor / 2) // divisor * divisor)
    if width_out < 0.9 * width:
        width_out += divisor
    return int(width_out)


def round_repeats(repeats, multiplier):
    """
    Round number of layers based on depth multiplier.
    """
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


class Swish(nn.Module):
    """
    Wrapper for the Swish activation function.
    """

    def forward(self, x):
        return SwishFunction.apply(x)


class SwishFunction(torch.autograd.Function):
    """
    Implementation of the Swish activation function: x * sigmoid(x).

    Searching for activation functions. Ramachandran, Prajit and Zoph, Barret
    and Le, Quoc V. 2017
    """

    @staticmethod
    def forward(ctx, x):
        result = x * torch.sigmoid(x)
        ctx.save_for_backward(x)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid_x = torch.sigmoid(x)
        return grad_output * (sigmoid_x * (1 + x * (1 - sigmoid_x)))


class Conv2plus1d(nn.Module):
    """
    Implementation of 2+1d Convolution by factorizing 3D Convolution into an 1D temporal
    Convolution and a 2D spatial Convolution with Normalization and Activation module
    in between:

    ::

                        Conv_t (or Conv_xy if conv_xy_first = True)
                                           ↓
                                     Normalization
                                           ↓
                                       Activation
                                           ↓
                        Conv_xy (or Conv_t if conv_xy_first = True)

    The 2+1d Convolution is used to build the R(2+1)D network.
    """

    def __init__(
        self,
        *,
        conv_t: nn.Module = None,
        norm: nn.Module = None,
        activation: nn.Module = None,
        conv_xy: nn.Module = None,
        conv_xy_first: bool = False,
    ) -> None:
        """
        Args:
            conv_t (torch.nn.modules): temporal convolution module.
            norm (torch.nn.modules): normalization module.
            activation (torch.nn.modules): activation module.
            conv_xy (torch.nn.modules): spatial convolution module.
            conv_xy_first (bool): If True, spatial convolution comes before temporal conv
        """
        super().__init__()
        set_attributes(self, locals())
        assert self.conv_t is not None
        assert self.conv_xy is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_xy(x) if self.conv_xy_first else self.conv_t(x)
        x = self.norm(x) if self.norm else x
        x = self.activation(x) if self.activation else x
        x = self.conv_t(x) if self.conv_xy_first else self.conv_xy(x)
        return x


def create_x3d_stem(
    *,
    # Conv configs.
    in_channels: int,
    out_channels: int,
    conv_kernel_size: Tuple[int] = (5, 3, 3),
    conv_stride: Tuple[int] = (1, 2, 2),
    conv_padding: Tuple[int] = (2, 1, 1),
    # BN configs.
    norm: Callable = nn.BatchNorm3d,
    norm_eps: float = 1e-5,
    norm_momentum: float = 0.1,
    # Activation configs.
    activation: Callable = nn.ReLU,
) -> nn.Module:
    """
    Creates the stem layer for X3D. It performs spatial Conv, temporal Conv, BN, and Relu.

    ::

                                        Conv_xy
                                           ↓
                                        Conv_t
                                           ↓
                                     Normalization
                                           ↓
                                       Activation

    Args:
        in_channels (int): input channel size of the convolution.
        out_channels (int): output channel size of the convolution.
        conv_kernel_size (tuple): convolutional kernel size(s).
        conv_stride (tuple): convolutional stride size(s).
        conv_padding (tuple): convolutional padding size(s).

        norm (callable): a callable that constructs normalization layer, options
            include nn.BatchNorm3d, None (not performing normalization).
        norm_eps (float): normalization epsilon.
        norm_momentum (float): normalization momentum.

        activation (callable): a callable that constructs activation layer, options
            include: nn.ReLU, nn.Softmax, nn.Sigmoid, and None (not performing
            activation).

    Returns:
        (nn.Module): X3D stem layer.
    """
    conv_xy_module = nn.Conv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=(1, conv_kernel_size[1], conv_kernel_size[2]),
        stride=(1, conv_stride[1], conv_stride[2]),
        padding=(0, conv_padding[1], conv_padding[2]),
        bias=False,
    )
    conv_t_module = nn.Conv3d(
        in_channels=out_channels,
        out_channels=out_channels,
        kernel_size=(conv_kernel_size[0], 1, 1),
        stride=(conv_stride[0], 1, 1),
        padding=(conv_padding[0], 0, 0),
        bias=False,
        groups=out_channels,
    )
    stacked_conv_module = Conv2plus1d(
        conv_t=conv_xy_module,
        norm=None,
        activation=None,
        conv_xy=conv_t_module,
    )

    norm_module = (
        None
        if norm is None
        else norm(num_features=out_channels, eps=norm_eps, momentum=norm_momentum)
    )
    activation_module = None if activation is None else activation()

    return ResNetBasicStem(
        conv=stacked_conv_module,
        norm=norm_module,
        activation=activation_module,
        pool=None,
    )


def create_fusion(
    *,
    # Convolution configs.
    dim_in: int,
    dim_inner: int,
    dim_out: int,
    fuse_type: str = "add",
    conv_kernel_size: Tuple[int] = (3, 3, 3),
    conv_stride: Tuple[int] = (1, 2, 2),
    # Norm configs.
    norm: Callable = nn.BatchNorm3d,
    norm_eps: float = 1e-5,
    norm_momentum: float = 0.1,
    se_ratio: float = 0.0625,
    # Activation configs.
    activation: Callable = nn.ReLU,
    inner_act: Callable = Swish,
    **kwargs,
) -> nn.Module:

    return CVFM(dim_in, dim_out, **kwargs)


class CVFM(nn.Module):
    def __init__(
        self,
        num_classes: int,
        att_size: Tuple,
        dim_in: int = 192,
        dim_inner: int = 432,
        dim_out: int = 2048,
        # Pooling configs.
        pool_act: Callable = nn.ReLU,
        pool_kernel_size: Tuple[int] = (16, 8, 8),
        # BN configs.
        norm: Callable = nn.BatchNorm3d,
        norm_eps: float = 1e-5,
        norm_momentum: float = 0.1,
        bn_lin5_on=False,
        # Dropout configs.
        dropout_rate: float = 0.5,
        # Activation configs.
        activation: bool = None,
        preconv=None,
        # Output configs.
        output_with_global_average: bool = False,
        mlp_dropout_rate: float = 0.05,
        num_heads: int = 4,
        expand_dim: int = 4,
    ):
        super(CVFM, self).__init__()
        self.dim_in = dim_in * expand_dim
        self.dim_out = dim_out
        self.dim_inner = dim_inner
        if preconv is not None:
            self.preconv_c = copy.deepcopy(preconv)
            self.preconv_s = copy.deepcopy(preconv)
        else:
            self.preconv_c = None
            self.preconv_s = None

        if expand_dim >= 2:
            self.dim_inner = self.dim_inner * expand_dim
        self.gamma = nn.Parameter(torch.Tensor([0]))
        pre_conv_module = nn.Conv3d(
            in_channels=self.dim_in,
            out_channels=self.dim_inner,
            kernel_size=(1, 1, 1),
            bias=False,
        )
        pre_norm_module = norm(
            num_features=self.dim_inner, eps=norm_eps, momentum=norm_momentum
        )
        pre_act_module = None if pool_act is None else pool_act()

        if pool_kernel_size is None:
            pool_module = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            pool_module = nn.AvgPool3d(pool_kernel_size, stride=1)

        post_conv_module = nn.Conv3d(
            in_channels=self.dim_inner,
            out_channels=self.dim_out,
            kernel_size=(1, 1, 1),
            bias=False,
        )

        if bn_lin5_on:
            post_norm_module = norm(
                num_features=self.dim_out, eps=norm_eps, momentum=norm_momentum
            )
        else:
            post_norm_module = None
        post_act_module = None if pool_act is None else pool_act()

        projected_pool_module = ProjectedPool(
            pre_conv=pre_conv_module,
            pre_norm=pre_norm_module,
            pre_act=pre_act_module,
            pool=pool_module,
            post_conv=post_conv_module,
            post_norm=post_norm_module,
            post_act=post_act_module,
        )

        if activation is None:
            activation_module = None
        elif activation == nn.Softmax:
            activation_module = activation(dim=1)
        elif activation == nn.Sigmoid:
            activation_module = activation()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation" "function.".format(activation)
            )

        if output_with_global_average:
            output_pool = nn.AdaptiveAvgPool3d(1)
            proj_linear = nn.Linear(self.dim_out, num_classes, bias=True)
        else:
            proj_linear = None
            output_pool = None
        self.fuse_conv = ResNetBasicHead(
            proj=proj_linear,
            activation=activation_module,
            pool=projected_pool_module,
            dropout=nn.Dropout(dropout_rate) if dropout_rate > 0 else None,
            output_pool=output_pool,
        )
        self.CVAFM = MultiScaleAttention(
            dim_in=dim_in,
            dropout_rate=mlp_dropout_rate,
            num_heads=num_heads,
            expand_dim=expand_dim,
            att_size=att_size,
        )
        self.angle = torch.nn.Parameter(torch.tensor(0, dtype=torch.float32))
        self.pi = torch.tensor(math.pi, dtype=torch.float32)

    def forward(self, cx, sx) -> torch.Tensor:
        if self.preconv_c is not None and self.preconv_c is not None:
            cx[-2] = self.preconv_c(cx[-2])
            sx[-2] = self.preconv_s(sx[-2])
        tmp = torch.cos(self.pi * torch.sigmoid(self.angle)) + 1e-8
        angle_tensor = (cx[-2] / tmp + sx[-2] / (1 - tmp)) * 0.5
        pythagorean_tensor = torch.sqrt(cx[-2] ** 2 + sx[-2] ** 2)
        out = angle_tensor + F.sigmoid(self.gamma) * pythagorean_tensor
        cvfm_tensor, _, _ = self.CVAFM(out, cx[-2], sx[-2])
        out = self.fuse_conv(cvfm_tensor)
        return out, dict(
            angle_tensor=angle_tensor,
            pythagorean_tensor=pythagorean_tensor,
            cvfm_tensor=cvfm_tensor,
        )


def create_conv_patch_embed(
    in_channels: int,
    out_channels: int,
    conv_kernel_size: Tuple[int] = (1, 16, 16),
    conv_stride: Tuple[int] = (1, 4, 4),
    conv_padding: Tuple[int] = (1, 7, 7),
    conv_bias: bool = True,
    conv: Callable = nn.Conv3d,
) -> nn.Module:
    """
    Creates the transformer basic patch embedding. It performs Convolution, flatten and
    transpose.

    ::

                                        Conv3d
                                           ↓
                                        flatten
                                           ↓
                                       transpose

    Args:
        in_channels (int): input channel size of the convolution.
        out_channels (int): output channel size of the convolution.
        conv_kernel_size (tuple): convolutional kernel size(s).
        conv_stride (tuple): convolutional stride size(s).
        conv_padding (tuple): convolutional padding size(s).
        conv_bias (bool): convolutional bias. If true, adds a learnable bias to the
            output.
        conv (callable): Callable used to build the convolution layer.

    Returns:
        (nn.Module): transformer patch embedding layer.
    """
    conv_module = conv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=conv_kernel_size,
        stride=conv_stride,
        padding=conv_padding,
        bias=conv_bias,
    )
    return PatchEmbed(patch_model=conv_module)


class PatchEmbed(nn.Module):
    """
    Transformer basic patch embedding module. Performs patchifying input, flatten and
    and transpose.

    ::

                                       PatchModel
                                           ↓
                                        flatten
                                           ↓
                                       transpose

    The builder can be found in `create_patch_embed`.

    """

    def __init__(
        self,
        *,
        patch_model: nn.Module = None,
    ) -> None:
        super().__init__()
        set_attributes(self, locals())
        assert self.patch_model is not None

    def forward(self, x) -> torch.Tensor:
        x = self.patch_model(x)
        # B C (T) H W -> B (T)HW C
        return x.flatten(2).transpose(1, 2)


class Mlp(nn.Module):
    """
    A MLP block that contains two linear layers with a normalization layer. The MLP
    block is used in a transformer model after the attention block.

    ::

                         Linear (in_features, hidden_features)
                                           ↓
                                 Normalization (act_layer)
                                           ↓
                                Dropout (p=dropout_rate)
                                           ↓
                         Linear (hidden_features, out_features)
                                           ↓
                                Dropout (p=dropout_rate)
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable = nn.GELU,
        dropout_rate: float = 0.0,
        bias_on: bool = True,
    ) -> None:
        """
        Args:
            in_features (int): Input feature dimension.
            hidden_features (Optional[int]): Hidden feature dimension. By default,
                hidden feature is set to input feature dimension.
            out_features (Optional[int]): Output feature dimension. By default, output
                features dimension is set to input feature dimension.
            act_layer (Callable): Activation layer used after the first linear layer.
            dropout_rate (float): Dropout rate after each linear layer. Dropout is not used
                by default.
        """
        super().__init__()
        self.dropout_rate = dropout_rate
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias_on)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias_on)

        if self.dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (tensor): Input tensor.
        """
        x = self.fc1(x)
        x = self.act(x)
        if self.dropout_rate > 0.0:
            x = self.dropout(x)
        x = self.fc2(x)
        if self.dropout_rate > 0.0:
            x = self.dropout(x)
        return x


def drop_path(
    x: torch.Tensor, drop_prob: float = 0.0, training: bool = False
) -> torch.Tensor:
    """
    Stochastic Depth per sample.

    Args:
        x (tensor): Input tensor.
        drop_prob (float): Probability to apply drop path.
        training (bool): If True, apply drop path to input. Otherwise (tesing), return input.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output


def _init_vit_weights(model: nn.Module, trunc_normal_std: float = 0.02) -> None:
    """
    Weight initialization for vision transformers.

    Args:
        model (nn.Module): Model to be initialized.
        trunc_normal_std (float): the expected standard deviation for fully-connected
            layer and ClsPositionalEncoding.
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=trunc_normal_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, SpatioTemporalClsPositionalEncoding):
            for weights in m.parameters():
                nn.init.trunc_normal_(weights, std=trunc_normal_std)


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        """
        Args:
            drop_prob (float): Probability to apply drop path.
        """
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (tensor): Input tensor.
        """
        return drop_path(x, self.drop_prob, self.training)


class SpatioTemporalClsPositionalEncoding(nn.Module):
    """
    Add a cls token and apply a spatiotemporal encoding to a tensor.
    """

    def __init__(
        self,
        embed_dim: int,
        patch_embed_shape: Tuple[int, int, int],
        sep_pos_embed: bool = False,
        has_cls: bool = True,
    ) -> None:
        """
        Args:
            embed_dim (int): Embedding dimension for input sequence.
            patch_embed_shape (Tuple): The number of patches in each dimension
                (T, H, W) after patch embedding.
            sep_pos_embed (bool): If set to true, one positional encoding is used for
                spatial patches and another positional encoding is used for temporal
                sequence. Otherwise, only one positional encoding is used for all the
                patches.
            has_cls (bool): If set to true, a cls token is added in the beginning of each
                input sequence.
        """
        super().__init__()
        assert (
            len(patch_embed_shape) == 3
        ), "Patch_embed_shape should be in the form of (T, H, W)."
        self.cls_embed_on = has_cls
        self.sep_pos_embed = sep_pos_embed
        self.patch_embed_shape = patch_embed_shape
        self.num_spatial_patch = patch_embed_shape[1] * patch_embed_shape[2]
        self.num_temporal_patch = patch_embed_shape[0]

        if self.cls_embed_on:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            num_patches = self.num_spatial_patch * self.num_temporal_patch + 1
        else:
            self.cls_token = torch.tensor(0)
            num_patches = self.num_spatial_patch * self.num_temporal_patch

        if self.sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(1, self.num_spatial_patch, embed_dim)
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, self.num_temporal_patch, embed_dim)
            )
            if self.cls_embed_on:
                self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
            else:
                self.pos_embed_class = torch.tensor([])  # for torchscriptability
            self.pos_embed = torch.tensor([])

        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            # Placeholders for torchscriptability, won't be used
            self.pos_embed_spatial = torch.tensor([])
            self.pos_embed_temporal = torch.tensor([])
            self.pos_embed_class = torch.tensor([])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor.
        """
        B, N, C = x.shape
        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.num_temporal_patch, 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.num_spatial_patch,
                dim=1,
            )
            if self.cls_embed_on:
                pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
            x = x + pos_embed
        else:
            x = x + self.pos_embed

        return x


class MultiScaleAttention(nn.Module):
    """
    Implementation of a multiscale attention block. Compare to a conventional attention
    block, a multiscale attention block optionally supports pooling (either
    before or after qkv projection). If pooling is not used, a multiscale attention
    block is equivalent to a conventional attention block.

    ::
                                   Input
                                     |
                    |----------------|-----------------|
                    ↓                ↓                 ↓
                  Linear           Linear            Linear
                    &                &                 &
                 Pool (Q)         Pool (K)          Pool (V)
                    → -------------- ←                 |
                             ↓                         |
                       MatMul & Scale                  |
                             ↓                         |
                          Softmax                      |
                             → ----------------------- ←
                                         ↓
                                   MatMul & Scale
                                         ↓
                                      DropOut
    """

    _version = 2
    """
    add SpatioTemporalClsPositionalEncoding by 4
    """

    def __init__(
        self,
        dim_in: int,
        att_size: Tuple[int] = (8, 16, 16),
        expand_dim: int = 4,
        num_heads: int = 4,
        qkv_bias: bool = False,
        dropout_rate: float = 0.05,
        mpl_ratio: int = 4,
        droppath_rate: float = 0,
        kernel_q: tuple = (3, 3, 3),
        kernel_kv: tuple = (3, 3, 3),
        stride_q: tuple = (1, 2, 2),
        stride_kv: tuple = (1, 4, 4),
        norm_layer: Callable = nn.LayerNorm,
        has_cls_embed: bool = False,
        pool_mode: str = "conv",
        pool_first: bool = False,
        residual_pool: bool = True,
        depthwise_conv: bool = True,
        bias_on: bool = True,
        separate_qkv: bool = True,
    ) -> None:
        """
        Args:
            dim (int): Input feature dimension.
            num_heads (int): Number of heads in the attention layer.
            qkv_bias (bool): If set to False, the qkv layer will not learn an additive
                bias. Default: False.
            dropout_rate (float): Dropout rate.
            kernel_q (_size_3_t): Pooling kernel size for q. If both pooling kernel
                size and pooling stride size are 1 for all the dimensions, pooling is
                disabled.
            kernel_kv (_size_3_t): Pooling kernel size for kv. If both pooling kernel
                size and pooling stride size are 1 for all the dimensions, pooling is
                disabled.
            stride_q (_size_3_t): Pooling kernel stride for q.
            stride_kv (_size_3_t): Pooling kernel stride for kv.
            norm_layer (nn.Module): Normalization layer used after pooling.
            has_cls_embed (bool): If set to True, the first token of the input tensor
                should be a cls token. Otherwise, the input tensor does not contain a
                cls token. Pooling is not applied to the cls token.
            pool_mode (str): Pooling mode. Option includes "conv" (learned pooling), "avg"
                (average pooling), and "max" (max pooling).
            pool_first (bool): If set to True, pool is applied before qkv projection.
                Otherwise, pool is applied after qkv projection. Default: False.
            residual_pool (bool): If set to True, use Improved Multiscale Vision
                Transformer's pooling residual connection.
            depthwise_conv (bool): Whether use depthwise or full convolution for pooling.
            bias_on (bool): Whether use biases for linear layers.
            separate_qkv (bool): Whether to use separate or one layer for qkv projections.
        """

        super().__init__()
        assert pool_mode in ["conv", "avg", "max"]

        self.pool_first = pool_first
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        dim = int(dim_in * (expand_dim / 2) * 2)
        self.dim_out = dim
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.has_cls_embed = has_cls_embed
        self.residual_pool = residual_pool
        self.separate_qkv = separate_qkv
        padding_q = [int(q // 2) for q in kernel_q]
        padding_kv = [int(kv // 2) for kv in kernel_kv]
        kernel_skip = [s + 1 if s > 1 else s for s in stride_q]
        stride_skip = stride_q
        padding_skip = [int(skip // 2) for skip in kernel_skip]
        self.out_norm = nn.BatchNorm3d(self.dim_out)
        self.norm2_mlpc = norm_layer(dim)
        self.norm2_mlps = norm_layer(dim)
        self.norm2_is_batchnorm_1d = isinstance(self.norm2_mlps, nn.BatchNorm1d)
        self.norm2_is_batchnorm_1d = isinstance(self.norm2_mlpc, nn.BatchNorm1d)

        self.mlp_c = Mlp(
            in_features=dim,
            hidden_features=dim * mpl_ratio,
            out_features=dim,
            act_layer=nn.GELU,
            dropout_rate=dropout_rate,
            bias_on=True,
        )
        self.drop_path_mlpc = (
            DropPath(droppath_rate) if droppath_rate > 0.0 else nn.Identity()
        )
        if dim != self.dim_out:
            self.proj_mlpc = nn.Linear(dim, self.dim_out, bias=bias_on)
        else:
            self.proj_mlpc = nn.Identity()

        self.pool_skip_mlpc = (
            nn.MaxPool3d(kernel_skip, stride_skip, padding_skip, ceil_mode=False)
            if len(stride_skip) > 0 and np.prod(stride_skip) > 1
            else None
        )
        if dim_in == dim:
            self.pool_skip_mlpf = (
                nn.MaxPool3d(kernel_skip, stride_skip, padding_skip, ceil_mode=False)
                if len(stride_skip) > 0 and np.prod(stride_skip) > 1
                else None
            )
        else:
            self.pool_skip_mlpf = nn.Sequential(
                nn.Conv3d(
                    dim_in,
                    dim,
                    kernel_skip,
                    stride_skip,
                    padding_skip,
                ),
                nn.BatchNorm3d(dim),
                nn.GELU(),
            )
        self._attention_pool_mlpc = _AttentionPool(
            self.pool_skip_mlpc, has_cls_embed=self.has_cls_embed, norm=None
        )

        self.mlp_s = Mlp(
            in_features=dim,
            hidden_features=dim * mpl_ratio,
            out_features=dim,
            act_layer=nn.GELU,
            dropout_rate=dropout_rate,
            bias_on=True,
        )
        self.drop_path_mlps = (
            DropPath(droppath_rate) if droppath_rate > 0.0 else nn.Identity()
        )
        if dim != self.dim_out:
            self.proj_mlps = nn.Linear(dim, self.dim_out, bias=bias_on)
        else:
            self.proj_mlps = nn.Identity()

        self.pool_skip_mlps = (
            nn.MaxPool3d(kernel_skip, stride_skip, padding_skip, ceil_mode=False)
            if len(stride_skip) > 0 and np.prod(stride_skip) > 1
            else None
        )
        self._attention_pool_mlps = _AttentionPool(
            self.pool_skip_mlps, has_cls_embed=self.has_cls_embed, norm=None
        )

        self.cls_positional_encoding_fxc = SpatioTemporalClsPositionalEncoding(
            embed_dim=dim,
            patch_embed_shape=att_size,
            sep_pos_embed=True,
            has_cls=False,
        )
        self.cls_positional_encoding_fxs = SpatioTemporalClsPositionalEncoding(
            embed_dim=dim,
            patch_embed_shape=att_size,
            sep_pos_embed=True,
            has_cls=False,
        )
        self.cls_positional_encoding_c = SpatioTemporalClsPositionalEncoding(
            embed_dim=dim,
            patch_embed_shape=att_size,
            sep_pos_embed=True,
            has_cls=False,
        )
        self.cls_positional_encoding_s = SpatioTemporalClsPositionalEncoding(
            embed_dim=dim,
            patch_embed_shape=att_size,
            sep_pos_embed=True,
            has_cls=False,
        )

        self.patch_fc = create_conv_patch_embed(
            in_channels=dim // expand_dim,
            out_channels=dim,
            conv_kernel_size=(3, 3, 3),
            conv_stride=(1, 1, 1),
            conv_padding=(1, 1, 1),
        )
        self.patch_fs = create_conv_patch_embed(
            in_channels=dim // expand_dim,
            out_channels=dim,
            conv_kernel_size=(3, 3, 3),
            conv_stride=(1, 1, 1),
            conv_padding=(1, 1, 1),
        )
        self.patch_c = create_conv_patch_embed(
            in_channels=dim // expand_dim,
            out_channels=dim,
            conv_kernel_size=(3, 3, 3),
            conv_stride=(1, 1, 1),
            conv_padding=(1, 1, 1),
        )
        self.patch_s = create_conv_patch_embed(
            in_channels=dim // expand_dim,
            out_channels=dim,
            conv_kernel_size=(3, 3, 3),
            conv_stride=(1, 1, 1),
            conv_padding=(1, 1, 1),
        )
        self.qfc = nn.Linear(dim, dim, bias=qkv_bias)
        self.kc = nn.Linear(dim, dim, bias=qkv_bias)
        self.vc = nn.Linear(dim, dim, bias=qkv_bias)

        self.qfs = nn.Linear(dim, dim, bias=qkv_bias)
        self.ks = nn.Linear(dim, dim, bias=qkv_bias)
        self.vs = nn.Linear(dim, dim, bias=qkv_bias)

        self.proj_c = nn.Linear(dim, dim, bias=True if bias_on else False)
        self.proj_s = nn.Linear(dim, dim, bias=True if bias_on else False)

        if dropout_rate > 0.0:
            self.proj_drop_c = nn.Dropout(dropout_rate)
            self.proj_drop_s = nn.Dropout(dropout_rate)
        else:
            self.proj_drop_c = nn.Identity()
            self.proj_drop_s = nn.Identity()

        # Skip pooling with kernel and stride size of (1, 1, 1).
        if (
            kernel_q is not None
            and self._prod(kernel_q) == 1
            and self._prod(stride_q) == 1
        ):
            kernel_q = None
        if (
            kernel_kv is not None
            and self._prod(kernel_kv) == 1
            and self._prod(stride_kv) == 1
        ):
            kernel_kv = None

        self.pool_qfc = (
            nn.Conv3d(
                head_dim,
                head_dim,
                kernel_q,
                stride=stride_q,
                padding=padding_q,
                groups=head_dim if depthwise_conv else 1,
                bias=False,
            )
            if kernel_q is not None
            else None
        )
        self.norm_qfc = norm_layer(head_dim) if kernel_q is not None else None

        self.pool_qfs = (
            nn.Conv3d(
                head_dim,
                head_dim,
                kernel_q,
                stride=stride_q,
                padding=padding_q,
                groups=head_dim if depthwise_conv else 1,
                bias=False,
            )
            if kernel_q is not None
            else None
        )
        self.norm_qfs = norm_layer(head_dim) if kernel_q is not None else None

        self.pool_kc = (
            nn.Conv3d(
                head_dim,
                head_dim,
                kernel_kv,
                stride=stride_kv,
                padding=padding_kv,
                groups=head_dim if depthwise_conv else 1,
                bias=False,
            )
            if kernel_kv is not None
            else None
        )
        self.norm_kc = norm_layer(head_dim) if kernel_kv is not None else None

        self.pool_vc = (
            nn.Conv3d(
                head_dim,
                head_dim,
                kernel_kv,
                stride=stride_kv,
                padding=padding_kv,
                groups=head_dim if depthwise_conv else 1,
                bias=False,
            )
            if kernel_kv is not None
            else None
        )
        self.norm_vc = norm_layer(head_dim) if kernel_kv is not None else None

        self.pool_ks = (
            nn.Conv3d(
                head_dim,
                head_dim,
                kernel_kv,
                stride=stride_kv,
                padding=padding_kv,
                groups=head_dim if depthwise_conv else 1,
                bias=False,
            )
            if kernel_kv is not None
            else None
        )
        self.norm_ks = norm_layer(head_dim) if kernel_kv is not None else None
        self.pool_vs = (
            nn.Conv3d(
                head_dim,
                head_dim,
                kernel_kv,
                stride=stride_kv,
                padding=padding_kv,
                groups=head_dim if depthwise_conv else 1,
                bias=False,
            )
            if kernel_kv is not None
            else None
        )
        self.norm_vs = norm_layer(head_dim) if kernel_kv is not None else None

        # Will not be used if `separate_qkv == True`
        self._attention_pool_qfc = _AttentionPool(
            self.pool_qfc,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_q if hasattr(self, "norm_q") else None,
        )
        self._attention_pool_kc = _AttentionPool(
            self.pool_kc,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_k if hasattr(self, "norm_k") else None,
        )
        self._attention_pool_vc = _AttentionPool(
            self.pool_vc,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_v if hasattr(self, "norm_v") else None,
        )

        self._attention_pool_qfs = _AttentionPool(
            self.pool_qfs,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_q if hasattr(self, "norm_q") else None,
        )
        self._attention_pool_ks = _AttentionPool(
            self.pool_ks,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_k if hasattr(self, "norm_k") else None,
        )
        self._attention_pool_vs = _AttentionPool(
            self.pool_vs,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_v if hasattr(self, "norm_v") else None,
        )

        self.apply(_init_vit_weights)

    def _qkv_proj(
        self,
        q: torch.Tensor,
        q_size: int,
        k: torch.Tensor,
        k_size: int,
        v: torch.Tensor,
        v_size: int,
        batch_size: int,
        chan_size: int,
        state="s",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if state == "c":
            q = (
                self.qfc(q)
                .reshape(
                    batch_size, q_size, self.num_heads, chan_size // self.num_heads
                )
                .permute(0, 2, 1, 3)
            )
            k = (
                self.kc(k)
                .reshape(
                    batch_size, k_size, self.num_heads, chan_size // self.num_heads
                )
                .permute(0, 2, 1, 3)
            )
            v = (
                self.vc(v)
                .reshape(
                    batch_size, v_size, self.num_heads, chan_size // self.num_heads
                )
                .permute(0, 2, 1, 3)
            )
        if state == "s":
            q = (
                self.qfs(q)
                .reshape(
                    batch_size, q_size, self.num_heads, chan_size // self.num_heads
                )
                .permute(0, 2, 1, 3)
            )
            k = (
                self.ks(k)
                .reshape(
                    batch_size, k_size, self.num_heads, chan_size // self.num_heads
                )
                .permute(0, 2, 1, 3)
            )
            v = (
                self.vs(v)
                .reshape(
                    batch_size, v_size, self.num_heads, chan_size // self.num_heads
                )
                .permute(0, 2, 1, 3)
            )
        return q, k, v

    def _qkv_pool(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        thw_shape: List[int],
        state="c",
    ) -> Tuple[
        torch.Tensor, List[int], torch.Tensor, List[int], torch.Tensor, List[int]
    ]:
        if state == "c":
            q, q_shape = self._attention_pool_qfc(q, thw_shape)
            k, k_shape = self._attention_pool_kc(k, thw_shape)
            v, v_shape = self._attention_pool_vc(v, thw_shape)
        else:
            q, q_shape = self._attention_pool_qfs(q, thw_shape)
            k, k_shape = self._attention_pool_ks(k, thw_shape)
            v, v_shape = self._attention_pool_vs(v, thw_shape)
        return q, q_shape, k, k_shape, v, v_shape

    def _get_qkv_length(
        self,
        q_shape: List[int],
        k_shape: List[int],
        v_shape: List[int],
    ) -> Tuple[int, int, int]:
        q_N = self._prod(q_shape) + 1 if self.has_cls_embed else self._prod(q_shape)
        k_N = self._prod(k_shape) + 1 if self.has_cls_embed else self._prod(k_shape)
        v_N = self._prod(v_shape) + 1 if self.has_cls_embed else self._prod(v_shape)
        return q_N, k_N, v_N

    def _prod(self, shape: List[int]) -> int:
        """Torchscriptable version of `numpy.prod`. Note that `_prod([]) == 1`"""
        p: int = 1
        for dim in shape:
            p *= dim
        return p

    def _reshape_qkv_to_seq(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_N: int,
        v_N: int,
        k_N: int,
        B: int,
        C: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = q.permute(0, 2, 1, 3).reshape(B, q_N, C)
        v = v.permute(0, 2, 1, 3).reshape(B, v_N, C)
        k = k.permute(0, 2, 1, 3).reshape(B, k_N, C)
        return q, k, v

    def _back_to_grid(
        self,
        x: torch.Tensor,
        B: int,
        C: int,
        DHW: tuple[int],
    ) -> torch.Tensor:
        x = x.reshape(
            B,
            C,
            DHW[0],
            DHW[1],
            DHW[2],
        ).contiguous()
        return x

    def forward(
        self,
        fx: torch.Tensor,
        cx: torch.Tensor,
        sx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor.
            thw_shape (List): The shape of the input tensor (before flattening).
        """
        fx_raw = fx.clone()

        fxc = self.patch_fc(fx)
        fxs = self.patch_fs(fx)
        cx = self.patch_c(cx)
        sx = self.patch_s(sx)

        fxc = self.cls_positional_encoding_fxc(fxc)
        fxs = self.cls_positional_encoding_fxs(fxs)
        cx = self.cls_positional_encoding_c(cx)
        sx = self.cls_positional_encoding_s(sx)

        B, N, C = fxc.shape
        qfc, kc, vc = self._qkv_proj(fxc, N, cx, N, cx, N, B, C, state="c")
        qfc, qfc_shape, kc, kc_shape, vc, vc_shape = self._qkv_pool(
            qfc, kc, vc, fx_raw.shape[2:]
        )

        qfs, ks, vs = self._qkv_proj(fxs, N, sx, N, sx, N, B, C, state="s")
        qfs, qfs_shape, ks, ks_shape, vs, vs_shape = self._qkv_pool(
            qfs, ks, vs, fx_raw.shape[2:]
        )

        attnc = (qfc * self.scale) @ kc.transpose(-2, -1)
        attnc = attnc.softmax(dim=-1)

        attns = (qfs * self.scale) @ ks.transpose(-2, -1)
        attns = attns.softmax(dim=-1)

        N = qfc.shape[2]

        if self.residual_pool:
            acx = (attnc @ vc + qfc).transpose(1, 2).reshape(B, N, C)
            asx = (attns @ vs + qfs).transpose(1, 2).reshape(B, N, C)
        else:
            acx = (attnc @ vc).transpose(1, 2).reshape(B, N, C)
            asx = (attns @ vs).transpose(1, 2).reshape(B, N, C)

        acx = self.proj_c(acx)
        asx = self.proj_s(asx)
        if self.dropout_rate > 0.0:
            acx = self.proj_drop_c(acx)
            asx = self.proj_drop_s(asx)

        #### Coronal
        fcx_rec, _ = self._attention_pool_mlpc(fxc, fx_raw.shape[2:])
        acx = fcx_rec + self.drop_path_mlpc(acx)
        acx_norm = (
            self.norm2_mlpc(acx.permute(0, 2, 1)).permute(0, 2, 1)
            if self.norm2_is_batchnorm_1d
            else self.norm2_mlpc(acx)
        )
        acx_mlp = self.mlp_c(acx_norm)
        if self.dim != self.dim_out:
            acx = self.proj_mlpc(acx_norm)
        acx = acx + self.drop_path_mlpc(acx_mlp)

        #### sagittal
        fxs_res, _ = self._attention_pool_mlps(fxs, fx_raw.shape[2:])
        asx = fxs_res + self.drop_path_mlps(asx)
        asx_norm = (
            self.norm2_mlps(asx.permute(0, 2, 1)).permute(0, 2, 1)
            if self.norm2_is_batchnorm_1d
            else self.norm2_mlps(asx)
        )
        asx_mlp = self.mlp_s(asx_norm)

        if self.dim != self.dim_out:
            asx = self.proj_mlps(asx_norm)
        asx = asx + self.drop_path_mlps(asx_mlp)

        ## fusion branch
        out = self.pool_skip_mlpf(fx_raw)
        tmpc = self._back_to_grid(acx, B=B, C=C, DHW=qfc_shape)
        tmps = self._back_to_grid(asx, B=B, C=C, DHW=qfc_shape)
        out = out + tmpc + tmps
        out = self.out_norm(out)
        return out, acx, asx


def create_x3d_bottleneck_block(
    *,
    # Convolution configs.
    dim_in: int,
    dim_inner: int,
    dim_out: int,
    conv_kernel_size: Tuple[int] = (3, 3, 3),
    conv_stride: Tuple[int] = (1, 2, 2),
    # Norm configs.
    norm: Callable = nn.BatchNorm3d,
    norm_eps: float = 1e-5,
    norm_momentum: float = 0.1,
    se_ratio: float = 0.0625,
    # Activation configs.
    activation: Callable = nn.ReLU,
    inner_act: Callable = Swish,
) -> nn.Module:
    """
    Bottleneck block for X3D: a sequence of Conv, Normalization with optional SE block,
    and Activations repeated in the following order:

    ::

                                    Conv3d (conv_a)
                                           ↓
                                 Normalization (norm_a)
                                           ↓
                                   Activation (act_a)
                                           ↓
                                    Conv3d (conv_b)
                                           ↓
                                 Normalization (norm_b)
                                           ↓
                                 Squeeze-and-Excitation
                                           ↓
                                   Activation (act_b)
                                           ↓
                                    Conv3d (conv_c)
                                           ↓
                                 Normalization (norm_c)

    Args:
        dim_in (int): input channel size to the bottleneck block.
        dim_inner (int): intermediate channel size of the bottleneck.
        dim_out (int): output channel size of the bottleneck.
        conv_kernel_size (tuple): convolutional kernel size(s) for conv_b.
        conv_stride (tuple): convolutional stride size(s) for conv_b.

        norm (callable): a callable that constructs normalization layer, examples
            include nn.BatchNorm3d, None (not performing normalization).
        norm_eps (float): normalization epsilon.
        norm_momentum (float): normalization momentum.
        se_ratio (float): if > 0, apply SE to the 3x3x3 conv, with the SE
            channel dimensionality being se_ratio times the 3x3x3 conv dim.

        activation (callable): a callable that constructs activation layer, examples
            include: nn.ReLU, nn.Softmax, nn.Sigmoid, and None (not performing
            activation).
        inner_act (callable): whether use Swish activation for act_b or not.

    Returns:
        (nn.Module): X3D bottleneck block.
    """
    # 1x1x1 Conv
    conv_a = nn.Conv3d(
        in_channels=dim_in, out_channels=dim_inner, kernel_size=(1, 1, 1), bias=False
    )
    norm_a = (
        None
        if norm is None
        else norm(num_features=dim_inner, eps=norm_eps, momentum=norm_momentum)
    )
    act_a = None if activation is None else activation()

    # 3x3x3 Conv
    conv_b = nn.Conv3d(
        in_channels=dim_inner,
        out_channels=dim_inner,
        kernel_size=conv_kernel_size,
        stride=conv_stride,
        padding=[size // 2 for size in conv_kernel_size],
        bias=False,
        groups=dim_inner,
        dilation=(1, 1, 1),
    )
    se = (
        SqueezeExcitation(
            num_channels=dim_inner,
            num_channels_reduced=round_width(dim_inner, se_ratio),
            is_3d=True,
        )
        if se_ratio > 0.0
        else nn.Identity()
    )
    norm_b = nn.Sequential(
        (
            nn.Identity()
            if norm is None
            else norm(num_features=dim_inner, eps=norm_eps, momentum=norm_momentum)
        ),
        se,
    )
    act_b = None if inner_act is None else inner_act()

    # 1x1x1 Conv
    conv_c = nn.Conv3d(
        in_channels=dim_inner, out_channels=dim_out, kernel_size=(1, 1, 1), bias=False
    )
    norm_c = (
        None
        if norm is None
        else norm(num_features=dim_out, eps=norm_eps, momentum=norm_momentum)
    )

    return BottleneckBlock(
        conv_a=conv_a,
        norm_a=norm_a,
        act_a=act_a,
        conv_b=conv_b,
        norm_b=norm_b,
        act_b=act_b,
        conv_c=conv_c,
        norm_c=norm_c,
    )


def create_x3d_res_block(
    *,
    # Bottleneck Block configs.
    dim_in: int,
    dim_inner: int,
    dim_out: int,
    bottleneck: Callable = create_x3d_bottleneck_block,
    use_shortcut: bool = True,
    # Conv configs.
    conv_kernel_size: Tuple[int] = (3, 3, 3),
    conv_stride: Tuple[int] = (1, 2, 2),
    # Norm configs.
    norm: Callable = nn.BatchNorm3d,
    norm_eps: float = 1e-5,
    norm_momentum: float = 0.1,
    se_ratio: float = 0.0625,
    # Activation configs.
    activation: Callable = nn.ReLU,
    inner_act: Callable = Swish,
) -> nn.Module:
    """
    Residual block for X3D. Performs a summation between an identity shortcut in branch1 and a
    main block in branch2. When the input and output dimensions are different, a
    convolution followed by a normalization will be performed.

    ::

                                         Input
                                           |-------+
                                           ↓       |
                                         Block     |
                                           ↓       |
                                       Summation ←-+
                                           ↓
                                       Activation

    Args:
        dim_in (int): input channel size to the bottleneck block.
        dim_inner (int): intermediate channel size of the bottleneck.
        dim_out (int): output channel size of the bottleneck.
        bottleneck (callable): a callable for create_x3d_bottleneck_block.

        conv_kernel_size (tuple): convolutional kernel size(s) for conv_b.
        conv_stride (tuple): convolutional stride size(s) for conv_b.

        norm (callable): a callable that constructs normalization layer, examples
            include nn.BatchNorm3d, None (not performing normalization).
        norm_eps (float): normalization epsilon.
        norm_momentum (float): normalization momentum.
        se_ratio (float): if > 0, apply SE to the 3x3x3 conv, with the SE
            channel dimensionality being se_ratio times the 3x3x3 conv dim.

        activation (callable): a callable that constructs activation layer, examples
            include: nn.ReLU, nn.Softmax, nn.Sigmoid, and None (not performing
            activation).
        inner_act (callable): whether use Swish activation for act_b or not.

    Returns:
        (nn.Module): X3D block layer.
    """

    norm_model = None
    if norm is not None and dim_in != dim_out:
        norm_model = norm(num_features=dim_out)

    return ResBlock(
        branch1_conv=(
            nn.Conv3d(
                dim_in,
                dim_out,
                kernel_size=(1, 1, 1),
                stride=conv_stride,
                bias=False,
            )
            if (dim_in != dim_out or np.prod(conv_stride) > 1) and use_shortcut
            else None
        ),
        branch1_norm=norm_model if dim_in != dim_out and use_shortcut else None,
        branch2=bottleneck(
            dim_in=dim_in,
            dim_inner=dim_inner,
            dim_out=dim_out,
            conv_kernel_size=conv_kernel_size,
            conv_stride=conv_stride,
            norm=norm,
            norm_eps=norm_eps,
            norm_momentum=norm_momentum,
            se_ratio=se_ratio,
            activation=activation,
            inner_act=inner_act,
        ),
        activation=None if activation is None else activation(),
        branch_fusion=lambda x, y: x + y,
    )


class ResBlock(nn.Module):
    """
    Residual block. Performs a summation between an identity shortcut in branch1 and a
    main block in branch2. When the input and output dimensions are different, a
    convolution followed by a normalization will be performed.

    ::


                                         Input
                                           |-------+
                                           ↓       |
                                         Block     |
                                           ↓       |
                                       Summation ←-+
                                           ↓
                                       Activation

    The builder can be found in `create_res_block`.
    """

    def __init__(
        self,
        branch1_conv: nn.Module = None,
        branch1_norm: nn.Module = None,
        branch2: nn.Module = None,
        activation: nn.Module = None,
        branch_fusion: Callable = None,
    ) -> nn.Module:
        """
        Args:
            branch1_conv (torch.nn.modules): convolutional module in branch1.
            branch1_norm (torch.nn.modules): normalization module in branch1.
            branch2 (torch.nn.modules): bottleneck block module in branch2.
            activation (torch.nn.modules): activation module.
            branch_fusion: (Callable): A callable or layer that combines branch1
                and branch2.
        """
        super().__init__()
        set_attributes(self, locals())
        assert self.branch2 is not None

    def forward(self, x) -> torch.Tensor:
        if self.branch1_conv is None:
            x = self.branch_fusion(x, self.branch2(x))
        else:
            shortcut = self.branch1_conv(x)
            if self.branch1_norm is not None:
                shortcut = self.branch1_norm(shortcut)
            x = self.branch_fusion(shortcut, self.branch2(x))
        if self.activation is not None:
            x = self.activation(x)
        return x


def create_x3d_res_stage(
    *,
    # Stage configs.
    depth: int,
    # Bottleneck Block configs.
    dim_in: int,
    dim_inner: int,
    dim_out: int,
    bottleneck: Callable = create_x3d_bottleneck_block,
    # Conv configs.
    conv_kernel_size: Tuple[int] = (3, 3, 3),
    conv_stride: Tuple[int] = (1, 2, 2),
    # Norm configs.
    norm: Callable = nn.BatchNorm3d,
    norm_eps: float = 1e-5,
    norm_momentum: float = 0.1,
    se_ratio: float = 0.0625,
    # Activation configs.
    activation: Callable = nn.ReLU,
    inner_act: Callable = Swish,
) -> nn.Module:
    """
    Create Residual Stage, which composes sequential blocks that make up X3D.

    ::

                                        Input
                                           ↓
                                       ResBlock
                                           ↓
                                           .
                                           .
                                           .
                                           ↓
                                       ResBlock

    Args:

        depth (init): number of blocks to create.

        dim_in (int): input channel size to the bottleneck block.
        dim_inner (int): intermediate channel size of the bottleneck.
        dim_out (int): output channel size of the bottleneck.
        bottleneck (callable): a callable for create_x3d_bottleneck_block.

        conv_kernel_size (tuple): convolutional kernel size(s) for conv_b.
        conv_stride (tuple): convolutional stride size(s) for conv_b.

        norm (callable): a callable that constructs normalization layer, examples
            include nn.BatchNorm3d, None (not performing normalization).
        norm_eps (float): normalization epsilon.
        norm_momentum (float): normalization momentum.
        se_ratio (float): if > 0, apply SE to the 3x3x3 conv, with the SE
            channel dimensionality being se_ratio times the 3x3x3 conv dim.

        activation (callable): a callable that constructs activation layer, examples
            include: nn.ReLU, nn.Softmax, nn.Sigmoid, and None (not performing
            activation).
        inner_act (callable): whether use Swish activation for act_b or not.

    Returns:
        (nn.Module): X3D stage layer.
    """
    res_blocks = []
    for idx in range(depth):
        block = create_x3d_res_block(
            dim_in=dim_in if idx == 0 else dim_out,
            dim_inner=dim_inner,
            dim_out=dim_out,
            bottleneck=bottleneck,
            conv_kernel_size=conv_kernel_size,
            conv_stride=conv_stride if idx == 0 else (1, 1, 1),
            norm=norm,
            norm_eps=norm_eps,
            norm_momentum=norm_momentum,
            se_ratio=(se_ratio if (idx + 1) % 2 else 0.0),
            activation=activation,
            inner_act=inner_act,
        )
        res_blocks.append(block)

    return ResStage(res_blocks=nn.ModuleList(res_blocks))


def create_x3d_head(
    *,
    # Projection configs.
    dim_in: int,
    dim_inner: int,
    dim_out: int,
    num_classes: int,
    # Pooling configs.
    pool_act: Callable = nn.ReLU,
    pool_kernel_size: Tuple[int] = (13, 5, 5),
    # BN configs.
    norm: Callable = nn.BatchNorm3d,
    norm_eps: float = 1e-5,
    norm_momentum: float = 0.1,
    bn_lin5_on=False,
    # Dropout configs.
    dropout_rate: float = 0.5,
    # Activation configs.
    activation: Callable = nn.Softmax,
    # Output configs.
    output_with_global_average: bool = True,
) -> nn.Module:
    """
    Creates X3D head. This layer performs an projected pooling operation followed
    by an dropout, a fully-connected projection, an activation layer and a global
    spatiotemporal averaging.

    ::

                                     ProjectedPool
                                           ↓
                                        Dropout
                                           ↓
                                       Projection
                                           ↓
                                       Activation
                                           ↓
                                       Averaging

    Args:
        dim_in (int): input channel size of the X3D head.
        dim_inner (int): intermediate channel size of the X3D head.
        dim_out (int): output channel size of the X3D head.
        num_classes (int): the number of classes for the video dataset.

        pool_act (callable): a callable that constructs resnet pool activation
            layer such as nn.ReLU.
        pool_kernel_size (tuple): pooling kernel size(s) when not using adaptive
            pooling.

        norm (callable): a callable that constructs normalization layer, examples
            include nn.BatchNorm3d, None (not performing normalization).
        norm_eps (float): normalization epsilon.
        norm_momentum (float): normalization momentum.
        bn_lin5_on (bool): if True, perform normalization on the features
            before the classifier.

        dropout_rate (float): dropout rate.

        activation (callable): a callable that constructs resnet head activation
            layer, examples include: nn.ReLU, nn.Softmax, nn.Sigmoid, and None (not
            applying activation).

        output_with_global_average (bool): if True, perform linear and global averaging on temporal
            and spatial dimensions and reshape output to batch_size x out_features.

    Returns:
        (nn.Module): X3D head layer.
    """
    pre_conv_module = nn.Conv3d(
        in_channels=dim_in, out_channels=dim_inner, kernel_size=(1, 1, 1), bias=False
    )

    pre_norm_module = norm(num_features=dim_inner, eps=norm_eps, momentum=norm_momentum)
    pre_act_module = None if pool_act is None else pool_act()

    if pool_kernel_size is None:
        pool_module = nn.AdaptiveAvgPool3d((1, 1, 1))
    else:
        pool_module = nn.AvgPool3d(pool_kernel_size, stride=1)

    post_conv_module = nn.Conv3d(
        in_channels=dim_inner, out_channels=dim_out, kernel_size=(1, 1, 1), bias=False
    )

    if bn_lin5_on:
        post_norm_module = norm(
            num_features=dim_out, eps=norm_eps, momentum=norm_momentum
        )
    else:
        post_norm_module = None
    post_act_module = None if pool_act is None else pool_act()

    projected_pool_module = ProjectedPool(
        pre_conv=pre_conv_module,
        pre_norm=pre_norm_module,
        pre_act=pre_act_module,
        pool=pool_module,
        post_conv=post_conv_module,
        post_norm=post_norm_module,
        post_act=post_act_module,
    )

    if activation is None:
        activation_module = None
    elif activation == nn.Softmax:
        activation_module = activation(dim=1)
    elif activation == nn.Sigmoid:
        activation_module = activation()
    else:
        raise NotImplementedError(
            "{} is not supported as an activation" "function.".format(activation)
        )

    if output_with_global_average:
        output_pool = nn.AdaptiveAvgPool3d(1)
        proj_linear = nn.Linear(dim_out, num_classes, bias=True)
    else:
        proj_linear = None
        output_pool = None
    return ResNetBasicHead(
        proj=proj_linear,
        activation=activation_module,
        pool=projected_pool_module,
        dropout=nn.Dropout(dropout_rate) if dropout_rate > 0 else None,
        output_pool=output_pool,
    )


def create_x3d(
    # Input clip configs.
    input_channel: int = 3,
    input_clip_length: int = 16,
    input_crop_size: int = 224,
    # Model configs.
    model_num_class: int = 5,
    dropout_rate: float = 0.5,
    width_factor: float = 2.0,
    depth_factor: float = 2.2,
    stage_depths: List[int] = [1, 2, 5, 3],
    # Normalization configs.
    norm: Callable = nn.BatchNorm3d,
    norm_eps: float = 1e-5,
    norm_momentum: float = 0.1,
    # Activation configs.
    activation: Callable = nn.ReLU,
    # Stem configs.
    stem_dim_in: int = 12,
    stem_conv_kernel_size: Tuple[int] = (5, 3, 3),
    stem_conv_stride: Tuple[int] = (1, 2, 2),
    # Stage configs.
    stage_conv_kernel_size: Tuple[Tuple[int]] = (
        (3, 3, 3),
        (3, 3, 3),
        (3, 3, 3),
        (3, 3, 3),
    ),
    stage_spatial_stride: Tuple[int] = (2, 2, 2, 2),
    stage_temporal_stride: Tuple[int] = (1, 1, 1, 1),
    bottleneck: Callable = create_x3d_bottleneck_block,
    bottleneck_factor: float = 2.25,
    se_ratio: float = 0.0625,
    inner_act: Callable = Swish,
    # Head configs.
    head_dim_out: int = 2048,
    head_pool_act: Callable = nn.ReLU,
    head_bn_lin5_on: bool = False,
    head_activation: Callable = None,
    head_output_with_global_average: bool = False,
    with_stem: bool = True,
    with_fusion: list = [None, None, None, None],
    middle_fusion: Callable = create_fusion,
    drop_start_prob: float = 0.1,
    drop_stop_prob: float = 0.3,
    drop_block_size: int = 32,
    **kwargs,
) -> nn.Module:
    """
    X3D model builder. It builds a X3D network backbone, which is a ResNet.

    Christoph Feichtenhofer.
    "X3D: Expanding Architectures for Efficient Video Recognition."
    https://arxiv.org/abs/2004.04730

    ::

                                         Input
                                           ↓
                                         Stem
                                           ↓
                                         Stage 1
                                           ↓
                                           .
                                           .
                                           .
                                           ↓
                                         Stage N
                                           ↓
                                         Head

    Args:
        input_channel (int): number of channels for the input video clip.
        input_clip_length (int): length of the input video clip. Value for
            different models: X3D-XS: 4; X3D-S: 13; X3D-M: 16; X3D-L: 16.
        input_crop_size (int): spatial resolution of the input video clip.
            Value for different models: X3D-XS: 160; X3D-S: 160; X3D-M: 224;
            X3D-L: 312.

        model_num_class (int): the number of classes for the video dataset.
        dropout_rate (float): dropout rate.
        width_factor (float): width expansion factor.
        depth_factor (float): depth expansion factor. Value for different
            models: X3D-XS: 2.2; X3D-S: 2.2; X3D-M: 2.2; X3D-L: 5.0.

        norm (callable): a callable that constructs normalization layer.
        norm_eps (float): normalization epsilon.
        norm_momentum (float): normalization momentum.

        activation (callable): a callable that constructs activation layer.

        stem_dim_in (int): input channel size for stem before expansion.
        stem_conv_kernel_size (tuple): convolutional kernel size(s) of stem.
        stem_conv_stride (tuple): convolutional stride size(s) of stem.

        stage_conv_kernel_size (tuple): convolutional kernel size(s) for conv_b.
        stage_spatial_stride (tuple): the spatial stride for each stage.
        stage_temporal_stride (tuple): the temporal stride for each stage.
        bottleneck_factor (float): bottleneck expansion factor for the 3x3x3 conv.
        se_ratio (float): if > 0, apply SE to the 3x3x3 conv, with the SE
            channel dimensionality being se_ratio times the 3x3x3 conv dim.
        inner_act (callable): whether use Swish activation for act_b or not.

        head_dim_out (int): output channel size of the X3D head.
        head_pool_act (callable): a callable that constructs resnet pool activation
            layer such as nn.ReLU.
        head_bn_lin5_on (bool): if True, perform normalization on the features
            before the classifier.
        head_activation (callable): a callable that constructs activation layer.
        head_output_with_global_average (bool): if True, perform global averaging on
            the head output.

    Returns:
        (nn.Module): the X3D network.
    """

    blocks = []
    # Create stem for X3D.
    stem_dim_out = round_width(stem_dim_in, width_factor)

    stem = create_x3d_stem(
        in_channels=input_channel,
        out_channels=stem_dim_out,
        conv_kernel_size=stem_conv_kernel_size,
        conv_stride=stem_conv_stride,
        conv_padding=[size // 2 for size in stem_conv_kernel_size],
        norm=norm,
        norm_eps=norm_eps,
        norm_momentum=norm_momentum,
        activation=activation,
    )
    blocks.append(stem)

    # Compute the depth and dimension for each stage
    exp_stage = 2.0
    stage_dim1 = stem_dim_in
    stage_dim2 = round_width(stage_dim1, exp_stage, divisor=8)
    stage_dim3 = round_width(stage_dim2, exp_stage, divisor=8)
    stage_dim4 = round_width(stage_dim3, exp_stage, divisor=8)
    stage_dims = [stage_dim1, stage_dim2, stage_dim3, stage_dim4]

    dim_in = stem_dim_out
    dim_inners = []
    # Create each stage for X3D.
    for idx in range(len(stage_depths)):
        dim_out = round_width(stage_dims[idx], width_factor)
        dim_inner = int(bottleneck_factor * dim_out)
        dim_inners.append(dim_inner)
        depth = round_repeats(stage_depths[idx], depth_factor)

        stage_conv_stride = (
            stage_temporal_stride[idx],
            stage_spatial_stride[idx],
            stage_spatial_stride[idx],
        )

        stage = create_x3d_res_stage(
            depth=depth,
            dim_in=dim_in,
            dim_inner=dim_inner,
            dim_out=dim_out,
            bottleneck=bottleneck,
            conv_kernel_size=stage_conv_kernel_size[idx],
            conv_stride=stage_conv_stride,
            norm=norm,
            norm_eps=norm_eps,
            norm_momentum=norm_momentum,
            se_ratio=se_ratio,
            activation=activation,
            inner_act=inner_act,
        )
        blocks.append(stage)
        dim_in = dim_out

    # Create head for X3D.
    total_spatial_stride = stem_conv_stride[1] * np.prod(stage_spatial_stride)
    total_temporal_stride = stem_conv_stride[0] * np.prod(stage_temporal_stride)

    assert (
        input_clip_length >= total_temporal_stride
    ), "Clip length doesn't match temporal stride!"
    assert (
        input_crop_size >= total_spatial_stride
    ), "Crop size doesn't match spatial stride!"

    head_pool_kernel_size = (
        input_clip_length // total_temporal_stride,
        int(math.ceil(input_crop_size / total_spatial_stride)),
        int(math.ceil(input_crop_size / total_spatial_stride)),
    )

    head = create_x3d_head(
        dim_in=dim_out,
        dim_inner=dim_inner,
        dim_out=head_dim_out,
        num_classes=model_num_class,
        pool_act=head_pool_act,
        pool_kernel_size=head_pool_kernel_size,
        norm=norm,
        norm_eps=norm_eps,
        norm_momentum=norm_momentum,
        bn_lin5_on=head_bn_lin5_on,
        dropout_rate=dropout_rate,
        activation=head_activation,
        output_with_global_average=head_output_with_global_average,
    )
    blocks.append(head)
    return Net(blocks=nn.ModuleList(blocks), dropblock=None)


class ProjectedPool(nn.Module):
    """
    A pooling module augmented with Conv, Normalization and Activation both
    before and after pooling for the head layer of X3D.

    ::

                                    Conv3d (pre_conv)
                                           ↓
                                 Normalization (pre_norm)
                                           ↓
                                   Activation (pre_act)
                                           ↓
                                        Pool3d
                                           ↓
                                    Conv3d (post_conv)
                                           ↓
                                 Normalization (post_norm)
                                           ↓
                                   Activation (post_act)
    """

    def __init__(
        self,
        *,
        pre_conv: nn.Module = None,
        pre_norm: nn.Module = None,
        pre_act: nn.Module = None,
        pool: nn.Module = None,
        post_conv: nn.Module = None,
        post_norm: nn.Module = None,
        post_act: nn.Module = None,
    ) -> None:
        """
        Args:
            pre_conv (torch.nn.modules): convolutional module.
            pre_norm (torch.nn.modules): normalization module.
            pre_act (torch.nn.modules): activation module.
            pool (torch.nn.modules): pooling module.
            post_conv (torch.nn.modules): convolutional module.
            post_norm (torch.nn.modules): normalization module.
            post_act (torch.nn.modules): activation module.
        """
        super().__init__()
        set_attributes(self, locals())
        assert self.pre_conv is not None
        assert self.pool is not None
        assert self.post_conv is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_conv(x)

        if self.pre_norm is not None:
            x = self.pre_norm(x)
        if self.pre_act is not None:
            x = self.pre_act(x)

        x = self.pool(x)
        x = self.post_conv(x)

        if self.post_norm is not None:
            x = self.post_norm(x)
        if self.post_act is not None:
            x = self.post_act(x)
        return x


class MARCLinear(nn.Module):
    """
    A wrapper for nn.Linear with support of MARC method.
    """

    def __init__(self, out_features, in_features=2048):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=True)
        self.output_pool = nn.AdaptiveAvgPool3d(output_size=1)
        self.a = torch.nn.Parameter(torch.ones(1, out_features))
        self.b = torch.nn.Parameter(torch.zeros(1, out_features))

    def forward(self, logit_before, *args):
        logit_before = logit_before.permute((0, 2, 3, 4, 1))
        logit_before = self.fc(logit_before)
        logit_before = logit_before.permute((0, 4, 1, 2, 3))
        logit_before = self.output_pool(logit_before)
        logit_before = logit_before.view(logit_before.shape[0], -1)
        w_norm = torch.norm(self.fc.weight.clone().detach(), dim=1)
        logit_after = self.a * logit_before + self.b * w_norm
        return logit_after


class SideOut(nn.Module):
    """
    A wrapper for nn.Linear with support of MARC method.
    """

    def __init__(
        self,
        dim_in: int,
        dim_inner: int,
        dim_out: int,
        num_classes: int,
        # Pooling configs.
        pool_act: Callable = nn.ReLU,
        pool_kernel_size: Tuple[int] = (13, 5, 5),
        conv_kernel_size: Tuple[int] = (3, 5, 5),
        conv_stride: Tuple[int] = (2, 2, 2),
        # BN configs.
        norm: Callable = nn.BatchNorm3d,
        norm_eps: float = 1e-5,
        norm_momentum: float = 0.1,
        bn_lin5_on=False,
        # Dropout configs.
        dropout_rate: float = 0.5,
        # Activation configs.
        activation: Callable = None,
        # Output configs.
        output_with_global_average: bool = False,
        out_features=5,
        **kwargs,
    ):
        super().__init__()
        self.stem = create_x3d_res_block(
            dim_in=dim_in,
            dim_inner=dim_inner,
            dim_out=dim_inner,
            conv_kernel_size=conv_kernel_size,
            conv_stride=conv_stride,
        )
        self.marc = MARCLinear(out_features=num_classes, in_features=dim_out)
        self.head = create_x3d_head(
            dim_in=dim_inner,
            dim_inner=int(dim_inner * 1.25),
            dim_out=dim_out,
            num_classes=num_classes,
            pool_act=pool_act,
            pool_kernel_size=pool_kernel_size,
            norm=norm,
            norm_eps=norm_eps,
            norm_momentum=norm_momentum,
            bn_lin5_on=bn_lin5_on,
            dropout_rate=dropout_rate,
            activation=activation,
            output_with_global_average=output_with_global_average,
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.head(x)
        x = self.marc(x)
        return x


class CVFSNet(nn.Module):
    def __init__(
        self,
        input_channel: int = 3,
        input_clip_length: int = 16,
        input_crop_size: int = 512,
        # Model configs.
        model_num_class: int = 5,
        dropout_rate: float = 0.5,
        width_factor: float = 2.0,
        depth_factor: float = 2.2,
        stage_depths: List[int] = [1, 2, 5, 3],
        # Normalization configs.
        norm: Callable = nn.BatchNorm3d,
        norm_eps: float = 1e-5,
        norm_momentum: float = 0.1,
        # Activation configs.
        activation: Callable = nn.ReLU,
        # Stem configs.
        stem_dim_in: int = 12,
        stem_conv_kernel_size: Tuple[int] = (5, 3, 3),
        stem_conv_stride: Tuple[int] = (1, 2, 2),
        # Stage configs.
        stage_conv_kernel_size: Tuple[Tuple[int]] = (
            (3, 3, 3),
            (3, 3, 3),
            (3, 3, 3),
            (3, 3, 3),
        ),
        stage_spatial_stride: Tuple[int] = (2, 2, 2, 2),
        stage_temporal_stride: Tuple[int] = (1, 1, 1, 1),
        bottleneck: Callable = create_x3d_bottleneck_block,
        bottleneck_factor: float = 2.25,
        se_ratio: float = 0.0625,
        inner_act: Callable = Swish,
        # Head configs.
        head_dim_out: int = 2048,
        head_pool_act: Callable = nn.ReLU,
        head_bn_lin5_on: bool = False,
        head_activation: Callable = None,
        head_output_with_global_average: bool = True,
        cor_pretrained=None,
        sag_pretrained=None,
        fuse_pretrained=None,
        all_pretrained=None,
        use_marc=True,
        with_stem: bool = True,
        with_fusion: list = [None, None, None, None],
        middle_fusion: Callable = create_fusion,
        use_fusion=False,
        deep_super=[False, False, False, False],
        OVA=False,
        mlp_dropout_rate: float = 0.05,
        num_heads: int = 4,
        expand_dim: int = 1,
        save_logit: bool = False,
        **kwargs,
    ):
        super(CVFSNet, self).__init__()
        self.use_marc = use_marc
        self.use_fusion = use_fusion
        self.loaded_keys = {}
        self.deep_super = deep_super
        self.visual_bottom = input_crop_size // 32
        self.save_logit = save_logit
        self.att_size = (
            input_clip_length,
            self.visual_bottom,
            self.visual_bottom,
        )
        if self.use_marc:
            self.coronal_marc_linear = MARCLinear(out_features=model_num_class)
            self.sagittal_marc_linear = MARCLinear(out_features=model_num_class)
            self.fusion_marc_linear = MARCLinear(out_features=model_num_class)
            head_output_with_global_average = False
        self.sagittal_model = create_x3d(
            input_channel,
            input_clip_length,
            input_crop_size,
            model_num_class,
            dropout_rate,
            width_factor,
            depth_factor,
            stage_depths,
            norm,
            norm_eps,
            norm_momentum,
            activation,
            stem_dim_in,
            stem_conv_kernel_size,
            stem_conv_stride,
            stage_conv_kernel_size,
            stage_spatial_stride,
            stage_temporal_stride,
            bottleneck,
            bottleneck_factor,
            se_ratio,
            inner_act,
            head_dim_out,
            head_pool_act,
            head_bn_lin5_on,
            head_activation,
            head_output_with_global_average,
            with_stem,
            with_fusion,
            middle_fusion,
            **kwargs,
        )

        self.coronal_model = copy.deepcopy(self.sagittal_model)
        if self.use_fusion == "CVFM":
            self.fusion_model = CVFM(
                num_classes=model_num_class,
                att_size=self.att_size,
                pool_kernel_size=(
                    input_clip_length,
                    self.visual_bottom // 2,
                    self.visual_bottom // 2,
                ),
                mlp_dropout_rate=mlp_dropout_rate,
                num_heads=num_heads,
                expand_dim=expand_dim,
                preconv=None,
            )
        else:
            self.fusion_model = None
        self.cor_side0 = None
        self.cor_side1 = (
            SideOut(
                dim_in=24,
                dim_inner=54,
                dim_out=256,
                num_classes=model_num_class,
                pool_kernel_size=(
                    input_clip_length // 2,
                    self.visual_bottom * 4,
                    self.visual_bottom * 4,
                ),
            )
            if self.deep_super[1]
            else None
        )

        self.cor_side2 = (
            SideOut(
                dim_in=48,
                dim_inner=108,
                dim_out=512,
                num_classes=model_num_class,
                pool_kernel_size=(
                    input_clip_length // 2,
                    self.visual_bottom * 2,
                    self.visual_bottom * 2,
                ),
            )
            if self.deep_super[2]
            else None
        )

        self.cor_side3 = (
            SideOut(
                dim_in=96,
                dim_inner=216,
                dim_out=1024,
                num_classes=model_num_class,
                pool_kernel_size=(
                    input_clip_length // 2,
                    self.visual_bottom,
                    self.visual_bottom,
                ),
            )
            if self.deep_super[3]
            else None
        )

        self.sag_side0 = None
        self.sag_side1 = (
            SideOut(
                dim_in=24,
                dim_inner=54,
                dim_out=256,
                num_classes=model_num_class,
                pool_kernel_size=(
                    input_clip_length // 2,
                    self.visual_bottom * 4,
                    self.visual_bottom * 4,
                ),
            )
            if self.deep_super[1]
            else None
        )

        self.sag_side2 = (
            SideOut(
                dim_in=48,
                dim_inner=108,
                dim_out=512,
                num_classes=model_num_class,
                pool_kernel_size=(
                    input_clip_length // 2,
                    self.visual_bottom * 2,
                    self.visual_bottom * 2,
                ),
            )
            if self.deep_super[2]
            else None
        )

        self.sag_side3 = (
            SideOut(
                dim_in=96,
                dim_inner=216,
                dim_out=1024,
                num_classes=model_num_class,
                pool_kernel_size=(
                    input_clip_length // 2,
                    self.visual_bottom,
                    self.visual_bottom,
                ),
            )
            if self.deep_super[3]
            else None
        )
        if cor_pretrained and not all_pretrained:
            checkpoint_all = torch.load(
                cor_pretrained, map_location=torch.device("cpu")
            )
            if checkpoint_all.get("net_dict") is None:
                checkpoint = checkpoint_all.get("model_state")
                drop_key = "proj"
            else:
                checkpoint = checkpoint_all.get("net_dict")
                drop_key = "#####"
            cor_raw_net_dict = self.coronal_model.state_dict()
            cor_new_state_dict = OrderedDict()
            cor_loaded_keys = []
            cor_unloaded_keys = []
            for k, v in checkpoint.items():
                if "module" in k:
                    k = k[7:]
                elif "model." in k:
                    k = k[6:]
                elif k in cor_raw_net_dict.keys() and drop_key not in k:
                    cor_new_state_dict[k] = v
                    cor_loaded_keys.append(k)
                else:
                    cor_unloaded_keys.append(k)
            self.coronal_model.load_state_dict(cor_new_state_dict, strict=False)
            self.loaded_keys["cor_loaded_keys"] = cor_loaded_keys
            cprint("cor_unloaded_keys {}".format(cor_unloaded_keys), color="yellow")

        if sag_pretrained and not all_pretrained:
            checkpoint_all = torch.load(
                sag_pretrained, map_location=torch.device("cpu")
            )
            if checkpoint_all.get("net_dict") is None:
                checkpoint = checkpoint_all.get("model_state")
                drop_key = "proj"
            else:
                checkpoint = checkpoint_all.get("net_dict")
                drop_key = "#####"
            sag_raw_net_dict = self.sagittal_model.state_dict()
            sag_new_state_dict = OrderedDict()
            sag_loaded_keys = []
            sag_unloaded_keys = []
            for k, v in checkpoint.items():
                if "module" in k:
                    k = k[7:]
                elif "model." in k:
                    k = k[6:]
                elif k in sag_raw_net_dict.keys() and drop_key not in k:
                    sag_new_state_dict[k] = v
                    sag_loaded_keys.append(k)
                else:
                    sag_unloaded_keys.append(k)
            self.sagittal_model.load_state_dict(sag_new_state_dict, strict=False)
            self.loaded_keys["sag_loaded_keys"] = sag_loaded_keys
            cprint("sag_unloaded_keys {}".format(sag_unloaded_keys), color="yellow")

        if fuse_pretrained and self.fusion_model and not all_pretrained:
            checkpoint_all = torch.load(
                fuse_pretrained, map_location=torch.device("cpu")
            )
            if checkpoint_all.get("net_dict") is None:
                checkpoint = checkpoint_all.get("model_state")
                drop_key = "proj"
            else:
                checkpoint = checkpoint_all.get("net_dict")
                drop_key = "#####"
            fusion_new_state_dict = OrderedDict()
            fusion_raw_net_dict = self.fusion_model.state_dict()
            fuse_loaded_keys = []
            fuse_unloaded_keys = []
            for k, v in checkpoint.items():
                if "module" in k:
                    k = k[7:]
                elif "model." in k:
                    k = k[6:]
                elif (
                    k.replace("proj", "fuse_conv") in fusion_raw_net_dict.keys()
                    and drop_key not in k
                ):
                    k = k.replace("proj", "fuse_conv")
                    fusion_new_state_dict[k] = v
                    fuse_loaded_keys.append(k)
                else:
                    fuse_unloaded_keys.append(k)
            self.fusion_model.load_state_dict(fusion_new_state_dict, strict=False)
            self.loaded_keys["fuse_loaded_keys"] = fuse_loaded_keys
            cprint("fuse_unloaded_keys {}".format(fuse_unloaded_keys), color="yellow")
        if all_pretrained:
            checkpoint_all = torch.load(
                all_pretrained, map_location=torch.device("cpu")
            )
            if checkpoint_all.get("net_dict") is None:
                checkpoint = checkpoint_all.get("model_state")
            else:
                checkpoint = checkpoint_all.get("net_dict")
            all_new_state_dict = OrderedDict()
            all_raw_net_dict = self.state_dict()
            all_unloaded_keys = []
            drop_key = "###"
            for k, v in checkpoint.items():
                if "module" in k:
                    k = k[7:]
                if k in all_raw_net_dict.keys() and drop_key not in k:
                    all_new_state_dict[k] = v
                else:
                    all_unloaded_keys.append(k)
            self.load_state_dict(all_new_state_dict, strict=False)
            self.loaded_keys["all_unloaded_keys"] = all_unloaded_keys
            cprint("all_unloaded_keys {}".format(all_unloaded_keys), color="yellow")

    def forward(self, view: list[torch.Tensor]):
        results = []
        cor_middel_list = self.coronal_model(view[0])
        sag_middel_list = self.sagittal_model(view[1])
        cor_output = self.coronal_marc_linear(cor_middel_list[-1])
        sag_output = self.sagittal_marc_linear(sag_middel_list[-1])
        fuse_output_ova = [None]
        if self.fusion_model:
            fuse_output, help_tensor = self.fusion_model(
                cor_middel_list, sag_middel_list
            )
            fuse_output = self.fusion_marc_linear(fuse_output)
        else:
            fuse_output = self.fusion_marc_linear(
                cor_middel_list[-1] + sag_middel_list[-1]
            )
        results.append(fuse_output)
        results.append(cor_output)
        results.append(sag_output)
        if self.save_logit:
            all_save_logit = dict(
                cor_middel_list=move_to_cpu(cor_middel_list),
                sag_middel_list=move_to_cpu(sag_middel_list),
                help_tensor=move_to_cpu(help_tensor),
                input_tensor=move_to_cpu(view),
            )
        if self.deep_super[1]:
            results.append(self.cor_side1(cor_middel_list[1]))
            results.append(self.sag_side1(sag_middel_list[1]))
        else:
            results.append(None)
            results.append(None)
        if self.deep_super[2]:
            results.append(self.cor_side2(cor_middel_list[2]))
            results.append(self.sag_side2(sag_middel_list[2]))
        else:
            results.append(None)
            results.append(None)
        if self.deep_super[3]:
            results.append(self.cor_side3(cor_middel_list[3]))
            results.append(self.sag_side3(sag_middel_list[3]))
        else:
            results.append(None)
            results.append(None)
        results = results + fuse_output_ova
        if self.save_logit:
            return results, all_save_logit
        else:
            return results


if __name__ == "__main__":
    depth = 8
    visual = 256
    cor_view = torch.autograd.Variable(torch.rand(2, 3, depth, visual, visual))
    sag_view = torch.autograd.Variable(torch.rand(2, 3, depth, visual, visual))
    net = CVFSNet(
        input_clip_length=depth,
        input_crop_size=visual,
        cor_pretrained=None,  # "/ai/mnt/code/DSFNet_MTICI/Src/X3D_M-Kinect.pyth",  # "/ai/mnt/code/DSFNet_MTICI/output_runs/mTICI_Single_LMDB/X3D/SINGLE_VIEW-COR-Fold1/03_30-22_32#oversample_weighted-fixnormvalue-crop-lmdb-visual32-use_marc-NLrs/Model/Best_Acc_Epoch_0080.pth",
        sag_pretrained=None,  # "/ai/mnt/code/DSFNet_MTICI/output_runs/mTICI_Single_LMDB/X3D/SINGLE_VIEW-SAG-Fold1/03_31-09_25#oversample_weighted-fixnormvalue-crop-lmdb-visual32-use_marc-NLrs/Model/Best_Acc_Epoch_0065.pth",
        use_fusion="CVFM",
        model_num_class=4,
        deep_super=[True, True, True, True],
        OVA=False,
        mlp_dropout_rate=0,
        num_heads=8,
        expand_dim=8,
        save_logit=True,
    )
    logger.add("/ai/mnt/code/DSFNet_MTICI/Src/DVCNet.log")
    logger.info(net)
    output, all_save_logit = net([cor_view, sag_view])
    logger.info(output)
    flops, params = profile(net, inputs=([cor_view, sag_view],))
    print("FLOPs = " + str(flops / 1000**3) + "G")
    print("Params = " + str(params / 1000**2) + "M")

    # if hasattr(net, "net.fusion_model.CVAFM"):
    #     logger.info("GOT CHA")
    # else:
    #     logger.info("MISS")
    # target = torch.Tensor([3, 1]).long()
    # from Loss import FocalLabelSmoothOVA_MISO

    # loss = FocalLabelSmoothOVA_MISO(num_classes=4)
    # l = loss(output, target)
    # logger.info(l)
    # logger.info(net.loaded_keys)
    # logger.info(output[0].shape, output[1].shape, output[2].shape)
    # cor_view = torch.autograd.Variable(torch.rand(4, 24, 16, 128, 128)).cuda()
    # net = SideOut(
    #     dim_in=24,
    #     dim_inner=54,
    #     dim_out=256,
    #     num_classes=5,
    #     pool_kernel_size=(16, 128, 128),
    #     in_features=256,
    # ).cuda()
    # tmp = net(cor_view)

    # depth = 32
    # att = MultiScaleAttention(dim_in=192, expand_dim=4, size=(depth, 16, 16)).cuda()
    # logger.info(att)
    # fx = torch.autograd.Variable(torch.rand(2, 192, depth, 16, 16)).cuda()
    # cx = torch.autograd.Variable(torch.rand(2, 192, depth, 16, 16)).cuda()
    # sx = torch.autograd.Variable(torch.rand(2, 192, depth, 16, 16)).cuda()
    # out = att(fx, cx, sx)
    # logger.info(out[0].shape)
