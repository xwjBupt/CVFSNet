import torch.nn as nn
import torch.nn.functional as F
import functools
import torch
import numpy as np
from torch.nn.modules.loss import _WeightedLoss
import json
from typing import Callable, List, Optional, Tuple


def soft_cross_entropy(
    pred, label, weight=None, reduction="mean", class_weight=None, avg_factor=None
):
    """Calculate the Soft CrossEntropy loss. The label can be float.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The gt label of the prediction with shape (N, C).
            When using "mixup", the label can be float.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (torch.Tensor, optional): The weight for each class with
            shape (C), C is the number of classes. Default None.

    Returns:
        torch.Tensor: The calculated loss
    """
    # element-wise losses
    loss = -label * F.log_softmax(pred, dim=-1)
    if class_weight is not None:
        loss *= class_weight
    loss = loss.sum(dim=-1)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor
    )
    return loss


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction="mean", avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == "mean":
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != "none":
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    ``loss_func(pred, target, **kwargs)``. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like ``loss_func(pred, target, weight=None, reduction='mean',
    avg_factor=None, **kwargs)``.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, avg_factor=2)
    tensor(1.5000)
    """

    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction="mean", avg_factor=None, **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper


def convert_to_one_hot(targets: torch.Tensor, classes=5) -> torch.Tensor:
    """This function converts target class indices to one-hot vectors, given
    the number of classes.

    Args:
        targets (Tensor): The ground truth label of the prediction
                with shape (N, 1)
        classes (int): the number of classes.

    Returns:
        Tensor: Processed loss values.
    """
    assert (
        torch.max(targets).item() < classes
    ), "Class Index must be less than number of classes"
    one_hot_targets = F.one_hot(targets.long().squeeze(-1), num_classes=classes)
    return one_hot_targets


class GRWCrossEntropyLoss(nn.Module):
    """
    Generalized Reweight Loss, introduced in
    Distribution Alignment: A Unified Framework for Long-tail Visual Recognition
    https://arxiv.org/abs/2103.16370
    """

    __constants__ = ["ignore_index", "reduction"]

    def _init_weights(self, fuse01, fold, exp_scale=1.2):
        if fold == "FOLD1":
            freq = [83, 24, 44, 50, 78]
        elif fold == "FOLD2":
            freq = [36, 11, 28, 22, 43]
        else:
            freq = [75, 26, 53, 49, 86]
        if fuse01:
            num_classes = 4
            freq[1] = freq[1] + freq[0]
            del freq[0]
        else:
            num_classes = 5
        num_shots = np.array(freq)
        ratio_list = num_shots / np.sum(num_shots)
        exp_reweight = 1 / (ratio_list**exp_scale)
        exp_reweight = exp_reweight / np.sum(exp_reweight)
        exp_reweight = exp_reweight * num_classes
        exp_reweight = torch.tensor(exp_reweight).float()
        return exp_reweight

    def __init__(
        self,
        fuse01,
        fold,
        use_dict=True,
        exp_scale=1.2,
        size_average=None,
        ignore_index=-100,
        reduce=None,
        reduction="mean",
    ):
        super(GRWCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.use_dict = use_dict
        self.reduction = reduction
        self.weight = self._init_weights(fuse01=fuse01, fold=fold, exp_scale=exp_scale)

    def forward(self, input, target):
        # if self.weight.device != input.device:
        #     self.weight.to(input.device)
        input = F.log_softmax(input, dim=-1)
        loss = F.cross_entropy(
            input,
            target.squeeze(1),
            weight=self.weight.to(input.device),
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )
        if self.use_dict:
            return loss, dict(loss=loss.item())
        else:
            return loss


class CrossEntropyLoss(nn.Module):
    """Cross entropy loss.

    Args:
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to 'mean'.
        loss_weight (float):  Weight of the loss. Defaults to 1.0.
        class_weight (List[float], optional): The weight for each class with
            shape (C), C is the number of classes. Default None.
        pos_weight (List[float], optional): The positive weight for each
            class with shape (C), C is the number of classes. Only enabled in
            BCE loss when ``use_sigmoid`` is True. Default None.
    """

    def __init__(
        self,
        reduction="mean",
        loss_weight=1.0,
        use_dict=True,
        class_weight=None,
        pos_weight=None,
    ):
        super(CrossEntropyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.pos_weight = pos_weight
        self.cls_criterion = soft_cross_entropy
        self.use_dict = use_dict

    def forward(
        self,
        cls_score,
        label,
        weight=None,
        avg_factor=None,
        reduction_override=None,
        **kwargs,
    ):
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction

        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        label = convert_to_one_hot(label)
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs,
        )
        if self.use_dict:
            return loss_cls, dict(loss=loss_cls.item())
        else:
            return loss_cls


class FocalLoss(nn.Module):
    """Multi-class Focal loss implementation"""

    def __init__(self, gamma=2, weight=None, num_classes=4, alphas=None, use_dict=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.use_dict = use_dict
        if num_classes == 4:
            self.alphas = torch.tensor([0.15, 0.35, 0.35, 0.15]).cuda()
        else:
            self.alphas = torch.tensor([0.1, 0.275, 0.275, 0.15, 0.1]).cuda()
        if alphas is None:
            self.alphas = torch.tensor([1, 1, 1, 1, 1, 1]).cuda()

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        alpha = self.alphas[target]  # 去取真实索引类别对应的alpha
        logpt = alpha.to(target.device) * (1 - pt) ** self.gamma * logpt
        loss_cls = F.nll_loss(logpt, target.squeeze(1), self.weight)
        if self.use_dict:
            return loss_cls, dict(loss=loss_cls.item())
        else:
            return loss_cls


class OHEM_CE(nn.Module):
    """Multi-class OHEM loss implementation"""

    def __init__(self, use_dict=True, keep_rate=0.7):
        super(OHEM_CE, self).__init__()
        self.use_dict = use_dict
        self.keep_rate = keep_rate

    def forward(self, cls_pred, cls_target):
        """Arguments:
        cls_pred (FloatTensor): [R, C]
        cls_target (LongTensor): [R]
        Returns:
            cls_loss (FloatTensor)
        """
        cls_pred = F.log_softmax(cls_pred, dim=1)
        ohem_cls_loss = F.cross_entropy(
            cls_pred, cls_target.squeeze(1), reduction="none", ignore_index=-1
        )
        sorted_ohem_loss, idx = torch.sort(ohem_cls_loss, descending=True)
        # 再对loss进行降序排列
        batch_size = cls_pred.shape[0]
        keep_num = min(sorted_ohem_loss.size()[0], int(batch_size * self.keep_rate))
        # 得到需要保留的loss数量
        if keep_num < sorted_ohem_loss.size()[0]:
            # 这句的作用是如果保留数目小于现有loss总数，则进行筛选保留，否则全部保留
            keep_idx_cuda = idx[:keep_num]  # 保留到需要keep的数目
            ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]
        cls_loss = ohem_cls_loss.sum() / keep_num

        if self.use_dict:
            return cls_loss, dict(loss=cls_loss.item())
        else:
            return cls_loss


class GHMC(nn.Module):
    def __init__(
        self,
        bins=10,
        momentum=0,
        use_sigmoid=True,
        loss_weight=1.0,
        use_dict=True,
        **kwargs,
    ):
        super(GHMC, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = [float(x) / bins for x in range(bins + 1)]
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = [0.0 for _ in range(bins)]
        self.use_sigmoid = use_sigmoid
        self.loss_weight = loss_weight

    def forward(self, pred, target, label_weight, *args, **kwargs):
        """Args:
        pred [batch_num, class_num]:
            The direct prediction of classification fc layer.
        target [batch_num, class_num]:
            Binary class target for each sample.
        label_weight [batch_num, class_num]:
            the value is 1 if the sample is valid and 0 if ignored.
        """
        if not self.use_sigmoid:
            raise NotImplementedError
        # the target should be binary class label
        if pred.dim() != target.dim():
            target, label_weight = self.expand_binary_labels(
                target, label_weight, pred.size(-1)
            )
        target, label_weight = target.float(), label_weight.float()
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(pred)

        # gradient length
        g = torch.abs(pred.sigmoid().detach() - target)

        valid = label_weight > 0
        tot = max(valid.float().sum().item(), 1.0)
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n

        loss = (
            F.binary_cross_entropy_with_logits(pred, target, weights, reduction="sum")
            / tot
        )
        return loss * self.loss_weight

    def expand_binary_labels(self, labels, label_weights, label_channels):
        bin_labels = labels.new_full((labels.size(0), label_channels), 0)
        inds = torch.nonzero(labels >= 1).squeeze()
        if inds.numel() > 0:
            bin_labels[inds, labels[inds] - 1] = 1
        bin_label_weights = label_weights.view(-1, 1).expand(
            label_weights.size(0), label_channels
        )
        return bin_labels, bin_label_weights


class AsymmetricLossSingleLabel(nn.Module):
    def __init__(self, gamma_pos=1, gamma_neg=4, eps: float = 0.1, reduction="mean"):
        super(AsymmetricLossSingleLabel, self).__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []  # prevent gpu repeated memory allocation
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, target, reduction=None):
        """ "
        Parameters
        ----------
        x: input logits
        y: targets (1-hot vector)
        """

        num_classes = inputs.size()[-1]
        target = convert_to_one_hot(target)
        log_preds = self.logsoftmax(inputs)
        self.targets_classes = torch.zeros_like(inputs).scatter_(
            1, target.long().unsqueeze(1), 1
        )

        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(
            1 - xs_pos - xs_neg,
            self.gamma_pos * targets + self.gamma_neg * anti_targets,
        )
        log_preds = log_preds * asymmetric_w

        if self.eps > 0:  # label smoothing
            self.targets_classes = self.targets_classes.mul(1 - self.eps).add(
                self.eps / num_classes
            )

        # loss calculation
        loss = -self.targets_classes.mul(log_preds)

        loss = loss.sum(dim=-1)
        if self.reduction == "mean":
            loss = loss.mean()
        return loss


class DistanceLoss(nn.Module):
    def __init__(self, loss_mode="mse", use_act=True, use_dict=True, **kwargs):
        super(DistanceLoss, self).__init__()

        if loss_mode == "mse":
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.SmoothL1Loss()
        self.use_dict = use_dict
        self.use_act = use_act

    def forward(self, input, target):
        centers = self.__get_centers(input, target)
        loss = torch.Tensor([1e-8]).float().to(input.device)
        if centers[0].shape == centers[1].shape:
            loss += 1 - self.loss(centers[0], centers[1])
        if centers[2].shape == centers[3].shape:
            loss += 1 - self.loss(centers[2], centers[3])

        if self.use_dict:
            return loss, dict(loss=loss.item())
        else:
            return loss

    def __get_centers(self, input, target):
        centers = [
            [torch.Tensor([1e-8]).float().to(input.device)],
            [torch.Tensor([1e-8]).float().to(input.device)],
            [torch.Tensor([1e-8]).float().to(input.device)],
            [torch.Tensor([1e-8]).float().to(input.device)],
            [torch.Tensor([1e-8]).float().to(input.device)],
        ]
        average_centers = [[], [], [], [], []]
        num_batch = input.shape[0]
        for i in range(num_batch):
            if self.use_act:
                centers[target[i].item()].append(F.softmax(input[i], dim=-1))
            else:
                centers[target[i].item()].append(input[i])
        for index, i in enumerate(centers):
            if len(i) >= 2:
                average_centers[index] = sum(i[1:]) / (len(i) - 1)
            else:
                average_centers[index] = i[0]
        return average_centers


class LabelSmoothingCrossEntropy(nn.Module):
    """NLL loss with label smoothing."""

    def __init__(self, smoothing=0.1, use_dict=True):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.use_dict = use_dict

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target)
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        if self.use_dict:
            return loss.mean(), dict(loss=loss.mean().item())
        else:
            return loss.mean()


class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction="mean"):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        pt = torch.sigmoid(predict)  # sigmoide获取概率
        # 在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
        loss = -1 * self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - (
            1 - self.alpha
        ) * pt**self.gamma * (1 - target) * torch.log(1 - pt)

        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        return loss


class OVALoss(nn.Module):
    def __init__(
        self,
        gamma=2,
        alpha=0.25,
        reduction="mean",
        num_classes=5,
        use_dict=True,
        **kwargs,
    ):
        super(OVALoss, self).__init__()
        self.loss = BCEFocalLoss()
        self.num_classes = num_classes
        self.use_dict = use_dict
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        onehot_target = convert_to_one_hot(target, self.num_classes).view(-1, 1)
        x = x.view(-1, 1)
        loss = self.loss(x, onehot_target)
        loss_dict = {}
        loss_dict["OVA"] = loss.mean().item()
        if self.use_dict:
            return loss, loss_dict
        else:
            return loss

    # def _convert_to_onehot(self, target, num_classes = 4):


class FocalLabelSmooth(nn.Module):
    """NLL loss with label smoothing."""

    def __init__(self, balance=[1, 1, 1, 1], use_dict=True):
        super(FocalLabelSmooth, self).__init__()
        self.label_smooth = LabelSmoothingCrossEntropy(use_dict=False)
        self.focal = FocalLoss(use_dict=False)
        self.balance = balance
        self.use_dict = use_dict

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        label_smooth_loss = self.label_smooth(x, target) * self.balance[1]
        focal_loss = self.focal(x, target) * self.balance[0]
        loss = focal_loss + label_smooth_loss
        if self.use_dict:
            return loss.mean(), dict(
                loss=loss.mean().item(),
                focal_loss=focal_loss.mean().item(),
                label_smooth_loss=label_smooth_loss.mean().item(),
            )
        else:
            return loss.mean()


class FocalLabelSmooth_MISO(nn.Module):
    """NLL loss with label smoothing."""

    def __init__(
        self,
        balance=[
            [1, 0.8],
            [1, 0.8],
            [1, 0.8],
            [0.5, 0.4],
            [0.5, 0.4],
            [0.25, 0.2],
            [0.25, 0.2],
            [0.125, 0.1],
            [0.125, 0.1],
        ],
        smoothing=0.4,
        use_dict=True,
        **kwargs,
    ):
        super(FocalLabelSmooth_MISO, self).__init__()
        self.label_smooth = LabelSmoothingCrossEntropy(
            smoothing=smoothing, use_dict=False
        )
        self.focal = FocalLoss(use_dict=False)
        self.balance = balance
        self.use_dict = use_dict

    def forward(self, xs: List, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        loss_dict = {}
        for index, x in enumerate(xs):
            if x is not None:
                label_smooth_loss = (
                    self.label_smooth(x, target) * self.balance[index][0]
                )
                focal_loss = self.focal(x, target) * self.balance[index][1]
                loss = focal_loss + label_smooth_loss + loss
                if index == 0:
                    phase = "FUSE"
                elif index == 1:
                    phase = "COR"
                elif index == 2:
                    phase = "SAG"
                else:
                    phase = str(index)
                loss_dict["focal_loss#%s" % phase] = focal_loss.mean().item()
                loss_dict["label_smooth_loss#%s" % phase] = (
                    label_smooth_loss.mean().item()
                )
        loss_dict["loss"] = loss.mean().item()
        if self.use_dict:
            return loss.mean(), loss_dict
        else:
            return loss.mean()


class FocalLabelSmoothOHEM_MISO(nn.Module):
    def __init__(
        self,
        balance=[
            [1, 0.8, 1.4],
            [1, 0.8, 1.4],
            [1, 0.8, 1.4],
            [0.5, 0.4, 0.7],
            [0.5, 0.4, 0.7],
            [0.25, 0.2, 0.35],
            [0.25, 0.2, 0.35],
            [0.125, 0.1, 0.2],
            [0.125, 0.1, 0.2],
        ],
        smoothing=0.4,
        keep_rate=0.55,
        use_dict=True,
        **kwargs,
    ):
        super(FocalLabelSmoothOHEM_MISO, self).__init__()
        self.label_smooth = LabelSmoothingCrossEntropy(
            smoothing=smoothing, use_dict=False
        )
        self.focal = FocalLoss(use_dict=False)
        self.ohem = OHEM_CE(keep_rate=keep_rate, use_dict=False)
        self.balance = balance
        self.use_dict = use_dict

    def forward(self, xs: List, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        loss_dict = {}
        for index, x in enumerate(xs):
            if x is not None:
                label_smooth_loss = (
                    self.label_smooth(x, target) * self.balance[index][0]
                )
                focal_loss = self.focal(x, target) * self.balance[index][1]
                ohem_loss = self.ohem(x, target) * self.balance[index][2]
                loss = focal_loss + label_smooth_loss + ohem_loss + loss
                if index == 0:
                    phase = "FUSE"
                elif index == 1:
                    phase = "COR"
                elif index == 2:
                    phase = "SAG"
                else:
                    phase = str(index)
                loss_dict["Fl#%s" % phase] = focal_loss.mean().item()
                loss_dict["LS#%s" % phase] = label_smooth_loss.mean().item()
                loss_dict["OHEM#%s" % phase] = ohem_loss.mean().item()
        loss_dict["loss"] = loss.mean().item()
        if self.use_dict:
            return loss.mean(), loss_dict
        else:
            return loss.mean()


class FocalLabelSmoothSeasaw_MISO(nn.Module):
    def __init__(
        self,
        balance=[
            [1, 0.8, 1.4],
            [1, 0.8, 1.4],
            [1, 0.8, 1.4],
            [0.5, 0.4, 0.7],
            [0.5, 0.4, 0.7],
            [0.25, 0.2, 0.35],
            [0.25, 0.2, 0.35],
            [0.125, 0.1, 0.2],
            [0.125, 0.1, 0.2],
        ],
        smoothing=0.4,
        use_dict=True,
        num_classes=4,
        p=0.8,
        q=2.0,
        eps=1e-2,
        **kwargs,
    ):
        super(FocalLabelSmoothSeasaw_MISO, self).__init__()
        self.label_smooth = LabelSmoothingCrossEntropy(
            smoothing=smoothing, use_dict=False
        )
        self.focal = FocalLoss(use_dict=False)
        self.sea = SeesawLoss(
            num_classes=num_classes, p=p, q=q, eps=eps, use_dict=False
        )
        self.balance = balance
        self.use_dict = use_dict

    def forward(self, xs: List, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        loss_dict = {}
        for index, x in enumerate(xs):
            if x is not None:
                label_smooth_loss = (
                    self.label_smooth(x, target) * self.balance[index][0]
                )
                focal_loss = self.focal(x, target) * self.balance[index][1]
                sea_loss = self.sea(x, target) * self.balance[index][2]
                loss = focal_loss + label_smooth_loss + sea_loss + loss
                if index == 0:
                    phase = "FUSE"
                elif index == 1:
                    phase = "COR"
                elif index == 2:
                    phase = "SAG"
                else:
                    phase = str(index)
                loss_dict["Fl#%s" % phase] = focal_loss.mean().item()
                loss_dict["LS#%s" % phase] = label_smooth_loss.mean().item()
                loss_dict["SS#%s" % phase] = sea_loss.mean().item()
        loss_dict["loss"] = loss.mean().item()
        if self.use_dict:
            return loss.mean(), loss_dict
        else:
            return loss.mean()


class LabelSmoothSeasaw_MISO(nn.Module):
    def __init__(
        self,
        balance=[
            [1, 0.8, 1.4],
            [1, 0.8, 1.4],
            [1, 0.8, 1.4],
            [0.5, 0.4, 0.7],
            [0.5, 0.4, 0.7],
            [0.25, 0.2, 0.35],
            [0.25, 0.2, 0.35],
            [0.125, 0.1, 0.2],
            [0.125, 0.1, 0.2],
        ],
        smoothing=0.4,
        use_dict=True,
        num_classes=4,
        p=0.8,
        q=2.0,
        eps=1e-2,
        **kwargs,
    ):
        super(LabelSmoothSeasaw_MISO, self).__init__()
        self.label_smooth = LabelSmoothingCrossEntropy(
            smoothing=smoothing, use_dict=False
        )
        self.sea = SeesawLoss(
            num_classes=num_classes, p=p, q=q, eps=eps, use_dict=False
        )
        self.balance = balance
        self.use_dict = use_dict

    def forward(self, xs: List, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        loss_dict = {}
        for index, x in enumerate(xs):
            if x is not None:
                label_smooth_loss = (
                    self.label_smooth(x, target) * self.balance[index][0]
                )
                sea_loss = self.sea(x, target) * self.balance[index][1]
                loss = label_smooth_loss + sea_loss + loss
                if index == 0:
                    phase = "FUSE"
                elif index == 1:
                    phase = "COR"
                elif index == 2:
                    phase = "SAG"
                else:
                    phase = str(index)
                loss_dict["LS#%s" % phase] = label_smooth_loss.mean().item()
                loss_dict["SS#%s" % phase] = sea_loss.mean().item()
        loss_dict["loss"] = loss.mean().item()
        if self.use_dict:
            return loss.mean(), loss_dict
        else:
            return loss.mean()


class FocalLabelSmoothOVA_MISO(nn.Module):
    """NLL loss with label smoothing."""

    def __init__(
        self,
        balance=[
            [1, 0.8, 1],
            [1, 0.8, 1],
            [1, 0.8, 1],
            [0.5, 0.4, 1],
            [0.5, 0.4, 1],
            [0.25, 0.2, 1],
            [0.25, 0.2, 1],
            [0.125, 0.1, 1],
            [0.125, 0.1, 1],
        ],
        smoothing=0.4,
        use_dict=True,
        num_classes=5,
        **kwargs,
    ):
        super(FocalLabelSmoothOVA_MISO, self).__init__()
        self.label_smooth = LabelSmoothingCrossEntropy(
            smoothing=smoothing, use_dict=False
        )
        self.focal = FocalLoss(use_dict=False)
        self.ova = OVALoss(num_classes=num_classes, use_dict=False)
        self.balance = balance
        self.use_dict = use_dict

    def forward(self, xs: List, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        loss_dict = {}
        if xs[-1]:
            ovaloss = self.ova(xs[-1], target)
            loss_dict["OVA"] = ovaloss.mean().item() * self.balance[0][-1]
        else:
            ovaloss = 0
            loss_dict["OVA"] = 0
        del xs[-1]
        for index, x in enumerate(xs):
            if x is not None:
                label_smooth_loss = (
                    self.label_smooth(x, target.long()) * self.balance[index][0]
                )
                focal_loss = self.focal(x, target.long()) * self.balance[index][1]
                loss = focal_loss + label_smooth_loss + loss
                if index == 0:
                    phase = "FUSE"
                elif index == 1:
                    phase = "COR"
                elif index == 2:
                    phase = "SAG"
                else:
                    phase = str(index)
                loss_dict["focal_loss#%s" % phase] = focal_loss.mean().item()
                loss_dict["label_smooth_loss#%s" % phase] = (
                    label_smooth_loss.mean().item()
                )
        loss_dict["loss"] = loss.mean().item()
        if self.use_dict:
            return loss.mean(), loss_dict
        else:
            return loss.mean()


class FocalLabelSmoothSupclu_MISO(nn.Module):
    def __init__(
        self,
        balance=[
            [1, 0.8, 0.6],
            [1, 0.8, 0.6],
            [1, 0.8, 0.6],
            [0.5, 0.4, 0.3],
            [0.5, 0.4, 0.3],
            [0.25, 0.2, 0.15],
            [0.25, 0.2, 0.15],
            [0.125, 0.1, 0.075],
            [0.125, 0.1, 0.015],
        ],
        smoothing=0.4,
        temperature=0.07,
        contrast_mode="all",
        base_temperature=0.07,
        use_dict=True,
    ):
        super(FocalLabelSmoothSupclu_MISO, self).__init__()
        self.label_smooth = LabelSmoothingCrossEntropy(
            smoothing=smoothing, use_dict=False
        )
        self.focal = FocalLoss(use_dict=False)
        self.supclu = SupCluLoss(
            temperature=temperature,
            contrast_mode=contrast_mode,
            base_temperature=base_temperature,
            use_dict=False,
        )
        self.balance = balance
        self.use_dict = use_dict

    def forward(self, xs: List, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        loss_dict = {}
        for index, x in enumerate(xs):
            if index == 0:
                phase = "FUSE"
            elif index == 1:
                phase = "COR"
            elif index == 2:
                phase = "SAG"
            else:
                phase = str(index)
            if x is not None:
                focal_loss = self.focal(x, target) * self.balance[index][0]
                loss_dict["focal#%s" % phase] = focal_loss.mean().item()
                label_smooth_loss = (
                    self.label_smooth(x, target) * self.balance[index][1]
                )
                loss_dict["label_smooth#%s" % phase] = label_smooth_loss.mean().item()
                supclu_loss = self.supclu(x, target) * self.balance[index][2]
                loss_dict["supclu#%s" % phase] = supclu_loss.mean().item()

                # if index <= 2:
                #     label_smooth_loss = (
                #         self.label_smooth(x, target) * self.balance[index][0]
                #     )
                #     loss_dict[
                #         "label_smooth_loss#%s" % phase
                #     ] = label_smooth_loss.mean().item()
                #     supclu_loss = 0
                #     loss_dict["supclu_loss#%s" % phase] = 0
                # else:
                #     label_smooth_loss = 0
                #     loss_dict["label_smooth_loss#%s" % phase] = 0
                #     supclu_loss = self.supclu(x, target) * self.balance[index][0]
                #     loss_dict["supclu_loss#%s" % phase] = supclu_loss.mean().item()
            loss = focal_loss + supclu_loss + label_smooth_loss + loss
        loss_dict["loss"] = loss.mean().item()
        if self.use_dict:
            return loss.mean(), loss_dict
        else:
            return loss.mean()


class GRWLabelSmooth_MISO(nn.Module):
    """NLL loss with label smoothing."""

    def __init__(
        self,
        fuse01,
        fold,
        smoothing=0.1,
        balance=[[1, 1], [1, 1], [1, 1], [1, 1]],
        use_dict=True,
    ):
        super(GRWLabelSmooth_MISO, self).__init__()
        self.label_smooth = LabelSmoothingCrossEntropy(
            smoothing=smoothing, use_dict=False
        )
        self.grw = GRWCrossEntropyLoss(fuse01, fold, use_dict=False)
        self.balance = balance
        self.use_dict = use_dict

    def forward(self, xs: List, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        loss_dict = {}
        for index, x in enumerate(xs):
            label_smooth_loss = self.label_smooth(x, target) * self.balance[index][0]
            grw_loss = self.grw(x, target) * self.balance[index][1]
            loss = grw_loss + label_smooth_loss + loss
            if index == 0:
                phase = "FUSE"
            elif index == 1:
                phase = "COR"
            elif index == 2:
                phase = "SAG"
            else:
                phase = str(index)
            loss_dict["grw_loss#%s" % phase] = grw_loss.mean().item()
            loss_dict["label_smooth_loss#%s" % phase] = label_smooth_loss.mean().item()
        loss_dict["loss"] = loss.mean().item()
        if self.use_dict:
            return loss.mean(), loss_dict
        else:
            return loss.mean()


class FocalLabelSmoothMSE(nn.Module):
    """Focal loss ,label smoothing, and MSE"""

    def __init__(self, balance=[1, 1, 1, 1], use_dict=True):
        super(FocalLabelSmoothMSE, self).__init__()
        self.label_smooth = LabelSmoothingCrossEntropy(use_dict=False)
        self.focal = FocalLoss(use_dict=False)
        self.mse = torch.nn.MSELoss(reduction="mean")
        self.balance = balance
        self.use_dict = use_dict

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        label_smooth_loss = self.label_smooth(x, target) * self.balance[1]
        focal_loss = self.focal(x, target) * self.balance[0]
        mse_loss = (
            self.mse(
                x, F.one_hot(target.long().squeeze(-1), num_classes=x.shape[-1]).float()
            )
            * self.balance[2]
        )
        loss = focal_loss + label_smooth_loss + mse_loss
        if self.use_dict:
            return loss.mean(), dict(
                loss=loss.mean().item(),
                focal_loss=focal_loss.mean().item(),
                label_smooth_loss=label_smooth_loss.mean().item(),
                mse_loss=mse_loss.mean().item(),
            )
        else:
            return loss.mean()


class OHEMLabelSmooth(nn.Module):
    def __init__(
        self, keep_rate=0.7, balance=[1, 1, 1, 1], smoothing=0.1, use_dict=True
    ):
        super(OHEMLabelSmooth, self).__init__()
        self.label_smooth = LabelSmoothingCrossEntropy(
            smoothing=smoothing, use_dict=False
        )
        self.ohem = OHEM_CE(keep_rate=keep_rate, use_dict=False)
        self.balance = balance
        self.use_dict = use_dict

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        label_smooth_loss = self.label_smooth(x, target) * self.balance[1]
        ohem_loss = self.ohem(x, target) * self.balance[0]
        loss = ohem_loss + label_smooth_loss
        if self.use_dict:
            return loss.mean(), dict(
                loss=loss.mean().item(),
                ohem_loss=ohem_loss.mean().item(),
                label_smooth_loss=label_smooth_loss.mean().item(),
            )
        else:
            return loss.mean()


class OHEMLabelSmooth_MISO(nn.Module):
    """NLL loss with label smoothing."""

    def __init__(
        self,
        balance=[[1, 0.8], [1, 0.8], [1, 0.8], [1, 0.8]],
        keep_rate=0.7,
        smoothing=0.1,
        use_dict=True,
    ):
        super(OHEMLabelSmooth_MISO, self).__init__()
        self.label_smooth = LabelSmoothingCrossEntropy(
            smoothing=smoothing, use_dict=False
        )
        self.ohem = OHEM_CE(keep_rate=keep_rate, use_dict=False)
        self.balance = balance
        self.use_dict = use_dict

    def forward(self, xs: List, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        loss_dict = {}
        for index, x in enumerate(xs):
            label_smooth_loss = self.label_smooth(x, target) * self.balance[index][1]
            ohem_loss = self.ohem(x, target) * self.balance[index][0]
            loss = ohem_loss + label_smooth_loss + loss
            if index == 0:
                phase = "FUSE"
            elif index == 1:
                phase = "COR"
            elif index == 2:
                phase = "SAG"
            else:
                phase = str(index)
            loss_dict["ohem_loss#%s" % phase] = ohem_loss.mean().item()
            loss_dict["label_smooth_loss#%s" % phase] = label_smooth_loss.mean().item()
        loss_dict["loss"] = loss.mean().item()
        if self.use_dict:
            return loss.mean(), loss_dict
        else:
            return loss.mean()


class GRWFOCAL_MISO(nn.Module):
    def __init__(
        self,
        fuse01,
        fold,
        balance=[[1, 0.8], [1, 0.8], [1, 0.8], [1, 0.8]],
        use_dict=True,
    ):
        super(GRWFOCAL_MISO, self).__init__()
        self.grw = GRWCrossEntropyLoss(fuse01, fold, use_dict=False)
        self.focal = FocalLoss(use_dict=False)
        self.balance = balance
        self.use_dict = use_dict

    def forward(self, xs: List, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        loss_dict = {}
        for index, x in enumerate(xs):
            focal_loss = self.focal(x, target) * self.balance[index][1]
            grw_loss = self.grw(x, target) * self.balance[index][0]
            loss = grw_loss + focal_loss + loss
            if index == 0:
                phase = "FUSE"
            elif index == 1:
                phase = "COR"
            elif index == 2:
                phase = "SAG"
            else:
                phase = str(index)
            loss_dict["grw_loss#%s" % phase] = grw_loss.mean().item()
            loss_dict["focal_loss#%s" % phase] = focal_loss.mean().item()
        loss_dict["loss"] = loss.mean().item()
        if self.use_dict:
            return loss.mean(), loss_dict
        else:
            return loss.mean()


class GRWOHEM_MISO(nn.Module):
    def __init__(
        self,
        fuse01,
        fold,
        keep_rate=0.7,
        balance=[[1, 0.8], [1, 0.8], [1, 0.8], [1, 0.8]],
        use_dict=True,
    ):
        super(GRWOHEM_MISO, self).__init__()
        self.grw = GRWCrossEntropyLoss(fuse01, fold, use_dict=False)
        self.ohem = OHEM_CE(keep_rate=keep_rate, use_dict=False)
        self.balance = balance
        self.use_dict = use_dict

    def forward(self, xs: List, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        loss_dict = {}
        for index, x in enumerate(xs):
            ohem_loss = self.ohem(x, target) * self.balance[index][1]
            grw_loss = self.grw(x, target) * self.balance[index][0]
            loss = grw_loss + ohem_loss + loss
            if index == 0:
                phase = "FUSE"
            elif index == 1:
                phase = "COR"
            elif index == 2:
                phase = "SAG"
            else:
                phase = str(index)
            loss_dict["grw_loss#%s" % phase] = grw_loss.mean().item()
            loss_dict["ohem_loss#%s" % phase] = ohem_loss.mean().item()
        loss_dict["loss"] = loss.mean().item()
        if self.use_dict:
            return loss.mean(), loss_dict
        else:
            return loss.mean()


class FocalOHEM_MISO(nn.Module):
    def __init__(
        self,
        keep_rate=0.7,
        balance=[[1, 0.8], [1, 0.8], [1, 0.8], [1, 0.8]],
        use_dict=True,
    ):
        super(FocalOHEM_MISO, self).__init__()
        self.focal = FocalLoss(use_dict=False)
        self.ohem = OHEM_CE(keep_rate=keep_rate, use_dict=False)
        self.balance = balance
        self.use_dict = use_dict

    def forward(self, xs: List, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        loss_dict = {}
        for index, x in enumerate(xs):
            ohem_loss = self.ohem(x, target) * self.balance[index][1]
            focal_loss = self.focal(x, target) * self.balance[index][0]
            loss = focal_loss + ohem_loss + loss
            if index == 0:
                phase = "FUSE"
            elif index == 1:
                phase = "COR"
            elif index == 2:
                phase = "SAG"
            else:
                phase = str(index)
            loss_dict["focal_loss#%s" % phase] = focal_loss.mean().item()
            loss_dict["ohem_loss#%s" % phase] = ohem_loss.mean().item()
        loss_dict["loss"] = loss.mean().item()
        if self.use_dict:
            return loss.mean(), loss_dict
        else:
            return loss.mean()


class DistanceFocalOHEM_MISO(nn.Module):
    def __init__(
        self,
        keep_rate=0.7,
        loss_mode="mse",
        balance=[[1, 0.8, 0.8], [1, 0.8, 0.8], [1, 0.8, 0.8], [1, 0.8, 0.8]],
        use_dict=True,
        **kwargs,
    ):
        super(DistanceFocalOHEM_MISO, self).__init__()
        self.focal = FocalLoss(use_dict=False)
        self.ohem = OHEM_CE(keep_rate=keep_rate, use_dict=False)
        self.dis = DistanceLoss(loss_mode=loss_mode, use_dict=False)
        self.balance = balance
        self.use_dict = use_dict

    def forward(self, xs: List, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        loss_dict = {}
        for index, x in enumerate(xs):
            dis_loss = self.dis(x, target) * self.balance[index][0]
            focal_loss = self.focal(x, target) * self.balance[index][1]
            ohem_loss = self.ohem(x, target) * self.balance[index][2]
            loss = focal_loss + ohem_loss + loss + dis_loss
            if index == 0:
                phase = "FUSE"
            elif index == 1:
                phase = "COR"
            elif index == 2:
                phase = "SAG"
            else:
                phase = str(index)
            loss_dict["dis_loss#%s" % phase] = dis_loss.mean().item()
            loss_dict["focal_loss#%s" % phase] = focal_loss.mean().item()
            loss_dict["ohem_loss#%s" % phase] = ohem_loss.mean().item()
        loss_dict["loss"] = loss.mean().item()
        if self.use_dict:
            return loss.mean(), loss_dict
        else:
            return loss.mean()


class SupCluLoss(nn.Module):
    def __init__(
        self,
        temperature=0.07,
        contrast_mode="all",
        base_temperature=0.07,
        use_dict=True,
    ):
        super(SupCluLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.use_dict = use_dict

    def forward(self, features, labels=None, mask=None):
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            # print('label_shape',labels.shape)
            # print('batch_size',batch_size)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = 1
        contrast_feature = features
        if self.contrast_mode == "one":
            assert False
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        if self.use_dict:
            return loss, dict(loss=loss.item())
        else:
            return loss


class FocalSupClu(nn.Module):
    def __init__(
        self,
        balance=[1, 1, 1, 1],
        temperature=0.07,
        contrast_mode="all",
        base_temperature=0.07,
        use_dict=True,
        **kwargs,
    ):
        super(FocalSupClu, self).__init__()
        self.focal = FocalLoss(use_dict=False)
        self.supclu = SupCluLoss(
            temperature=temperature,
            contrast_mode=contrast_mode,
            base_temperature=base_temperature,
            use_dict=False,
        )
        self.balance = balance
        self.use_dict = use_dict

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        focal_loss = self.focal(x, target) * self.balance[0]
        supclu_loss = self.supclu(x, target) * self.balance[1]
        loss = focal_loss + supclu_loss
        if self.use_dict:
            return loss.mean(), dict(
                loss=loss.mean().item(),
                focal_loss=focal_loss.mean().item(),
                supclu_loss=supclu_loss.mean().item(),
            )
        else:
            return loss.mean()


class FocalSupClu_MISO(nn.Module):
    def __init__(
        self,
        temperature=0.07,
        contrast_mode="all",
        base_temperature=0.07,
        balance=[[3, 0.75, 0.8], [2, 0.75, 0.8], [2, 0.75, 0.8], [1, 0.8, 0.8]],
        use_dict=True,
        **kwargs,
    ):
        super(FocalSupClu_MISO, self).__init__()
        self.focal = FocalLoss(use_dict=False)
        self.supclu = SupCluLoss(
            temperature=temperature,
            contrast_mode=contrast_mode,
            base_temperature=base_temperature,
            use_dict=False,
        )
        self.balance = balance
        self.use_dict = use_dict

    def forward(self, xs: List, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        loss_dict = {}
        for index, x in enumerate(xs):
            focal_loss = self.focal(x, target) * self.balance[index][1]
            supclu_loss = self.supclu(x, target) * self.balance[index][2]
            loss = focal_loss + supclu_loss + loss
            if index == 0:
                phase = "FUSE"
            elif index == 1:
                phase = "COR"
            elif index == 2:
                phase = "SAG"
            else:
                phase = str(index)
            loss_dict["focal_loss#%s" % phase] = focal_loss.mean().item()
            loss_dict["supclu_loss#%s" % phase] = supclu_loss.mean().item()
        loss_dict["loss"] = loss.mean().item()
        if self.use_dict:
            return loss.mean(), loss_dict
        else:
            return loss.mean()


class DistanceFocalLabelsoomth(nn.Module):
    def __init__(
        self,
        loss_mode="mse",
        smoothing=0.4,
        balance=[[1, 0.8, 0.8], [1, 0.8, 0.8], [1, 0.8, 0.8], [1, 0.8, 0.8]],
        use_dict=True,
        **kwargs,
    ):
        super(DistanceFocalLabelsoomth, self).__init__()
        self.focal = FocalLoss(use_dict=False)
        self.label_smooth = LabelSmoothingCrossEntropy(
            smoothing=smoothing, use_dict=False
        )
        self.dis = DistanceLoss(loss_mode=loss_mode, use_dict=False)
        self.balance = balance
        self.use_dict = use_dict

    def forward(self, xs: List, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        loss_dict = {}
        for index, x in enumerate(xs):
            dis_loss = self.dis(x, target) * self.balance[index][0]
            focal_loss = self.focal(x, target) * self.balance[index][1]
            label_smooth_loss = self.label_smooth(x, target) * self.balance[index][2]
            loss = focal_loss + label_smooth_loss + loss + dis_loss
            if index == 0:
                phase = "FUSE"
            elif index == 1:
                phase = "COR"
            elif index == 2:
                phase = "SAG"
            else:
                phase = str(index)
            loss_dict["dis_loss#%s" % phase] = dis_loss.mean().item()
            loss_dict["focal_loss#%s" % phase] = focal_loss.mean().item()
            loss_dict["label_smooth_loss#%s" % phase] = label_smooth_loss.mean().item()
        loss_dict["loss"] = loss.mean().item()
        if self.use_dict:
            return loss.mean(), loss_dict
        else:
            return loss.mean()


def weight_reduce_loss(loss, weight=None, reduction="mean", avg_factor=None):
    """Apply element-wise weight and reduce loss.
    Args:
            loss (Tensor): Element-wise loss.
            weight (Tensor): Element-wise weights.
            reduction (str): Same as built-in losses of PyTorch.
            avg_factor (float): Average factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

        # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == "mean":
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != "none":
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def seesaw_ce_loss(
    cls_score,
    labels,
    weight,
    cum_samples,
    num_classes,
    p,
    q,
    eps,
    reduction="mean",
    avg_factor=None,
):
    """Calculate the Seesaw CrossEntropy loss.

    Args:
        cls_score (torch.Tensor): The prediction with shape (N, C),
             C is the number of classes.
        labels (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor): Sample-wise loss weight.
        cum_samples (torch.Tensor): Cumulative samples for each category.
        num_classes (int): The number of classes.
        p (float): The ``p`` in the mitigation factor.
        q (float): The ``q`` in the compenstation factor.
        eps (float): The minimal value of divisor to smooth
             the computation of compensation factor
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        torch.Tensor: The calculated loss
    """

    assert cls_score.size(-1) == num_classes
    assert len(cum_samples) == num_classes

    onehot_labels = F.one_hot(labels, num_classes)
    seesaw_weights = cls_score.new_ones(onehot_labels.size())

    # mitigation factor
    if p > 0:
        sample_ratio_matrix = cum_samples[None, :].clamp(min=1) / cum_samples[
            :, None
        ].clamp(min=1)
        index = (sample_ratio_matrix < 1.0).float()
        sample_weights = sample_ratio_matrix.pow(p) * index + (1 - index)  # M_{ij}
        mitigation_factor = sample_weights[labels.long(), :]
        seesaw_weights = seesaw_weights * mitigation_factor

    # compensation factor
    if q > 0:
        scores = F.softmax(cls_score.detach(), dim=1)
        self_scores = scores[
            torch.arange(0, len(scores)).to(scores.device).long(), labels.long()
        ]
        score_matrix = scores / self_scores[:, None].clamp(min=eps)
        index = (score_matrix > 1.0).float()
        compensation_factor = score_matrix.pow(q) * index + (1 - index)
        seesaw_weights = seesaw_weights * compensation_factor

    cls_score = cls_score + (seesaw_weights.log() * (1 - onehot_labels))

    loss = F.cross_entropy(cls_score, labels, weight=None, reduction="none")

    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor
    )
    return loss


class SeesawLoss(nn.Module):
    """Implementation of seesaw loss.

    Refers to `Seesaw Loss for Long-Tailed Instance Segmentation (CVPR 2021)
    <https://arxiv.org/abs/2008.10032>`_

    Args:
        use_sigmoid (bool): Whether the prediction uses sigmoid of softmax.
             Only False is supported. Defaults to False.
        p (float): The ``p`` in the mitigation factor.
             Defaults to 0.8.
        q (float): The ``q`` in the compenstation factor.
             Defaults to 2.0.
        num_classes (int): The number of classes.
             Defaults to 1000 for the ImageNet dataset.
        eps (float): The minimal value of divisor to smooth
             the computation of compensation factor, default to 1e-2.
        reduction (str): The method that reduces the loss to a scalar.
             Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float): The weight of the loss. Defaults to 1.0
    """

    def __init__(
        self,
        use_sigmoid=False,
        p=0.8,
        q=2.0,
        num_classes=4,
        eps=1e-2,
        reduction="mean",
        loss_weight=1.0,
        use_dict=True,
        **kwargs,
    ):
        super(SeesawLoss, self).__init__()
        assert not use_sigmoid, "`use_sigmoid` is not supported"
        self.use_sigmoid = False
        self.p = p
        self.q = q
        self.num_classes = num_classes
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.use_dict = use_dict

        self.cls_criterion = seesaw_ce_loss
        self.cum_samples = torch.zeros(self.num_classes, dtype=torch.float).cuda()
        # cumulative samples for each category
        # self.register_buffer(
        #     "cum_samples", torch.zeros(self.num_classes, dtype=torch.float)
        # )

    def forward(
        self,
        cls_score,
        labels,
        weight=None,  # torch.tensor([0.75, 1.25, 1.25, 0.75]),
        avg_factor=None,
        reduction_override=None,
    ):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The prediction with shape (N, C).
            labels (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                 the loss. Defaults to None.
            reduction (str, optional): The method used to reduce the loss.
                 Options are "none", "mean" and "sum".
        Returns:
            torch.Tensor: The calculated loss
        """
        loss_dict = {}
        # self.cum_samples.to(labels.device)
        assert reduction_override in (None, "none", "mean", "sum"), (
            f'The `reduction_override` should be one of (None, "none", '
            f'"mean", "sum"), but get "{reduction_override}".'
        )
        assert cls_score.size(0) == labels.view(-1).size(0), (
            f"Expected `labels` shape [{cls_score.size(0)}], "
            f"but got {list(labels.size())}"
        )
        reduction = reduction_override if reduction_override else self.reduction
        assert cls_score.size(-1) == self.num_classes, (
            f"The channel number of output ({cls_score.size(-1)}) does "
            f"not match the `num_classes` of seesaw loss ({self.num_classes})."
        )

        # accumulate the samples for each category
        unique_labels = labels.unique()
        for u_l in unique_labels:
            inds_ = labels == u_l.item()
            self.cum_samples[u_l] += inds_.sum()

        if weight is not None:
            weight = weight.float().to(labels.device)
        else:
            weight = labels.new_ones(labels.size(), dtype=torch.float).to(labels.device)
        labels = labels.squeeze(1)
        # calculate loss_cls_classes
        loss = self.loss_weight * self.cls_criterion(
            cls_score,
            labels,
            weight,
            self.cum_samples,
            self.num_classes,
            self.p,
            self.q,
            self.eps,
            reduction,
            avg_factor,
        )
        loss_dict["loss"] = loss.mean().item()
        if self.use_dict:
            return loss, loss_dict
        else:
            return loss


class LabelSmoothSeasawQualitative_MISO(nn.Module):
    def __init__(
        self,
        balance=[
            [1, 0.8, 1.4],
            [1, 0.8, 1.4],
            [1, 0.8, 1.4],
            [0.5, 0.4, 0.7],
            [0.5, 0.4, 0.7],
            [0.25, 0.2, 0.35],
            [0.25, 0.2, 0.35],
            [0.125, 0.1, 0.2],
            [0.125, 0.1, 0.2],
        ],
        smoothing=0.4,
        use_dict=True,
        num_classes=4,
        p=0.8,
        q=2.0,
        eps=1e-2,
        **kwargs,
    ):
        super(LabelSmoothSeasawQualitative_MISO, self).__init__()
        self.label_smooth = LabelSmoothingCrossEntropy(
            smoothing=smoothing, use_dict=False
        )
        self.sea = SeesawLoss(
            num_classes=num_classes, p=p, q=q, eps=eps, use_dict=False
        )
        self.sea_qul = SeesawLoss(num_classes=2, p=p, q=q, eps=eps, use_dict=False)
        self.balance = balance
        self.use_dict = use_dict

    def forward(self, xs: List, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        loss_dict = {}
        for index, x in enumerate(xs):
            if x is not None:
                label_smooth_loss = (
                    self.label_smooth(x, target) * self.balance[index][0]
                )
                sea_loss = self.sea(x, target) * self.balance[index][1]
                xs_qul, target_qul = self.get_qualitative_inputs(x, target)
                sea_qul_loss = self.sea_qul(xs_qul, target_qul) * self.balance[index][1]

                loss = label_smooth_loss + sea_loss + sea_qul_loss + loss
                if index == 0:
                    phase = "FUSE"
                elif index == 1:
                    phase = "COR"
                elif index == 2:
                    phase = "SAG"
                else:
                    phase = str(index)
                loss_dict["LS#%s" % phase] = label_smooth_loss.mean().item()
                loss_dict["SS#%s" % phase] = sea_loss.mean().item()
                loss_dict["SSQul#%s" % phase] = sea_qul_loss.mean().item()
        loss_dict["loss"] = loss.mean().item()
        if self.use_dict:
            return loss.mean(), loss_dict
        else:
            return loss.mean()

    def get_qualitative_inputs(self, x, target, **kwargs):
        xs_qul = F.sigmoid(x)
        xs_bad = xs_qul[:, 0, ...] + xs_qul[:, 1, ...]
        xs_bad = xs_bad.unsqueeze(0).permute(1, 0)
        xs_good = xs_qul[:, 2, ...] + xs_qul[:, 3, ...]
        xs_good = xs_good.unsqueeze(0).permute(1, 0)
        xs_qul = torch.cat([xs_bad, xs_good], -1)
        xs_qul = F.softmax(xs_qul, -1)
        target_qul = torch.where(target >= 2, 1, 0)
        return xs_qul, target_qul


if __name__ == "__main__":
    # logits = torch.rand(2, 5).cuda()
    # labels = torch.LongTensor([[2], [3]]).cuda()
    # fl = GRWLabelSmooth(fuse01=False, fold="FOLD1")
    # print(fl(logits, labels))

    logits = torch.rand(4, 4)
    labels = torch.LongTensor([1, 3, 2, 0])
    fl = SeesawLoss()
    print(fl(logits, labels))
