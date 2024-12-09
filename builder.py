import torch
import importlib
import torch.optim as optim
import math
import importlib
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import Src
import Data
import Lib
import Loss
from prefetch_generator import BackgroundGenerator
from torchsampler import ImbalancedDatasetSampler
from torch.utils.data import WeightedRandomSampler


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT
    return getattr(module, config[name]["type"])(*args, **config[name]["args"])


def build_loss(cfg):
    loss = getattr(Loss, cfg.Name)(**cfg.Para)
    return loss


def build_transform(cfg):
    pass


def build_model(cfg):
    model = getattr(Src, cfg.Name)(**cfg.Para)
    return model


def get_group_parameters(net, lr, trans_scaler=3):
    if trans_scaler == 1:
        return net.parameters()
    trans_params_id = []
    try:
        trans_params_id = list(map(id, net.fusion_model.CVAFM.parameters()))
    except Exception:
        trans_params_id = []
    finally:
        base_params = filter(lambda p: id(p) not in trans_params_id, net.parameters())
        trans_params = filter(lambda p: id(p) in trans_params_id, net.parameters())
        param_group = [
            {"params": base_params, "lr": lr},
            {"params": trans_params, "lr": lr * trans_scaler},
        ]
    return param_group


def build_opt(cfg, net):
    params = get_group_parameters(net, cfg.Para.lr, cfg.Trans_scaler)
    opt_name = cfg.Name
    optimizer = getattr(importlib.import_module("torch.optim"), opt_name)(
        params=params, **cfg.Para
    )
    return optimizer


class Poly_scheduler(object):
    def __init__(
        self,
        optimizer,
        max_peoch,
        initial_lr,
    ):
        super().__init__()
        self.max_epoch = max_peoch
        self.initial_lr = initial_lr
        self.exponent = 0.9
        self.optimizer = optimizer

    def step(self, epoch):
        current_lr = self.initial_lr * (1 - epoch / self.max_epoch) ** self.exponent
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = current_lr


def build_scheduler(cfg, optimizer):
    if not cfg.Name:
        scheduler = None
        return scheduler
    if cfg.Name != "poly":
        scheduler = getattr(
            importlib.import_module("torch.optim.lr_scheduler"), cfg.Name
        )(optimizer, **cfg.Para)
    else:
        scheduler = Poly_scheduler(
            optimizer,
            **cfg.Para,
        )
    return scheduler


def build_loader(cfg, dist=False):
    dataset = getattr(Data, cfg.DataPara.name)(**cfg.DataPara)
    sampler = None
    is_shuffle = True
    if dist:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        is_shuffle = False
    if cfg.DataPara.state != "train":
        is_shuffle = False
    if cfg.DataPara.state == "train":
        sampler = WeightedRandomSampler(
            dataset.get_weighted_count(),
            num_samples=len(dataset),
            replacement=True,
        )
        is_shuffle = False
    dataloader = DataLoaderX(
        dataset,
        **cfg.LoaderPara,
        shuffle=is_shuffle,
        pin_memory=False,
        sampler=sampler,
        drop_last=False,
    )
    return dataloader
