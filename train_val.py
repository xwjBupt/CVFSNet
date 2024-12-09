import torch
from tqdm import tqdm
import random
import pdb
from termcolor import cprint
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from Lib import MetricLogger, get_metrics


def train(
    net,
    view_mode,
    device,
    dataloader,
    optimizer,
    epoch,
    criterion,
    iter_dis,
    rank=0,
    wandb=None,
    **kwargs,
):
    loss_logger = MetricLogger()
    metrics = MetricLogger()
    data_item = len(dataloader)
    tbar = tqdm(dataloader, dynamic_ncols=True)
    gt_labels = []
    pred_labels_fuse = []
    pred_labels_cor = []
    pred_labels_sag = []
    net.train()
    name_pred_dict = {}
    for index, data in enumerate(tbar):
        label = data["label"].to(device, non_blocking=True)
        # if data["label"].squeeze(dim=-1).dim() >= 1:
        #     gt_labels.append(data["label"].squeeze(dim=-1))
        # else:
        #     gt_labels.append(data["label"])
        gt_labels.append(data["label"])
        if view_mode == 3:
            cor_fast_view = (
                data["cor_fast_view"].to(device, non_blocking=True).contiguous()
            )
            cor_slow_view = (
                data["cor_slow_view"].to(device, non_blocking=True).contiguous()
            )
            sag_fast_view = (
                data["sag_fast_view"].to(device, non_blocking=True).contiguous()
            )
            sag_slow_view = (
                data["sag_slow_view"].to(device, non_blocking=True).contiguous()
            )
            pred_logits = net(
                cor_fast_view, cor_slow_view, sag_fast_view, sag_slow_view
            )
        elif view_mode == 2:
            cor_view = data["cor_view"].to(device, non_blocking=True).contiguous()
            sag_view = data["sag_view"].to(device, non_blocking=True).contiguous()
            pred_logits = net([cor_view, sag_view])
        elif view_mode == 1:
            slow_view = data["slow_view"].to(device, non_blocking=True).contiguous()
            pred_logits = net(slow_view)
        elif view_mode == 0:
            fast_view = data["fast_view"].to(device, non_blocking=True).contiguous()
            pred_logits = net(fast_view)

        loss, loss_dict = criterion(pred_logits, label)
        loss_logger.update(**loss_dict)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 12)
        optimizer.step()

        pred_label_fuse = (
            torch.argmax(F.softmax(pred_logits[0], dim=-1), dim=-1).detach().cpu()
        )
        pred_label_cor = (
            torch.argmax(F.softmax(pred_logits[1], dim=-1), dim=-1).detach().cpu()
        )
        pred_label_sag = (
            torch.argmax(F.softmax(pred_logits[2], dim=-1), dim=-1).detach().cpu()
        )
        pred_labels_fuse.append(pred_label_fuse)
        pred_labels_cor.append(pred_label_cor)
        pred_labels_sag.append(pred_label_sag)
        pred_label_fuse = [str(i.item()) for i in list(pred_label_fuse)]
        for name, pred in zip(data["name"], pred_label_fuse):
            name_pred_dict[name] = pred
        if index % iter_dis == 0 and rank == 0:
            tbar.set_description(
                "TRAIN || RANK {} || Epoch {} || ITEM {}/{} || Loss: {}".format(
                    rank, epoch, index, data_item, loss_logger.lineout()
                )
            )

            # if wandb and epoch % (iter_dis * 10) == 0:
            #     sl = cor_view.shape[2] // 8
            #     raw_cor_view = (
            #         data["raw_cor_view"]
            #         .contiguous()
            #         .numpy()
            #         .astype(np.uint8)
            #         .transpose(0, 2, 1, 3, 4)
            #     )
            #     raw_sag_view = (
            #         data["raw_sag_view"]
            #         .contiguous()
            #         .numpy()
            #         .astype(np.uint8)
            #         .transpose(0, 2, 1, 3, 4)
            #     )
            #     trans_cor_view = (
            #         255
            #         * (data["cor_view"] - data["cor_view"].min())
            #         / (data["cor_view"].max() - data["cor_view"].min())
            #     )
            #     trans_cor_view = (
            #         trans_cor_view.contiguous()
            #         .numpy()
            #         .astype(np.uint8)
            #         .transpose(0, 2, 1, 3, 4)
            #     )

            #     trans_sag_view = (
            #         255
            #         * (data["sag_view"] - data["sag_view"].min())
            #         / (data["sag_view"].max() - data["cor_view"].min())
            #     )
            #     trans_sag_view = (
            #         trans_sag_view.contiguous()
            #         .numpy()
            #         .astype(np.uint8)
            #         .transpose(0, 2, 1, 3, 4)
            #     )

            #     wandb.log({"Raw Cor View": wandb.Video(raw_cor_view, fps=sl)})
            #     wandb.log({"Raw Sag View": wandb.Video(raw_sag_view, fps=sl)})
            #     wandb.log({"Trans Cor View": wandb.Video(trans_cor_view, fps=sl)})
            #     wandb.log({"Trans Sag View": wandb.Video(trans_sag_view, fps=sl)})

    gt_labels_flat = list(torch.cat(gt_labels, dim=0).squeeze(dim=-1).numpy())
    pred_labels_fuse_flat = list(torch.cat(pred_labels_fuse).numpy())
    pred_labels_cor_flat = list(torch.cat(pred_labels_cor).numpy())
    pred_labels_sag_flat = list(torch.cat(pred_labels_sag).numpy())
    metrics.update(**get_metrics(gt_labels_flat, pred_labels_fuse_flat, index=""))
    metrics.update(**get_metrics(gt_labels_flat, pred_labels_cor_flat, index="#COR"))
    metrics.update(**get_metrics(gt_labels_flat, pred_labels_sag_flat, index="#SAG"))
    return (
        net,
        loss_logger,
        gt_labels_flat,
        pred_labels_fuse_flat,
        pred_labels_cor_flat,
        pred_labels_sag_flat,
        metrics,
        name_pred_dict,
    )


@torch.no_grad()
def val(
    net,
    view_mode,
    device,
    dataloader,
    epoch,
    criterion,
    iter_dis,
    savename,
    rank=0,
    **kwargs,
):
    loss_logger = MetricLogger()
    metrics = MetricLogger()
    data_item = len(dataloader)
    tbar = tqdm(dataloader, dynamic_ncols=True)
    gt_labels = []
    pred_labels_fuse = []
    pred_labels_cor = []
    pred_labels_sag = []
    net.eval()
    name_pred_dict = {}
    for index, data in enumerate(tbar):
        label = data["label"].to(device, non_blocking=True)
        gt_labels.append(data["label"])
        if view_mode == 3:
            cor_fast_view = (
                data["cor_fast_view"].to(device, non_blocking=True).contiguous()
            )
            cor_slow_view = (
                data["cor_slow_view"].to(device, non_blocking=True).contiguous()
            )
            sag_fast_view = (
                data["sag_fast_view"].to(device, non_blocking=True).contiguous()
            )
            sag_slow_view = (
                data["sag_slow_view"].to(device, non_blocking=True).contiguous()
            )
            pred_logits = net(
                cor_fast_view, cor_slow_view, sag_fast_view, sag_slow_view
            )
        elif view_mode == 2:
            cor_view = data["cor_view"].to(device, non_blocking=True).contiguous()
            sag_view = data["sag_view"].to(device, non_blocking=True).contiguous()
            pred_logits = net([cor_view, sag_view])
        elif view_mode == 1:
            slow_view = data["slow_view"].to(device, non_blocking=True).contiguous()
            pred_logits = net(slow_view)
        elif view_mode == 0:
            fast_view = data["fast_view"].to(device, non_blocking=True).contiguous()
            pred_logits = net(fast_view)
        loss, loss_dict = criterion(pred_logits, label)
        loss_logger.update(**loss_dict)

        pred_label_fuse = (
            torch.argmax(F.softmax(pred_logits[0], dim=-1), dim=-1).detach().cpu()
        )
        pred_label_cor = (
            torch.argmax(F.softmax(pred_logits[1], dim=-1), dim=-1).detach().cpu()
        )
        pred_label_sag = (
            torch.argmax(F.softmax(pred_logits[2], dim=-1), dim=-1).detach().cpu()
        )
        pred_labels_fuse.append(pred_label_fuse)
        pred_labels_cor.append(pred_label_cor)
        pred_labels_sag.append(pred_label_sag)
        pred_label_fuse = [str(i.item()) for i in list(pred_label_fuse)]
        for name, pred in zip(data["name"], pred_label_fuse):
            name_pred_dict[name] = pred
        if index % iter_dis == 0 and rank == 0:
            tbar.set_description(
                "VAL || RANK {} || Epoch {} || ITEM {}/{} || Loss: {}".format(
                    rank, epoch, index, data_item, loss_logger.lineout()
                )
            )
    gt_labels_flat = list(torch.cat(gt_labels, dim=0).squeeze(dim=-1).numpy())
    pred_labels_fuse_flat = list(torch.cat(pred_labels_fuse).numpy())
    pred_labels_cor_flat = list(torch.cat(pred_labels_cor).numpy())
    pred_labels_sag_flat = list(torch.cat(pred_labels_sag).numpy())
    metrics.update(**get_metrics(gt_labels_flat, pred_labels_fuse_flat, index=""))
    metrics.update(**get_metrics(gt_labels_flat, pred_labels_cor_flat, index="#COR"))
    metrics.update(**get_metrics(gt_labels_flat, pred_labels_sag_flat, index="#SAG"))
    return (
        net,
        loss_logger,
        gt_labels_flat,
        pred_labels_fuse_flat,
        pred_labels_cor_flat,
        pred_labels_sag_flat,
        metrics,
        name_pred_dict,
    )
