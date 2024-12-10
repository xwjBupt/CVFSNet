from yacs.config import CfgNode as CN
from collections import OrderedDict
import os
import matplotlib
import pdb
from tqdm import tqdm
from matplotlib import cm
import matplotlib.pyplot as plt
import argparse
from loguru import logger
import numpy as np
import torch
import csv
import glob
from operator import itemgetter
import json
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

# matplotlib.use('TkAgg')
# personal package
from builder import build_loader, build_model
from Lib import (
    MetricLogger,
    get_metrics,
    write_to_csv,
    write_json,
    plot_confusion_matrix,
)


def get_logits_to_label(
    csvname, pred_label_fuse_logits, pred_label_fuse_tags, gt_labels, names=None
):
    pred_label_fuse_logits = torch.cat(pred_label_fuse_logits, dim=0)
    num_classes = pred_label_fuse_logits.shape[-1]
    gt_labels = torch.cat(gt_labels, dim=0)
    pred_label_fuse_tags = torch.cat(pred_label_fuse_tags, dim=0)
    logits_to_label = torch.cat(
        (
            pred_label_fuse_logits,
            pred_label_fuse_tags.unsqueeze(1),
            gt_labels.unsqueeze(1),
        ),
        dim=1,
    )
    bs = logits_to_label.shape[0]
    for b in range(bs):
        if num_classes == 5:
            content = dict(
                name=names[b].split("/")[-1],
                T0=logits_to_label[b][0].item(),
                T1=logits_to_label[b][1].item(),
                T2a=logits_to_label[b][2].item(),
                T2b=logits_to_label[b][3].item(),
                T3=logits_to_label[b][4].item(),
                Pred=logits_to_label[b][-2].item(),
                Gt=logits_to_label[b][-1].item(),
                diff=abs(logits_to_label[b][-2].item() - logits_to_label[b][-1].item()),
            )
        else:
            content = dict(
                name=names[b].split("/")[-1],
                T0=logits_to_label[b][0].item(),
                T1=-1,
                T2a=logits_to_label[b][1].item(),
                T2b=logits_to_label[b][2].item(),
                T3=logits_to_label[b][3].item(),
                Pred=logits_to_label[b][-2].item(),
                Gt=logits_to_label[b][-1].item(),
                diff=abs(logits_to_label[b][-2].item() - logits_to_label[b][-1].item()),
            )
        write_to_csv(csvname, content)
    return logits_to_label


@logger.catch()
def infer(
    infer_path, logger, state="test", device="cuda", epochs=-1, args=None, **kwargs
):
    model_paths = []
    if args is None:
        args = "NONE"
    if epochs == -1:
        model_paths = glob.glob(os.path.join(infer_path, "Model/Best_*_Epoch*.pth"))
    else:
        model_paths.append(
            glob.glob(os.path.join(infer_path, "Model/Best_*_Epoch_%04d.pth" % epochs))[
                0
            ]
        )
    logger.info(">>> Going to infer with args as {}".format(args))
    logger.info(model_paths)
    metrics = None
    name_pred_dict = None
    for mp in model_paths:
        if "Loss" in mp:
            phase_index = "BEST_LOSS"
        elif "Acc" in mp:
            if "Fuse" in mp:
                phase_index = "BEST_ACC_FUSE"
            else:
                phase_index = "BEST_ACC"
        elif "Recall" in mp:
            phase_index = "BEST_Recall"
        elif "F1" in mp:
            phase_index = "BEST_F1"
        else:
            phase_index = "BEST"

        epoch_nr = mp.split("_")[-1][:-4]
        checkpoint = torch.load(mp, map_location=torch.device("cpu"))
        cfg = checkpoint["cfg"]
        cfg.defrost()
        cfg.MODEL.Para.sag_pretrained = None
        cfg.MODEL.Para.cor_pretrained = None
        cfg.MODEL.Para.fuse_pretrained = None
        cfg.MODEL.Para.all_pretrained = None
        cfg.DATA.Val.DataPara.json_file_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "Data", "ReNamedAll.json"
        )
        cfg.freeze()
        net = build_model(cfg.MODEL)
        pre_net_dict = checkpoint["net_dict"]
        raw_net_dict = net.state_dict()
        new_state_dict = OrderedDict()
        unloaded_keys = []
        for k, v in pre_net_dict.items():
            if "module" in k and k[7:] in raw_net_dict.keys():
                new_state_dict[k[7:]] = v
            elif k in raw_net_dict.keys():
                new_state_dict[k] = v
            else:
                unloaded_keys.append(k)
        raw_net_dict.update(new_state_dict)
        net.load_state_dict(raw_net_dict)
        net.to(device)
        net.eval()
        logger.info(
            "loaded pretrained weights, unloaded keys: {}".format(unloaded_keys)
        )
        logger.info("\n\n>>> Start infer %s <<<" % mp)
        csvname = None

        logger.info("\n\n >>> Infer on {} <<<".format(cfg.DATA.Val))
        infer_dataloader = build_loader(cfg.DATA.Val)
        state = "val"
        csvname = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            cfg.DATA.Val.Class + "_results.csv",
        )
        tbar = tqdm(infer_dataloader, dynamic_ncols=True)
        metrics = MetricLogger()
        gt_labels = []
        pred_label_fuse_logits = []
        pred_label_fuse_tags = []
        pred_label_fuse_ova_logits = []
        names = []
        pred_labels_fuse = []
        pred_labels_cor = []
        pred_labels_sag = []
        name_pred_dict = {}
        with torch.no_grad():
            for index, data in enumerate(tbar):
                label = data.get("label")
                names = names + data.get("name")
                if label is not None:
                    gt_labels.append(label.squeeze())
                cor_view = data["cor_view"].to(device, non_blocking=True).contiguous()
                sag_view = data["sag_view"].to(device, non_blocking=True).contiguous()
                pred_logits = net([cor_view, sag_view])

                pred_label_fuse = (
                    torch.argmax(F.softmax(pred_logits[0], dim=-1), dim=-1)
                    .detach()
                    .cpu()
                )
                pred_label_fuse_logits.append(
                    F.softmax(pred_logits[0], dim=-1).detach().cpu()
                )
                if pred_logits[-1] is not None:
                    pred_label_fuse_ova_logits.append(
                        F.sigmoid(pred_logits[-1]).detach().cpu()
                    )
                pred_label_fuse_tags.append(pred_label_fuse)
                pred_label_cor = (
                    torch.argmax(F.softmax(pred_logits[1], dim=-1), dim=-1)
                    .detach()
                    .cpu()
                )
                pred_label_sag = (
                    torch.argmax(F.softmax(pred_logits[2], dim=-1), dim=-1)
                    .detach()
                    .cpu()
                )
                pred_labels_fuse.append(pred_label_fuse)
                pred_labels_cor.append(pred_label_cor)
                pred_labels_sag.append(pred_label_sag)
                pred_label_fuse = [str(i.item()) for i in list(pred_label_fuse)]
                for name, pred in zip(data["name"], pred_label_fuse):
                    name_pred_dict[name] = pred

            if label is not None:
                gt_labels_flat = list(torch.cat(gt_labels).numpy())
                pred_labels_fuse_flat = list(torch.cat(pred_labels_fuse).numpy())
                pred_labels_cor_flat = list(torch.cat(pred_labels_cor).numpy())
                pred_labels_sag_flat = list(torch.cat(pred_labels_sag).numpy())
                infer_fuse_cm = confusion_matrix(gt_labels_flat, pred_labels_fuse_flat)
                infer_cor_cm = confusion_matrix(gt_labels_flat, pred_labels_cor_flat)
                infer_sag_cm = confusion_matrix(gt_labels_flat, pred_labels_sag_flat)
                infer_cm = [infer_fuse_cm, infer_cor_cm, infer_sag_cm]
                plot_confusion_matrix(
                    infer_cm,
                    savename=infer_path
                    + "/Epoch_%s#%s_%s_confusion_matrix.png"
                    % (epoch_nr, phase_index, state),
                )
                # metrics.update(**get_metrics(gt_labels_flat, pred_labels_fuse_flat))
                metrics.update(
                    **get_metrics(
                        gt_labels_flat,
                        pred_labels_fuse_flat,
                        pred_logit=torch.cat(pred_label_fuse_logits, 0).numpy(),
                        save_path=infer_path
                        + "/Epoch_%s#%s_infer_auc.png" % (epoch_nr, phase_index),
                    )
                )
                if csvname:
                    write_to_csv(
                        csvname,
                        dict(
                            METHOD=mp,
                            ACC=metrics.acc.avg,
                            ACC_SMOOTH=metrics.acc_smooth.avg,
                            RECALL=metrics.recall.avg,
                            F1=metrics.f1.avg,
                            AUC=metrics.auc.avg,
                        ),
                    )
                    logger.info(
                        ">>> Infer {} \n on state {} Results: {} \n write result to {} <<< \n".format(
                            mp,
                            state,
                            metrics.lineout(),
                            csvname,
                        )
                    )
                else:
                    logger.info(
                        ">>> Infer {} \n on state {} Results: {} \n".format(
                            mp,
                            state,
                            metrics.lineout(),
                        )
                    )
            logits_to_label = get_logits_to_label(
                infer_path
                + "/Epoch_%s#%s_%s_logits_to_label.csv"
                % (epoch_nr, phase_index, state),
                pred_label_fuse_logits,
                pred_label_fuse_tags,
                gt_labels,
                names=names,
            )
            logger.info(
                " {} logits_to_label with format as [ pred_label_fuse_logits - pred_label_fuse_tags(1) - gt_labels(1) ]\n{}".format(
                    state, logits_to_label
                )
            )
            if len(pred_label_fuse_ova_logits) > 1:
                ova_logits_to_label = get_logits_to_label(
                    infer_path
                    + "/Epoch_%s#%s_%s_logits_ova_to_label.csv"
                    % (epoch_nr, phase_index, state),
                    pred_label_fuse_ova_logits,
                    pred_label_fuse_tags,
                    gt_labels,
                    names=names,
                )
                logger.info(
                    "{} ova_logits_to_label with format as [ pred_label_fuse_ova_logits - pred_label_fuse_tags(1) - gt_labels(1) ]\n{}".format(
                        state, ova_logits_to_label
                    )
                )

            write_json(
                name_pred_dict,
                outfile=infer_path
                + "/Epoch_%s#%s_%s_results.json" % (epoch_nr, phase_index, state),
            )
            logger.info(
                ">>> Finish infer {} write result to {} <<<\n\n".format(
                    mp,
                    infer_path
                    + "/Epoch_%s#%s_%s_results.json" % (epoch_nr, phase_index, state),
                )
            )
    return metrics, name_pred_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer your model")
    parser.add_argument(
        "--infer_path",
        default="/ai/mnt/code/DSFNet_MTICI/output_runs/mTICI_Dual_LMDB/DVCNet/DUAL_VIEW-ALL-fuse01-T08#V512-RENAMED/11_11-10_55#COR_Kinect-SAG_Kinect-CVFMTrans_(drop0,head8,expand8,4STPE)_OptScaler20-LabelSmoothSeasaw_MISO_blance_0.8,1.2_q1-TIO_AUG_stablenorm_EroDilK3-noclip-DS13-init0914-fixdataset-fixmlp-fixloss/",  # 'Vessel/ToTest/UNetWithAttetnion/02_19-17_52/',
        help="the path of experiments to infer",
        type=str,
    )
    parser.add_argument("--epochs", default=-1, type=int)
    parser.add_argument("--state", default="test", type=str)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument(
        "--ensemble",
        default=True,
        help="save the raw output(before sigmoid or softmax) for later ensemble",
    )
    args = parser.parse_args()
    print(args)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    logger.add(args.infer_path + "/output.log")

    metrics = infer(
        infer_path=args.infer_path,
        logger=logger,
        state=args.state,
        epochs=args.epochs,
        args=args,
    )
