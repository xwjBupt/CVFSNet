import pickle
import os.path as osp
import glob
import tqdm
from git import Repo
import glob
import yaml
import shutil
import json
import torch.nn.functional as F
from collections import defaultdict
from collections import deque
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import pdb
import os
import csv
from multiprocessing import Pool
from typing import Callable, List, Optional, Tuple
from functools import partial


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count


class Averagvalue(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricLogger(object):
    def __init__(self, delimiter=" "):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def flush(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "  [ %s ] [ avg : %.4f ] [ meidan : %.4f ]\n"
                % (name, meter.global_avg, meter.median)
            )
        return self.delimiter.join(loss_str)

    def lineout(self):
        loss_str = []
        loss_str.append("[")
        for name, meter in self.meters.items():
            loss_str.append(" %s:%.4f " % (name, meter.global_avg))
        loss_str.append("]")
        return self.delimiter.join(loss_str)

    def return_dict(self, method=None):
        output = {}
        if method:
            output.update(method)
        for name, meter in self.meters.items():
            output[name] = meter.global_avg
        return output


def get_learning_rate(optimizer):
    """return right now lr"""
    return optimizer.param_groups[0]["lr"]


def right_replace(string, old, new, max=1):
    return string[::-1].replace(old[::-1], new[::-1], max)[::-1]


def write_json(content, outfile):
    with open(outfile, "w") as f:
        json.dump(content, f)


def read_json(input_file):
    with open(input_file, "r") as f:
        content = json.load(f)
        return content


def write_pickle(content, outfile):
    f = open(outfile, "wb")
    data = {"data": content}
    pickle.dump(data, f)
    f.close()


def read_pickle(inputfile):
    f = open(inputfile, "rb")
    data = pickle.load(f, encoding="bytes")
    f.close()
    return data["data"]


def update_config(cfg, args):
    cfg.defrost()
    # cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def warmup_lr_changer(maxlr, epoch, warmup_epoch=10):
    last_epoch = -1
    newlr = maxlr * math.e ** ((epoch - last_epoch - warmup_epoch) / warmup_epoch)
    if epoch >= warmup_epoch:
        newlr = maxlr

    return newlr


def nodeTodict(cfg, result={}):
    for k, v in cfg.items():
        if type(v) == type(cfg):
            temp = nodeTodict(v, {})
            result[k] = temp
        else:
            result[k] = v
    return result


def read_yaml(yaml_file):
    with open(yaml_file, encoding="utf8") as a_yaml_file:
        parsed_yaml_file = yaml.load(a_yaml_file, Loader=yaml.FullLoader)
    return parsed_yaml_file


def write_yaml(yaml_file, content):
    with open(yaml_file, "w", encoding="utf-8") as f:
        yaml.dump(content, f)


def iter_update_dict(new_dict, raw_dict):
    for k, v in raw_dict.items():
        if type(v) == dict:
            iter_update_dict(new_dict, v)
        else:
            if k in new_dict:
                raw_dict[k] = new_dict[k]
    return raw_dict


def git_commit(
    work_dir,
    timestamp,
    levels=5,
    postfixs=[".py", ".sh"],
    commit_info="",
    debug=False,
):
    cid = "not generate"
    branch = "master"
    if not debug:
        repo = Repo(work_dir)
        toadd = []
        branch = repo.active_branch.name
        for i in range(levels):
            for postfix in postfixs:
                filename = glob.glob(work_dir + (i + 1) * "/*" + postfix)
                for x in filename:
                    if (
                        not ("play" in x)
                        and not ("local" in x)
                        and not ("Untitled" in x)
                        and not ("wandb" in x)
                    ):
                        toadd.append(x)
        index = repo.index
        index.add(toadd)
        index.commit(commit_info)
        cid = repo.head.commit.hexsha

    commit_tag = (
        commit_info
        + "\n"
        + "COMMIT BRANCH >>> "
        + branch
        + " <<< \n"
        + "COMMIT ID >>> "
        + cid
        + " <<<"
    )
    record_commit_info = "\n" + "***" * 10 + "\n%s\n" % commit_tag + "***" * 10 + "\n"
    return record_commit_info


def update_config(cfg, args):
    cfg.defrost()
    # cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def nodeTodict(cfg, result={}):
    for k, v in cfg.items():
        if type(v) == type(cfg):
            temp = nodeTodict(v, {})
            result[k] = temp
        else:
            result[k] = v
    return result


def read_yaml(yaml_file):
    with open(yaml_file, encoding="utf8") as a_yaml_file:
        parsed_yaml_file = yaml.load(a_yaml_file, Loader=yaml.FullLoader)
    return parsed_yaml_file


def write_yaml(yaml_file, content):
    with open(yaml_file, "w", encoding="utf-8") as f:
        yaml.dump(content, f)


def iter_update_dict(new_dict, raw_dict):
    for k, v in raw_dict.items():
        if type(v) == dict:
            iter_update_dict(new_dict, v)
        else:
            if k in new_dict:
                raw_dict[k] = new_dict[k]
    return raw_dict


def save_checkpoint(
    save_model,
    epoch,
    save_dict,
    best_loss_flag=False,
    best_acc_flag=False,
    best_recall_flag=False,
    best_f1_flag=False,
    cover_up=True,
):
    if best_acc_flag:
        if cover_up:
            exists_models = glob.glob(save_model + "/Best_Acc_Epoch*.pth")
            for exists_model in exists_models:
                os.remove(exists_model)
        save_name = save_model + "/Best_Acc_Epoch_%04d.pth" % (epoch)
        torch.save(save_dict, save_name)
    elif best_recall_flag:
        if cover_up:
            exists_models = glob.glob(save_model + "/Best_Recall_Epoch*.pth")
            for exists_model in exists_models:
                os.remove(exists_model)
        save_name = save_model + "/Best_Recall_Epoch_%04d.pth" % (epoch)
        torch.save(save_dict, save_name)
    elif best_loss_flag:
        if cover_up:
            exists_models = glob.glob(save_model + "/Best_Loss_Epoch*.pth")
            for exists_model in exists_models:
                os.remove(exists_model)
        save_name = save_model + "/Best_Loss_Epoch_%04d.pth" % (epoch)
        torch.save(save_dict, save_name)
    elif best_f1_flag:
        if cover_up:
            exists_models = glob.glob(save_model + "/Best_F1_Epoch*.pth")
            for exists_model in exists_models:
                os.remove(exists_model)
        save_name = save_model + "/Best_F1_Epoch_%04d.pth" % (epoch)
        torch.save(save_dict, save_name)
    else:
        save_name = save_model + "/checkpoint.pth"
        torch.save(save_dict, save_name)


def iter_wandb_log(wandb, metric, phase="Train", index=1):
    towrite = {"index": index}
    for k, v in metric.meters.items():
        towrite[phase + "_" + k] = v.global_avg
    wandb.log(towrite)


def write_to_csv(filename, content):
    file_exist = os.path.exists(filename)
    with open(filename, "a+", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=content.keys())
        if not file_exist:
            writer.writeheader()
        writer.writerow(content)


def fuse_order_rule(
    fuse_pre, ova_pre=None, num_classes=4, less_confi=0.3, high_confi=0.85, **kwargs
):
    """
    input must be logit not tags, aka before sigmoid or softmax
    current only supports bs==1
    """
    if ova_pre is None:
        return torch.argmax(F.softmax(fuse_pre, dim=-1), dim=-1).detach().cpu()
    else:
        assert (
            fuse_pre.shape[0] == ova_pre.shape[0]
        ), "fuse_pre has different batch dim with ova_pre"
        batch = fuse_pre.shape[0]
        final_tags = []
        for b in range(batch):
            fuse_logit = F.softmax(fuse_pre[b], dim=-1).detach().cpu()
            fuse_tag = torch.argmax(fuse_logit, dim=-1)

            ova_logits = F.sigmoid(ova_pre[b]).detach().cpu()
            ### rule filter ###
            if fuse_logit[0] > high_confi and ova_logits[0] > high_confi:
                final_tags.append(0)


def imap_tqdm(
    function, iterable, processes, chunksize=1, desc=None, disable=False, **kwargs
):
    """
    Run a function in parallel with a tqdm progress bar and an arbitrary number of arguments.
    Results are always ordered and the performance should be the same as of Pool.map.
    :param function: The function that should be parallelized.
    :param iterable: The iterable passed to the function.
    :param processes: The number of processes used for the parallelization.
    :param chunksize: The iterable is based on the chunk size chopped into chunks and submitted to the process pool as separate tasks.
    :param desc: The description displayed by tqdm in the progress bar.
    :param disable: Disables the tqdm progress bar.
    :param kwargs: Any additional arguments that should be passed to the function.
    """
    if kwargs:
        function_wrapper = partial(_wrapper, function=function, **kwargs)
    else:
        function_wrapper = partial(_wrapper, function=function)

    results = [None] * len(iterable)
    with Pool(processes=processes) as p:
        with tqdm.tqdm(
            desc=desc,
            total=len(iterable),
            disable=disable,
            dynamic_ncols=True,
        ) as pbar:
            for i, result in p.imap_unordered(
                function_wrapper, enumerate(iterable), chunksize=chunksize
            ):
                results[i] = result
                pbar.update()
    return results


def _wrapper(enum_iterable, function, **kwargs):
    i = enum_iterable[0]
    result = function(enum_iterable[1], **kwargs)
    return i, result
