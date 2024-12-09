import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot
import os
import lmdb
import glob
from loguru import logger
from tqdm import tqdm
import cv2
import random
import pickle
import torch
import torch.nn.functional as F
from Lib.lib import read_pickle, write_pickle, right_replace, write_json

if __name__ == "__main__":
    visuals = [256]
    temporals = [8, 16, 24, 32]
    dcms_dir = "/ai/mnt/data/dicom/pair/"  # need to be set
    dcms = glob.glob(dcms_dir + "/*.dcm")
    env = lmdb.open(os.path.realpath(__file__), map_size=1099511627776)
    lib = env.begin(write=True)
    meta_data = {}
    for visual in visuals:
        for temporal in temporals:
            print("Processing with temporal %d and visual as %d" % (temporal, visual))
            meta_data["T%02d#V%d#mean" % (temporal, visual)] = 0.0
            meta_data["T%02d#V%d#std" % (temporal, visual)] = 0.0

            for index, dcm in enumerate(tqdm(dcms)):
                name = os.path.basename(dcm)
                try:
                    pixel_array = sitk.GetArrayFromImage(sitk.ReadImage(dcm))
                    dcm_tensor = (
                        torch.tensor(pixel_array, dtype=torch.float32)
                        .permute(3, 0, 1, 2)
                        .unsqueeze(0)
                        .contiguous()
                    )
                except:
                    print(dcm)
                    continue

                dcm_tensor_temporal = F.interpolate(
                    dcm_tensor,
                    (temporal, visual, visual),
                    mode="trilinear",
                    align_corners=True,
                ).contiguous()[0]

                meta_data["T%02d#V%d#mean" % (temporal, visual)] = (
                    meta_data["T%02d#V%d#mean" % (temporal, visual)]
                    + dcm_tensor_temporal.mean()
                )
                meta_data["T%02d#V%d#std" % (temporal, visual)] = (
                    meta_data["T%02d#V%d#std" % (temporal, visual)]
                    + dcm_tensor_temporal.std()
                )

                lib.put(
                    key=("T%02d#V%d#" % (temporal, visual) + name).encode(),
                    value=pickle.dumps(dcm_tensor_temporal),
                )
                lib.put(
                    key=("RAW#" + name).encode(),
                    value=pickle.dumps(dcm_tensor),
                )
                if (index + 1) % 5 == 0:
                    lib.commit()
                    # commit 之后需要再次 begin
                    lib = env.begin(write=True)
        meta_data["T%02d#V%d#mean" % (temporal, visual)] = meta_data[
            "T%02d#V%d#mean" % (temporal, visual)
        ] / len(dcms)
        meta_data["T%02d#V%d#std" % (temporal, visual)] = meta_data[
            "T%02d#V%d#std" % (temporal, visual)
        ] / len(dcms)

    lib.put("meta_data".encode(), value=pickle.dumps(meta_data))
    lib.commit()
    env.close()
    print("DONE PREPRO AND WRITE")
