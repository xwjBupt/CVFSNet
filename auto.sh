#!/bin/bash
# T08V256
python main.py --no_debug --opts LOSS.Para.p 0.8 LOSS.Para.q 1.0 BASIC.Seed 14207 METHOD.Desc DUAL_VIEW-FUSE01-T08#V256-RENAMED/CVFM-LabelSmoothSeasaw_MISO-Loss_p0.8_Q1.0-try1 DATA.Train.DataPara.fast_time_size 8 DATA.Train.DataPara.visual_size 256 DATA.Train.LoaderPara.batch_size 4 DATA.Train.LoaderPara.num_workers 8 DATA.Val.DataPara.fast_time_size 8 DATA.Val.DataPara.visual_size 256 DATA.Val.LoaderPara.batch_size 4 DATA.Val.LoaderPara.num_workers 8 MODEL.Para.input_clip_length 8 MODEL.Para.input_crop_size 256
python main.py --no_debug --opts LOSS.Para.p 0.8 LOSS.Para.q 1.0 BASIC.Seed 1207 METHOD.Desc DUAL_VIEW-FUSE01-T08#V256-RENAMED/CVFM-LabelSmoothSeasaw_MISO-Loss_p0.8_Q1.0-try2 DATA.Train.DataPara.fast_time_size 8 DATA.Train.DataPara.visual_size 256 DATA.Train.LoaderPara.batch_size 4 DATA.Train.LoaderPara.num_workers 8 DATA.Val.DataPara.fast_time_size 8 DATA.Val.DataPara.visual_size 256 DATA.Val.LoaderPara.batch_size 4 DATA.Val.LoaderPara.num_workers 8 MODEL.Para.input_clip_length 8 MODEL.Para.input_crop_size 256
python main.py --no_debug --opts LOSS.Para.p 0.8 LOSS.Para.q 1.0 BASIC.Seed 1407 METHOD.Desc DUAL_VIEW-FUSE01-T08#V256-RENAMED/CVFM-LabelSmoothSeasaw_MISO-Loss_p0.8_Q1.0-try3 DATA.Train.DataPara.fast_time_size 8 DATA.Train.DataPara.visual_size 256 DATA.Train.LoaderPara.batch_size 4 DATA.Train.LoaderPara.num_workers 8 DATA.Val.DataPara.fast_time_size 8 DATA.Val.DataPara.visual_size 256 DATA.Val.LoaderPara.batch_size 4 DATA.Val.LoaderPara.num_workers 8 MODEL.Para.input_clip_length 8 MODEL.Para.input_crop_size 256









