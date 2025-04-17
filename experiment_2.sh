#! /bin/bash

cd tools

export PYTHONWARNINGS=ignore::UserWarning:pcdet.ops.iou3d_nms.iou3d_nms_utils

python switching_overhead.py
