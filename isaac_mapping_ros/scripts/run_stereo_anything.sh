#!/bin/bash

# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

DATA_DIR=${1? "Directory for reading raw images and writing depth images"}

cd /workspaces/ncore-converter

cp -r pretrained_models/ dependency_all/stereo_anything_inference/

python dependency_all/stereo_anything_inference/scripts/run_sfm_data_inference.py  \
    --ckpt_dir pretrained_models/stereo-anything/model_best.pth \
    --imgdir $DATA_DIR/raw \
    --metadata_file $DATA_DIR/keyframes/frames_meta.json \
    --out_dir $DATA_DIR/depth \
    --camera "front_stereo_camera_left,left_stereo_camera_left,right_stereo_camera_left,back_stereo_camera_left" \
    --num_gpus 2
