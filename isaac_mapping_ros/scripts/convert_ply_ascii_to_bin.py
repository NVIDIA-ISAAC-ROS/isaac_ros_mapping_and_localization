#!/usr/bin/env python3

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

# This script converts a ply in ASCII to bin format to load in CloudCompare.
# Install plyfile before running the script.
# Usage:
# python
# src/isaac_ros_mapping_and_localization/isaac_mapping_ros/scripts/convert_ply_ascii_to_bin.py
# <ply file>

import argparse
import os
import sys

import plyfile


def convert_file_path(file_path):
    # Split the file path into directory, filename, and extension
    dir_name, base_name = os.path.split(file_path)
    file_name, file_ext = os.path.splitext(base_name)

    # Add '_bin' to the file name
    new_file_name = file_name + '_bin' + file_ext

    # Join the directory and the new file name
    new_file_path = os.path.join(dir_name, new_file_name)

    return new_file_path


def main(args):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('input_ply', type=str)
    args = arg_parser.parse_args()

    data = plyfile.PlyData.read(args.input_ply)
    data.text = False
    output_filename = convert_file_path(args.input_ply)
    print('output filename', output_filename)
    data.write(output_filename)


if __name__ == '__main__':
    main(sys.argv[1:])
