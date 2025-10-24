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

from argparse import ArgumentParser
import os
import ctypes

import cv2
import numpy as np
import tensorrt as trt

from stereo_inference_base import BaseStereoInference, BaseStereoRunner

DEFAULT_ENGINE_FILE_PATH = os.getenv('ISAAC_ROS_WS', ".") \
    + '/isaac_ros_assets/models/dnn_stereo_disparity' \
    + '/dnn_stereo_disparity_v4.1.0_onnx_trt10.13/ess.engine'

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1200
NETWORK_WIDTH = 960
NETWORK_HEIGHT = 576
SECONDS_TO_NANOSECONDS = 1000000000


class ESSInference(BaseStereoInference):
    """TensorRT-based ESS inference for stereo depth estimation."""

    def __init__(self, engine_file_path, threshold=0.4, verbose=False):
        """Initialize TensorRT inference engine.

        Args:
            engine_file_path: Path to the TensorRT engine file
            threshold: Confidence threshold for depth filtering
            verbose: Enable verbose logging
        """
        self.threshold = threshold
        super().__init__(engine_file_path, verbose)

    def load_plugins(self):
        """Load TensorRT plugins needed for ESS model."""
        try:
            # Initialize TensorRT plugins registry
            trt.init_libnvinfer_plugins(None, "")

            # Load ESS specific plugins
            # Determine architecture
            import platform
            architecture = platform.machine()
            if architecture == 'aarch64':
                plugin_subdir = 'plugins/aarch64'
            else:
                plugin_subdir = 'plugins/x86_64'

            # Get engine directory and construct plugin path
            engine_dir = os.path.dirname(self.engine_file_path)
            ess_plugin_path = os.path.join(engine_dir, plugin_subdir, 'ess_plugins.so')

            if os.path.exists(ess_plugin_path):
                ctypes.CDLL(ess_plugin_path)  # Load library but don't keep reference
                if self.verbose:
                    print(f"Loaded ESS plugins from {ess_plugin_path}")
            else:
                print(f"Warning: ESS plugin not found at expected path: {ess_plugin_path}")
                print("This is an issue if running ESS >=4.0")

        except Exception as e:
            print(f"Warning: Error loading TensorRT plugins: {e}")
            print("Attempting to continue without explicit plugin loading...")

    def get_input_binding_names(self):
        """Get list of input binding names for the ESS model."""
        return ['input_left', 'input_right']

    def get_output_binding_names(self):
        """Get list of output binding names for the ESS model."""
        return ['output_left', 'output_conf']

    def preprocess_image(self, image):
        """Preprocess an image for ESS inference.

        Args:
            image: BGR image (HWC format)

        Returns:
            Tuple of (preprocessed image, preprocessing metadata)
        """
        # Resize to network input size
        resized = cv2.resize(image, (NETWORK_WIDTH, NETWORK_HEIGHT))

        # Convert to RGB format and normalize
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Convert to floating point and normalize
        rgb_float = rgb.astype(np.float32) / 255.0

        # Convert to CHW format (TensorRT expects NCHW)
        chw = np.transpose(rgb_float, (2, 0, 1))

        # Add batch dimension
        nchw = np.expand_dims(chw, axis=0)

        # Store preprocessing metadata (empty for ESS as it doesn't need coordinate transformation)
        preprocess_metadata = {}

        # Ensure array is contiguous in memory (important for CUDA operations)
        return np.ascontiguousarray(nchw), preprocess_metadata

    def process_inference_results(self, results, preprocess_metadata):
        """Process raw inference results from ESS model.

        Args:
            results: Raw inference results from GPU
            preprocess_metadata: Metadata from preprocessing (unused for ESS)

        Returns:
            Tuple of (disparity, confidence) arrays
        """
        return results['output_left'], results['output_conf']

    def apply_threshold(self, disparity, confidence):
        """Apply confidence threshold to disparity.

        Args:
            disparity: Disparity array
            confidence: Confidence array

        Returns:
            Filtered disparity array
        """
        # Apply confidence threshold
        mask = confidence < self.threshold
        filtered_disparity = disparity.copy()
        filtered_disparity[mask] = 0.0

        return filtered_disparity


class ESSRunner(BaseStereoRunner):
    """ESS stereo inference runner."""

    def __init__(self,
                 image_dir,
                 output_dir,
                 engine_file_path=DEFAULT_ENGINE_FILE_PATH,
                 ess_threshold=0.4,
                 frames_meta_file=None,
                 verbose=False):
        self.ess_threshold = ess_threshold
        super().__init__(image_dir, output_dir, engine_file_path, frames_meta_file, verbose)

    def create_inference_engine(self):
        """Create ESS inference engine."""
        return ESSInference(
            engine_file_path=self.engine_file_path,
            threshold=self.ess_threshold,
            verbose=self.verbose
        )

    def _process_stereo_pair(self, left_image, right_image, left_image_name,
                             focal_length, baseline):
        """Process a single stereo pair using ESS inference."""
        # Run ESS inference
        disparity, confidence = self.stereo_inference.infer(left_image, right_image)

        # Apply confidence threshold
        filtered_disparity = self.stereo_inference.apply_threshold(disparity[0], confidence[0])

        # Convert disparity to depth
        # Scale focal length to network output resolution
        scaled_focal_length = focal_length * (NETWORK_WIDTH / IMAGE_WIDTH)
        depth = self.convert_disparity_to_depth(
            filtered_disparity, scaled_focal_length, baseline)

        # Resize back to original resolution
        depth = cv2.resize(depth, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)

        # Save depth image
        output_path = self.output_dir + "/" + left_image_name.split('.')[0] + '.png'
        self.ensure_directory_exists_for_file(output_path)
        cv2.imwrite(output_path, depth.astype(np.uint16))

        if self.verbose:
            print(f"Saved depth image: {output_path}")

        # Increment counter
        self.image_count += 1
        if self.image_count % 100 == 0:
            print(
                "Processed",
                self.image_count,
                "image pairs, currently on",
                left_image_name,
            )


def parse_args():
    parser = ArgumentParser(prog='run_ess_offline.py')
    parser.add_argument('--image_dir',
                        help='directory containing image folders',
                        type=str,
                        required=True)
    parser.add_argument('--output_dir', help='output directory', type=str, required=True)
    parser.add_argument('--engine_file_path',
                        help='ESS engine file path',
                        type=str,
                        default=DEFAULT_ENGINE_FILE_PATH)
    parser.add_argument('--ess_threshold', help='Threshold for ESS', type=float, default=0.4)
    parser.add_argument('--frames_meta_file', help='frames meta file', type=str, default=None)
    parser.add_argument('--verbose', help='Enable verbose logging', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    with ESSRunner(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        engine_file_path=args.engine_file_path,
        ess_threshold=args.ess_threshold,
        frames_meta_file=args.frames_meta_file,
        verbose=args.verbose,
    ) as runner:
        runner.extract_data()


if __name__ == '__main__':
    main()
