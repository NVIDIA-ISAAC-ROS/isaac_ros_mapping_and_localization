#!/usr/bin/env python3

# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import cv2
import numpy as np
import tensorrt as trt

from stereo_inference_base import BaseStereoInference, BaseStereoRunner

DEFAULT_ENGINE_FILE_PATH = os.getenv('ISAAC_ROS_WS', ".") \
    + '/isaac_ros_assets/models/foundationstereo/' + \
    'deployable_foundation_stereo_small_v1.0/foundationstereo_576x960.engine'

# Model input specifications
MODEL_INPUT_WIDTH = 960
MODEL_INPUT_HEIGHT = 576
MODEL_NUM_CHANNELS = 3  # RGB channels

# ImageNet normalization parameters used in Foundation Stereo
IMAGENET_MEAN = [123.675, 116.28, 103.53]
IMAGENET_STDDEV = [58.395, 57.12, 57.375]


class FoundationStereoInference(BaseStereoInference):
    """TensorRT-based Foundation Stereo inference for stereo depth estimation."""

    def __init__(self, engine_file_path, verbose=False):
        """Initialize TensorRT inference engine.

        Args:
            engine_file_path: Path to the TensorRT engine file
            verbose: Enable verbose logging
        """
        super().__init__(engine_file_path, verbose)

    def load_plugins(self):
        """Load TensorRT plugins (Foundation Stereo doesn't require special plugins)."""
        try:
            # Initialize TensorRT plugins registry
            trt.init_libnvinfer_plugins(None, "")
        except Exception as e:
            print(f"Warning: Error loading TensorRT plugins: {e}")

    def get_input_binding_names(self):
        """Get list of input binding names for the Foundation Stereo model."""
        return ['left_image', 'right_image']

    def get_output_binding_names(self):
        """Get list of output binding names for the Foundation Stereo model."""
        return ['disparity']

    def preprocess_image(self, image):
        """Preprocess an image for Foundation Stereo inference.

        This follows the same preprocessing pipeline as defined in the launch file:
        1. Resize with aspect ratio preservation
        2. Pad to model input size
        3. Convert to RGB
        4. Normalize with ImageNet statistics
        5. Convert to CHW tensor format

        Args:
            image: BGR image (HWC format)

        Returns:
            Tuple of (preprocessed image, preprocessing metadata)
            - preprocessed image ready for inference
            - metadata dict containing scale, padding info for coordinate transformation
        """
        # Step 1: Resize with aspect ratio preservation
        # Calculate the scaling factor to fit within model input dimensions
        h, w = image.shape[:2]
        scale_w = MODEL_INPUT_WIDTH / w
        scale_h = MODEL_INPUT_HEIGHT / h
        scale = min(scale_w, scale_h)

        new_width = int(w * scale)
        new_height = int(h * scale)

        # Choose optimal interpolation method based on scaling direction
        if scale > 1.0:  # Upsampling - use cubic for better quality
            interpolation = cv2.INTER_CUBIC
        else:  # Downsampling - use area for better anti-aliasing
            interpolation = cv2.INTER_AREA

        resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)

        # Step 2: Pad to model input size with border replication
        # Calculate padding
        pad_w = MODEL_INPUT_WIDTH - new_width
        pad_h = MODEL_INPUT_HEIGHT - new_height

        # Pad with REPLICATE border type
        padded = cv2.copyMakeBorder(
            resized, 0, pad_h, 0, pad_w,
            cv2.BORDER_REPLICATE
        )

        # Step 3: Convert BGR to RGB
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

        # Step 4: Convert to float32 and normalize with ImageNet statistics
        rgb_float = rgb.astype(np.float32)

        # Apply ImageNet normalization: (pixel - mean) / stddev
        for i in range(3):
            rgb_float[:, :, i] = (rgb_float[:, :, i] - IMAGENET_MEAN[i]) / IMAGENET_STDDEV[i]

        # Step 5: Convert to CHW format (TensorRT expects NCHW)
        chw = np.transpose(rgb_float, (2, 0, 1))

        # Add batch dimension
        nchw = np.expand_dims(chw, axis=0)

        # Store preprocessing metadata for coordinate transformation
        preprocess_metadata = {
            'scale': scale,
            'resized_width': new_width,
            'resized_height': new_height,
            'pad_width': pad_w,
            'pad_height': pad_h,
            'original_width': w,
            'original_height': h
        }

        # Ensure array is contiguous in memory (important for CUDA operations)
        return np.ascontiguousarray(nchw), preprocess_metadata

    def process_inference_results(self, results, preprocess_metadata):
        """Process raw inference results from Foundation Stereo model.

        Args:
            results: Raw inference results from GPU
            preprocess_metadata: Metadata from preprocessing

        Returns:
            Tuple of (disparity array, preprocessing metadata)
        """
        return results['disparity'], preprocess_metadata

    def transform_disparity_to_original_coordinates(self, disparity, preprocess_metadata):
        """Transform disparity from model coordinate space back to original image coordinate space.

        Args:
            disparity: Disparity array from model output (in model resolution with padding)
            preprocess_metadata: Metadata from preprocessing containing scale and padding info

        Returns:
            Transformed disparity array in original image resolution
        """
        # Step 1: Remove padding from disparity (crop to actual resized content)
        resized_height = preprocess_metadata['resized_height']
        resized_width = preprocess_metadata['resized_width']

        # Crop disparity to remove padding
        disparity_cropped = disparity[:resized_height, :resized_width]

        # Step 2: Scale disparity values to account for the resize transformation
        # Disparity values need to be scaled by the same factor used for resizing
        scale = preprocess_metadata['scale']
        disparity_scaled = disparity_cropped / scale

        # Step 3: Resize to original image dimensions
        original_height = preprocess_metadata['original_height']
        original_width = preprocess_metadata['original_width']

        disparity_final = cv2.resize(
            disparity_scaled,
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST
        )

        return disparity_final


class FoundationStereoRunner(BaseStereoRunner):
    """Foundation Stereo inference runner."""

    def __init__(self,
                 image_dir,
                 output_dir,
                 engine_file_path=DEFAULT_ENGINE_FILE_PATH,
                 frames_meta_file=None,
                 verbose=False):
        super().__init__(image_dir, output_dir, engine_file_path, frames_meta_file, verbose)

    def create_inference_engine(self):
        """Create Foundation Stereo inference engine."""
        return FoundationStereoInference(
            engine_file_path=self.engine_file_path,
            verbose=self.verbose
        )

    def _process_stereo_pair(self, left_image, right_image, left_image_name,
                             focal_length, baseline):
        """Process a single stereo pair using Foundation Stereo inference."""
        if self.verbose:
            print(f"Processing images: {left_image_name}")
            print(f"Left image shape: {left_image.shape}")
            print(f"Right image shape: {right_image.shape}")

        # Get image dimensions
        image_height, image_width = left_image.shape[:2]

        if self.verbose:
            print(f"Original image size: {image_width}x{image_height}")

        # Run Foundation Stereo inference
        disparity, preprocess_metadata = self.stereo_inference.infer(left_image, right_image)

        if self.verbose:
            print(f"Raw disparity output shape: {disparity.shape}")
            print(f"Raw disparity min/max: {np.min(disparity):.3f}/{np.max(disparity):.3f}")

        # Process disparity output (remove batch dimension if present)
        if len(disparity.shape) == 4:  # [batch, height, width, channels] or [batch, channels, height, width]  # noqa: E501
            disparity = disparity[0]  # Remove batch dimension
        # Handle different channel arrangements
        if len(disparity.shape) == 3:
            if disparity.shape[0] == 1:  # [1, height, width] -> [height, width]
                disparity = disparity[0]
            elif disparity.shape[2] == 1:  # [height, width, 1] -> [height, width]
                disparity = disparity[:, :, 0]
            # If shape is [channels, height, width] and channels > 1, take first channel
            elif disparity.shape[0] > disparity.shape[2]:
                disparity = disparity[0]

        if self.verbose:
            print(f"Disparity shape after initial processing: {disparity.shape}")
            print(f"Disparity min/max: {np.min(disparity):.3f}/{np.max(disparity):.3f}")

        # Transform disparity from model coordinates back to original image coordinates
        disparity = (
            self.stereo_inference
            .transform_disparity_to_original_coordinates(
                disparity, preprocess_metadata))

        if self.verbose:
            print(f"Disparity shape after coordinate transformation: {disparity.shape}")
            print(
                f"Transformed disparity min/max: "
                f"{np.min(disparity):.3f}/{np.max(disparity):.3f}")

        # Convert disparity to depth using original focal length (no scaling needed)
        depth = self.convert_disparity_to_depth(disparity, focal_length, baseline)

        if self.verbose:
            print(f"Final depth image shape: {depth.shape}")

        # Ensure depth is 2D
        if len(depth.shape) != 2:
            raise ValueError(f"Expected 2D depth array, got shape {depth.shape}")

        # Verify depth is already in original image dimensions
        if depth.shape != (image_height, image_width):
            raise ValueError(
                f"Depth shape {depth.shape} doesn't match expected "
                f"{(image_height, image_width)}")

        # Save depth image
        output_path = self.output_dir + "/" + left_image_name.split('.')[0] + '.png'
        self.ensure_directory_exists_for_file(output_path)
        cv2.imwrite(output_path, depth.astype(np.uint16))

        if self.verbose:
            print(f"Saved depth image: {output_path}")
            print(f"Final depth image size: {depth.shape}")
            print(f"Depth value range: {np.min(depth):.1f} - {np.max(depth):.1f} mm")

        # Increment counter
        self.image_count += 1
        if self.image_count % 10 == 0:
            print(
                "Processed",
                self.image_count,
                "image pairs, currently on",
                left_image_name,
            )


def parse_args():
    parser = ArgumentParser(prog='run_foundationstereo_trt_offline.py')
    parser.add_argument('--image_dir',
                        help='directory containing image folders',
                        type=str,
                        required=True)
    parser.add_argument('--output_dir', help='output directory', type=str, required=True)
    parser.add_argument('--engine_file_path',
                        help='Foundation Stereo engine file path',
                        type=str,
                        default=DEFAULT_ENGINE_FILE_PATH)
    parser.add_argument('--frames_meta_file', help='frames meta file', type=str, default=None)
    parser.add_argument('--verbose', help='Enable verbose logging', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    with FoundationStereoRunner(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        engine_file_path=args.engine_file_path,
        frames_meta_file=args.frames_meta_file,
        verbose=args.verbose,
    ) as runner:
        runner.extract_data()


if __name__ == '__main__':
    main()
