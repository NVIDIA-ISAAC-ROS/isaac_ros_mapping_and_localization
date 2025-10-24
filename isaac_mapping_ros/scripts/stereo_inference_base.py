#!/usr/bin/env python3

# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Base classes for TensorRT-based stereo inference implementations."""

from abc import ABC, abstractmethod
import os
import json

import cv2
import numpy as np
import tensorrt as trt
from cuda.bindings import driver as cuda_driver
from cuda.bindings import runtime as cuda_runtime
from scipy.spatial.transform import Rotation as R


class BaseStereoInference(ABC):
    """Base class for TensorRT-based stereo inference."""

    def __init__(self, engine_file_path, verbose=False):
        """Initialize TensorRT inference engine.

        Args:
            engine_file_path: Path to the TensorRT engine file
            verbose: Enable verbose logging
        """
        self.engine_file_path = engine_file_path
        self.verbose = verbose
        self._initialized = False

        # Check CUDA availability
        self.check_cuda_availability()

        # Load model-specific plugins
        self.load_plugins()

        # Initialize TensorRT
        self.logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
        self.runtime = trt.Runtime(self.logger)

        # Load engine
        self._load_engine()

        # Create execution context
        self.context = self.engine.create_execution_context()

        # Set input shapes for TensorRT >= 8.x
        self._set_input_shapes()

        # Get input and output binding information
        self.input_binding_names = self.get_input_binding_names()
        self.output_binding_names = self.get_output_binding_names()

        # Allocate device memory
        self.setup_io_bindings()

        if verbose:
            print(f"{self.__class__.__name__} TensorRT inference initialized with engine: "
                  f"{engine_file_path}")
            print(f"TensorRT version: {trt.__version__}")
            print(f"Input shapes: {self.input_shapes}")
            print(f"Output shapes: {self.output_shapes}")

        self._initialized = True

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()

    def cleanup(self):
        """Clean up CUDA resources and free allocated memory."""
        if not self._initialized:
            return

        try:
            # Free all allocated CUDA memory
            if hasattr(self, 'inputs') and self.inputs:
                for name, mem_info in self.inputs.items():
                    try:
                        err, = cuda_runtime.cudaFree(mem_info[0])
                        if err != cuda_runtime.cudaError_t.cudaSuccess:
                            print(f"Warning: Failed to free input memory for {name}: "
                                  f"error code {err}")
                    except Exception as e:
                        print(f"Warning: Failed to free input memory for {name}: {e}")

            if hasattr(self, 'outputs') and self.outputs:
                for name, mem_info in self.outputs.items():
                    try:
                        err, = cuda_runtime.cudaFree(mem_info[0])
                        if err != cuda_runtime.cudaError_t.cudaSuccess:
                            print(f"Warning: Failed to free output memory for {name}: "
                                  f"error code {err}")
                    except Exception as e:
                        print(f"Warning: Failed to free output memory for {name}: {e}")

            # Clean up TensorRT resources
            if hasattr(self, 'context') and self.context:
                del self.context
            if hasattr(self, 'engine') and self.engine:
                del self.engine

            # Clean up CUDA stream
            if hasattr(self, 'stream_handle') and self.stream_handle:
                err, = cuda_runtime.cudaStreamDestroy(self.stream_handle)
                if err != cuda_runtime.cudaError_t.cudaSuccess:
                    print(f"Warning: Failed to destroy CUDA stream: error code {err}")

        except Exception as e:
            print(f"Warning: Error during CUDA resource cleanup: {e}")
        finally:
            self._initialized = False

    def __del__(self):
        """Fallback cleanup in case context manager is not used."""
        self.cleanup()

    def check_cuda_availability(self):
        """Check if CUDA is available and properly initialized."""
        try:
            # Initialize CUDA
            err, = cuda_driver.cuInit(0)
            if err != cuda_driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"CUDA initialization failed with error code {err}")

            # Get CUDA device count
            err, device_count = cuda_driver.cuDeviceGetCount()
            if err != cuda_driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"Failed to get device count: error code {err}")

            if device_count == 0:
                raise RuntimeError("No CUDA devices found")

            # Get current device for information purposes
            err, device = cuda_driver.cuDeviceGet(0)  # Use first device by default
            if err != cuda_driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"Failed to get device: error code {err}")

            if self.verbose:
                err, name = cuda_driver.cuDeviceGetName(100, device)
                if err != cuda_driver.CUresult.CUDA_SUCCESS:
                    name = "Unknown"
                else:
                    name = name.decode()

                print(f"Using CUDA device: {name}")
                print(f"CUDA device count: {device_count}")

        except Exception as e:
            raise RuntimeError(f"CUDA initialization failed: {str(e)}")

    def _load_engine(self):
        """Load TensorRT engine from file."""
        try:
            with open(self.engine_file_path, 'rb') as f:
                engine_data = f.read()
                self.engine = self.runtime.deserialize_cuda_engine(engine_data)

                if self.engine is None:
                    trt_version = trt.__version__
                    raise RuntimeError(
                        f"Failed to load TensorRT engine: engine is None. "
                        f"Current TensorRT version: {trt_version}. "
                        f"Check if the engine was created with a compatible TensorRT version."
                    )
        except Exception as e:
            trt_version = trt.__version__
            raise RuntimeError(
                f"Failed to load TensorRT engine from {self.engine_file_path}. "
                f"Error: {str(e)}. Current TensorRT version: {trt_version}. "
                f"The engine might be incompatible with this version of TensorRT."
            ) from e

    def _set_input_shapes(self):
        """Set input shapes for TensorRT >= 8.x."""
        for binding_idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(binding_idx)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                shape = self.engine.get_tensor_shape(name)
                self.context.set_input_shape(name, shape)

    def setup_io_bindings(self):
        """Setup input/output bindings for TensorRT inference."""
        # Get shapes for inputs and outputs
        self.input_shapes = {}
        self.output_shapes = {}
        self.bindings = []
        self.inputs = {}
        self.outputs = {}

        # Process all bindings
        binding_names = []
        binding_is_input = []

        # Get all binding names
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            binding_names.append(name)
            binding_is_input.append(self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT)

        # Setup memory for inputs
        for i, binding_name in enumerate(binding_names):
            if binding_is_input[i] and binding_name in self.input_binding_names:
                shape = tuple(self.engine.get_tensor_shape(binding_name))
                self.input_shapes[binding_name] = shape

                # Allocate device memory
                dtype = trt.nptype(self.engine.get_tensor_dtype(binding_name))
                input_size = trt.volume(shape) * np.dtype(dtype).itemsize

                err, d_input = cuda_runtime.cudaMalloc(input_size)
                if err != cuda_runtime.cudaError_t.cudaSuccess:
                    err_msg = f"Failed to allocate input memory for {binding_name}: " \
                              f"error code {err}"
                    raise RuntimeError(err_msg)

                self.bindings.append(int(d_input))
                self.inputs[binding_name] = (d_input, input_size, shape, dtype)

        # Setup memory for outputs
        for i, binding_name in enumerate(binding_names):
            if not binding_is_input[i] and binding_name in self.output_binding_names:
                shape = tuple(self.engine.get_tensor_shape(binding_name))
                self.output_shapes[binding_name] = shape

                # Allocate device memory
                dtype = trt.nptype(self.engine.get_tensor_dtype(binding_name))
                output_size = trt.volume(shape) * np.dtype(dtype).itemsize

                err, d_output = cuda_runtime.cudaMalloc(output_size)
                if err != cuda_runtime.cudaError_t.cudaSuccess:
                    err_msg = f"Failed to allocate output memory for {binding_name}: " \
                              f"error code {err}"
                    raise RuntimeError(err_msg)

                self.bindings.append(int(d_output))
                self.outputs[binding_name] = (d_output, output_size, shape, dtype)

        # Validate that we found all expected bindings
        if len(self.inputs) != len(self.input_binding_names):
            missing = set(self.input_binding_names) - set(self.inputs.keys())
            raise ValueError(f"Missing input bindings: {missing}")

        if len(self.outputs) != len(self.output_binding_names):
            missing = set(self.output_binding_names) - set(self.outputs.keys())
            raise ValueError(f"Missing output bindings: {missing}")

        # Create stream for async execution
        err, self.stream_handle = cuda_runtime.cudaStreamCreate()
        if err != cuda_runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Failed to create CUDA stream: error code {err}")

    def infer(self, left_image, right_image):
        """Run inference on left and right stereo images.

        Args:
            left_image: Left stereo image (BGR format)
            right_image: Right stereo image (BGR format)

        Returns:
            Model-specific inference results
        """
        # Preprocess images
        left_processed, preprocess_metadata = self.preprocess_image(left_image)
        right_processed, _ = self.preprocess_image(right_image)

        # Ensure arrays are contiguous in memory
        if not left_processed.flags.c_contiguous:
            left_processed = np.ascontiguousarray(left_processed)
        if not right_processed.flags.c_contiguous:
            right_processed = np.ascontiguousarray(right_processed)

        # Copy input data to device
        self._copy_inputs_to_device(left_processed, right_processed)

        # Set tensor addresses for TensorRT
        for name, (mem, _, _, _) in self.inputs.items():
            self.context.set_tensor_address(name, int(mem))
        for name, (mem, _, _, _) in self.outputs.items():
            self.context.set_tensor_address(name, int(mem))

        # Execute inference
        self.context.execute_async_v3(stream_handle=self.stream_handle)

        # Get results from device
        results = self._copy_outputs_from_device()

        # Synchronize to wait for results
        err, = cuda_runtime.cudaStreamSynchronize(self.stream_handle)
        if err != cuda_runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Failed to synchronize CUDA stream: error code {err}")

        return self.process_inference_results(results, preprocess_metadata)

    def _copy_inputs_to_device(self, left_processed, right_processed):
        """Copy preprocessed input data to GPU memory."""
        input_names = list(self.input_binding_names)

        # Copy left image
        err, = cuda_runtime.cudaMemcpyAsync(
            self.inputs[input_names[0]][0],
            left_processed.ctypes.data,
            self.inputs[input_names[0]][1],
            cuda_runtime.cudaMemcpyKind.cudaMemcpyHostToDevice,
            self.stream_handle
        )
        if err != cuda_runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Failed to copy left input data to device: error code {err}")

        # Copy right image
        err, = cuda_runtime.cudaMemcpyAsync(
            self.inputs[input_names[1]][0],
            right_processed.ctypes.data,
            self.inputs[input_names[1]][1],
            cuda_runtime.cudaMemcpyKind.cudaMemcpyHostToDevice,
            self.stream_handle
        )
        if err != cuda_runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Failed to copy right input data to device: error code {err}")

    def _copy_outputs_from_device(self):
        """Copy inference results from GPU memory to host."""
        results = {}

        for output_name in self.output_binding_names:
            shape = self.output_shapes[output_name]
            dtype = self.outputs[output_name][3]  # dtype is stored at index 3

            h_output = np.empty(shape, dtype=dtype)

            err, = cuda_runtime.cudaMemcpyAsync(
                h_output.ctypes.data,
                self.outputs[output_name][0],
                self.outputs[output_name][1],
                cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self.stream_handle
            )
            if err != cuda_runtime.cudaError_t.cudaSuccess:
                raise RuntimeError(f"Failed to copy {output_name} output data to host: "
                                   f"error code {err}")

            results[output_name] = h_output

        return results

    @abstractmethod
    def load_plugins(self):
        """Load model-specific TensorRT plugins."""
        pass

    @abstractmethod
    def get_input_binding_names(self):
        """Get list of input binding names for the model."""
        pass

    @abstractmethod
    def get_output_binding_names(self):
        """Get list of output binding names for the model."""
        pass

    @abstractmethod
    def preprocess_image(self, image):
        """Preprocess an image for model inference.

        Args:
            image: BGR image (HWC format)

        Returns:
            Tuple of (preprocessed image, preprocessing metadata)
        """
        pass

    @abstractmethod
    def process_inference_results(self, results, preprocess_metadata):
        """Process raw inference results into final output format.

        Args:
            results: Raw inference results from GPU
            preprocess_metadata: Metadata from preprocessing

        Returns:
            Processed inference results
        """
        pass


class BaseStereoRunner(ABC):
    """Base class for stereo inference runners."""

    def __init__(self, image_dir, output_dir, engine_file_path, frames_meta_file=None,
                 verbose=False):
        """Initialize stereo runner.

        Args:
            image_dir: Directory containing input images
            output_dir: Directory for output depth images
            engine_file_path: Path to TensorRT engine file
            frames_meta_file: Optional path to frames metadata file
            verbose: Enable verbose logging
        """
        self.image_dir = image_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.engine_file_path = engine_file_path
        self.frames_meta_file = frames_meta_file
        self.verbose = verbose
        self.image_count = 0
        self.camera_meta = None

        # Check if engine file exists
        self._validate_engine_file()

        # Initialize inference engine
        self.stereo_inference = self.create_inference_engine()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if hasattr(self, 'stereo_inference'):
            self.stereo_inference.cleanup()

    def _validate_engine_file(self):
        """Validate that the engine file exists."""
        if not os.path.exists(self.engine_file_path):
            engine_dir = os.path.dirname(self.engine_file_path)
            if os.path.exists(engine_dir):
                print("Available files in engine directory:")
                for file in os.listdir(engine_dir):
                    print(f"  - {file}")
            else:
                print(f"Engine directory does not exist: {engine_dir}")
            raise ValueError(f"Engine file not found: {self.engine_file_path}")

    def image_name_to_timestamp(self, image_name):
        """Convert image name to timestamp."""
        return int(image_name.split('.')[0].split('/')[-1])

    def ensure_directory_exists_for_file(self, file_path):
        """Ensure directory exists for the given file path."""
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)

    def get_camera_transform(self, camera_param_id):
        """Get camera transform from metadata."""
        camera_params = self.get_camera_params(camera_param_id)
        translation = \
            camera_params['sensor_meta_data']['sensor_to_vehicle_transform']['translation']
        translation_vector = np.array([translation['x'], translation['y'], translation['z']])
        rotation = \
            camera_params['sensor_meta_data']['sensor_to_vehicle_transform'].get('axis_angle')
        if rotation:
            axis = [rotation['x'], rotation['y'], rotation['z']]
            angle = np.deg2rad(rotation['angle_degrees'])
            rotation_matrix = R.from_rotvec(np.array(axis) * angle).as_matrix()
        else:
            rotation_matrix = np.eye(3)
        return translation_vector, rotation_matrix

    def compute_baseline(self, camera_param_id_1, camera_param_id_2):
        """Compute baseline between two cameras."""
        translation_1, rotation_1 = self.get_camera_transform(camera_param_id_1)
        translation_2, rotation_2 = self.get_camera_transform(camera_param_id_2)

        T_camera_1_vehicle = np.eye(4)
        T_camera_1_vehicle[:3, :3] = rotation_1
        T_camera_1_vehicle[:3, 3] = translation_1

        T_camera_2_vehicle = np.eye(4)
        T_camera_2_vehicle[:3, :3] = rotation_2
        T_camera_2_vehicle[:3, 3] = translation_2

        T_left_to_right_transform = np.linalg.inv(T_camera_2_vehicle) @ T_camera_1_vehicle
        baseline = -T_left_to_right_transform[0, 3]

        if baseline == 0:
            raise ValueError(
                "The computed baseline is zero, which is invalid for depth calculation.")
        print(
            "Baseline between camera, ",
            camera_param_id_1,
            camera_param_id_2,
            " is:",
            baseline,
        )
        return baseline

    def get_camera_params(self, camera_params_id: str):
        """Get camera parameters by ID."""
        if camera_params_id not in self.camera_meta['camera_params_id_to_camera_params']:
            raise ValueError(f"No camera data found for camera id {camera_params_id}")
        return self.camera_meta['camera_params_id_to_camera_params'][camera_params_id]

    def get_focal_length(self, camera_params_id: str):
        """Get focal length from camera parameters.

        Args:
            camera_params_id: Camera parameter ID

        Returns:
            Focal length (fx)
        """
        camera_params = self.get_camera_params(camera_params_id)
        calibration = camera_params["calibration_parameters"]
        # Extract fx from the projection matrix (first element)
        return float(calibration['projection_matrix']['data'][0])

    def get_baseline(self, left_camera_id: str, right_camera_id: str):
        """Get baseline between left and right cameras.

        Args:
            left_camera_id: Left camera parameter ID
            right_camera_id: Right camera parameter ID

        Returns:
            Baseline in meters
        """
        # Check if baseline is defined in stereo pairs
        for pair in self.camera_meta["stereo_pair"]:
            left_id = pair.get('left_camera_param_id', '0')
            right_id = pair.get('right_camera_param_id', '0')
            if left_id == left_camera_id and right_id == right_camera_id:
                baseline = pair.get('baseline_meters', 0)
                if baseline != 0:
                    return baseline
                # If baseline is not defined, compute it
                return self.compute_baseline(left_camera_id, right_camera_id)

        # If not found in stereo pairs, compute it
        return self.compute_baseline(left_camera_id, right_camera_id)

    def convert_disparity_to_depth(self, disparity, focal_length, baseline,
                                   original_width=None, network_width=None):
        """Convert disparity to depth using stereo camera parameters.

        Args:
            disparity: Disparity array
            focal_length: Camera focal length
            baseline: Stereo baseline in meters
            original_width: Original image width (for focal length scaling)
            network_width: Network input width (for focal length scaling)

        Returns:
            Depth array in millimeters (uint16 format)
        """
        depth = np.zeros_like(disparity)

        # Scale focal length if needed
        if original_width and network_width:
            scaled_focal_length = focal_length * (network_width / original_width)
        else:
            scaled_focal_length = focal_length

        # Define minimum disparity threshold to avoid division by zero or extreme values
        MIN_DISPARITY_THRESHOLD = 0.01  # 1cm minimum disparity
        MAX_DEPTH_VALUE = 65535.0       # Maximum for uint16 output

        # Create mask for valid disparity values
        valid_disparity = disparity > MIN_DISPARITY_THRESHOLD

        if np.any(valid_disparity):
            # Calculate depth only for valid disparity values
            disp_valid = disparity[valid_disparity]

            # Calculate depth = baseline * focal_length / disparity
            # Multiply by 1000 to convert from meters to millimeters for uint16 output
            depth_values = (1000 * scaled_focal_length * baseline) / disp_valid

            # Clip extreme depth values
            depth_values[depth_values > MAX_DEPTH_VALUE] = 0

            # Assign depth values to output array
            depth[valid_disparity] = depth_values

        return depth

    def process_camera_pair(self, camera_keyframes, left_camera_id: str, right_camera_id: str):
        """Process a stereo camera pair."""
        # Get focal length and baseline directly
        focal_length = self.get_focal_length(left_camera_id)
        baseline = self.get_baseline(left_camera_id, right_camera_id)

        if self.verbose:
            print(f"Using focal length: {focal_length}")
            print(f"Baseline: {baseline}")

        if left_camera_id not in camera_keyframes:
            print(f"Camera {left_camera_id} not found in keyframes")
            return

        if right_camera_id not in camera_keyframes:
            print(f"Camera {right_camera_id} not found in keyframes")
            return

        left_keyframes = camera_keyframes[left_camera_id]
        if not left_keyframes:
            print(f"No keyframes found for camera {left_camera_id}, skipping this camera")
            return

        right_keyframes = camera_keyframes[right_camera_id]
        if not right_keyframes:
            print(f"No keyframes found for camera {right_camera_id}, skipping this camera")
            return

        for synced_sample_id, left_keyframe in left_keyframes.items():
            if synced_sample_id not in right_keyframes:
                print(
                    "Right image not found corresponding left image: ",
                    left_keyframe['image_name'],
                    f"with synced_sample_id: {synced_sample_id}, skipping this frame",
                )
                continue

            left_image_name = left_keyframe['image_name']
            right_image_name = right_keyframes[synced_sample_id]['image_name']

            # Load and validate images
            left_image, right_image = self._load_and_validate_images(
                left_image_name, right_image_name)
            if left_image is None or right_image is None:
                continue

            # Process stereo pair
            self._process_stereo_pair(
                left_image, right_image, left_image_name,
                focal_length, baseline)

    def _load_and_validate_images(self, left_image_name, right_image_name):
        """Load and validate stereo image pair."""
        left_image_path = os.path.join(self.image_dir, left_image_name)
        if not os.path.exists(left_image_path):
            print(f"Left image not found for {left_image_path}, skipping this image")
            return None, None

        right_image_path = os.path.join(self.image_dir, right_image_name)
        if not os.path.exists(right_image_path):
            print(f"Right image not found for {right_image_path}, skipping this image")
            return None, None

        left_image = cv2.imread(left_image_path, cv2.IMREAD_COLOR)
        if left_image is None:
            print(f"Error reading image {left_image_name}, skipping this image")
            return None, None

        right_image = cv2.imread(right_image_path, cv2.IMREAD_COLOR)
        if right_image is None:
            print(f"Error reading image {right_image_name}, skipping this image")
            return None, None

        return left_image, right_image

    def extract_data(self):
        """Extract and process stereo data from metadata."""
        # Load metadata
        if self.frames_meta_file:
            metadata_file = self.frames_meta_file
        else:
            metadata_file = os.path.join(self.image_dir, 'frames_meta.json')
        with open(metadata_file, 'r') as f:
            self.camera_meta = json.load(f)

        if len(self.camera_meta['stereo_pair']) == 0:
            print('Cannot find stereo pairs in frame metadata!')
            exit(1)

        # Parse keyframes based on camera param id + synced_sample_id
        camera_keyframes = {}
        print("Processing number of keyframes:", len(self.camera_meta['keyframes_metadata']))
        for keyframe in self.camera_meta['keyframes_metadata']:
            camera_params_id = keyframe.get('camera_params_id', '0')
            synced_sample_id = keyframe.get('synced_sample_id')

            # Skip keyframes without synced_sample_id
            if synced_sample_id is None or synced_sample_id == '':
                if self.verbose:
                    print(f"Skipping keyframe {keyframe.get('image_name', 'unknown')} "
                          f"without synced_sample_id")
                continue

            if camera_params_id not in camera_keyframes:
                camera_keyframes[camera_params_id] = {}
            camera_keyframes[camera_params_id][synced_sample_id] = keyframe

        if not camera_keyframes:
            raise ValueError("No keyframes found in the metadata file")

        if ('stereo_pair' not in self.camera_meta or
                not self.camera_meta['stereo_pair'] or
                len(self.camera_meta['stereo_pair']) == 0):
            raise ValueError("No stereo pairs found in the metadata file")

        # Process each stereo pair
        for pair in self.camera_meta['stereo_pair']:
            left_camera_id = pair.get('left_camera_param_id', '0')
            right_camera_id = pair.get('right_camera_param_id', '0')
            print("Processing camera pair: ", left_camera_id, right_camera_id)
            self.process_camera_pair(camera_keyframes, left_camera_id, right_camera_id)

        print("Processing completed")

    @abstractmethod
    def create_inference_engine(self):
        """Create the model-specific inference engine."""
        pass

    @abstractmethod
    def _process_stereo_pair(self, left_image, right_image, left_image_name,
                             focal_length, baseline):
        """Process a single stereo pair and save results."""
        pass
