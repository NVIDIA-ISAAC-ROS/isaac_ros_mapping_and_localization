// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

namespace nvidia
{
namespace isaac_ros
{
namespace visual_global_localization
{

static const double kMillisecondsToSeconds = 1.0 / 1000.0;
static const double kSecondsToMilliseconds = 1000.0;
static const double kMillisecondsToNanoseconds = 1e6;
static const double kMicrosecondsToNanoseconds = 1e3;
static const double kNanosecondsToMicroseconds = 1.0 / 1000.0;
static const uint64_t kSecondsToNanoseconds = 1e9;
static const uint64_t kSecondsToMicroseconds = 1e6;
static const double kRotationWeight = 5.0;
// The threshold for image sync in milliseconds
static const double kImageSyncMatchThresholdMs = 3.0;
static const int32_t kCameraFrequency = 30;

static const std::string kAprilTagLocalizationTopic = "pose";
static const std::string kObservationsFileName = "observations.pb";
// the QoS settings for the image input topics
static const std::string kImageQosProfile = "SYSTEM_DEFAULT";
static const std::string KOdomFrame = "odom";

} // namespace visual_global_localization
} // namespace isaac_ros
} // namespace nvidia
