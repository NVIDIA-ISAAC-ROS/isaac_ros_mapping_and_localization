// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

#include <glog/logging.h>


namespace isaac_ros
{
namespace isaac_mapping_ros
{

// A simple video decoder that wraps ffmpeg c API to decode packets into cv::Mat
class VideoDecoder
{
public:
  VideoDecoder() {}

  bool Init(const std::string & codec_name = "h264");

  ~VideoDecoder();

  bool DecodePacket(
    const std::vector<uint8_t> & input_data,
    std::vector<cv::Mat> & frames);

private:
  const AVCodec * codec_ = nullptr;
  AVCodecContext * codec_context_ = nullptr;
  AVFrame * frame_ = nullptr;
  AVPacket * packet_ = nullptr;

  bool AVFrameToCVMat(AVFrame * frame, cv::Mat & mat);
};

}
}
