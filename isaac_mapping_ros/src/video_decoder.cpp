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

#include "isaac_mapping_ros/video_decoder.hpp"
namespace isaac_ros
{
namespace isaac_mapping_ros
{

bool VideoDecoder::Init(const std::string & codec_name)
{
  av_log_set_level(AV_LOG_ERROR);

  // Find the decoder
  codec_ = avcodec_find_decoder_by_name(codec_name.c_str());
  if (!codec_) {
    LOG(ERROR) << "Codec not found";
    return false;
  }

  // Allocate codec context
  codec_context_ = avcodec_alloc_context3(codec_);
  if (!codec_context_) {
    LOG(ERROR) << "Could not allocate video codec context";
    return false;
  }

  // Open codec
  if (avcodec_open2(codec_context_, codec_, nullptr) < 0) {
    LOG(ERROR) << "Could not open codec";
    return false;
  }

  // Allocate frame
  frame_ = av_frame_alloc();
  if (!frame_) {
    LOG(ERROR) << "Could not allocate video frame";
    return false;
  }

  // Allocate packet
  packet_ = av_packet_alloc();
  if (!packet_) {
    LOG(ERROR) << "Could not allocate packet";
    return false;
  }

  return true;
}

VideoDecoder::~VideoDecoder()
{
  avcodec_free_context(&codec_context_);
  av_frame_free(&frame_);
  av_packet_free(&packet_);
}

bool VideoDecoder::DecodePacket(
  const std::vector<uint8_t> & input_data,
  std::vector<cv::Mat> & frames)
{
  frames.clear();
  // Fill the packet with input data
  av_packet_unref(packet_);
  packet_->data = const_cast<uint8_t *>(input_data.data());
  packet_->size = input_data.size();

  // Send the packet to the decoder
  if (avcodec_send_packet(codec_context_, packet_) < 0) {
    LOG(WARNING) << "Error sending packet for decoding";
    return false;
  }

  // Receive the frame from the decoder
  while (avcodec_receive_frame(codec_context_, frame_) == 0) {
    // Convert AVFrame to cv::Mat
    cv::Mat mat_frame;
    if (!AVFrameToCVMat(frame_, mat_frame)) {
      return false;
    }
    frames.push_back(mat_frame);
  }

  return true;
}


bool VideoDecoder::AVFrameToCVMat(AVFrame * frame, cv::Mat & mat)
{
  // Convert the frame to RGB format using libswscale
  SwsContext * sws_context = sws_getContext(
    frame->width, frame->height, static_cast<AVPixelFormat>(frame->format),
    frame->width, frame->height, AV_PIX_FMT_BGR24,
    SWS_BILINEAR, nullptr, nullptr, nullptr);

  if (!sws_context) {
    LOG(ERROR) << "Could not initialize the conversion context";
    return false;
  }

  // Allocate an OpenCV Mat with the same dimensions as the frame
  mat = cv::Mat(frame->height, frame->width, CV_8UC3);

  // Create an array of pointers to the data planes of the destination image
  uint8_t * dest[4] = {mat.data, nullptr, nullptr, nullptr};
  int dest_linesize[4] = {static_cast<int>(mat.step), 0, 0, 0};

  // Perform the conversion
  sws_scale(sws_context, frame->data, frame->linesize, 0, frame->height, dest, dest_linesize);

  // Free the conversion context
  sws_freeContext(sws_context);

  return true;
}

}
}
