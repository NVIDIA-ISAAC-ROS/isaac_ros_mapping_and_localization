/*
Copyright 2024 NVIDIA CORPORATION

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include <gtest/gtest.h>

#include "isaac_mapping_ros/nvblox_utils/mapping_data_loader.hpp"

using namespace nvblox;

class MappingDataLoaderTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    color_dir_ = "data/galileo/raw";
    depth_dir_ = "data/galileo/depth";
    frames_meta_file_ = "data/galileo/frames_meta.json";
  }

  std::string color_dir_;
  std::string depth_dir_;
  std::string frames_meta_file_;
};

TEST_F(MappingDataLoaderTest, TestLoadImage) {
  auto loader = datasets::mapping_data::DataLoader::create(
    color_dir_, depth_dir_,
    frames_meta_file_, false);

  DepthImage depth_image(nvblox::MemoryType::kDevice);
  ColorImage color_image(nvblox::MemoryType::kDevice);
  Transform T_L_C;
  Camera camera;

  const size_t kNumTotalImages = 8;
  for (size_t i = 0; i < kNumTotalImages; ++i) {
    EXPECT_EQ(
      loader->loadNext(
        &depth_image, &T_L_C, &camera,
        &color_image), datasets::DataLoadResult::kSuccess);
  }

  EXPECT_EQ(
    loader->loadNext(
      &depth_image, &T_L_C, &camera,
      &color_image), datasets::DataLoadResult::kNoMoreData);
}

int main(int argc, char ** argv)
{
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
