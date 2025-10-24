find_package(ament_cmake_gtest REQUIRED)
find_package(ament_lint_auto REQUIRED)
find_package(isaac_ros_test_cmake)

# googletest is already included by visual_mapping subdirectory

ament_lint_auto_find_test_dependencies()

# Copy test data into the testing space (build/tests)
file(INSTALL "${CMAKE_CURRENT_SOURCE_DIR}/data" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

add_executable(mapping_data_loader_test mapping_data_loader_test.cpp)
target_link_libraries(mapping_data_loader_test nvblox_utils gtest gtest_main)
add_test(NAME mapping_data_loader_test COMMAND mapping_data_loader_test)

get_isaac_ros_asset_install_script_path(install_ess_models_script install_ess_models isaac_ros_ess_models_install)
ament_add_test(setup_ess_models
    COMMAND "${install_ess_models_script}"
    ENV "ISAAC_ROS_SKIP_ASSET_INSTALL="
    TIMEOUT 3600
    GENERATE_RESULT_FOR_RETURN_CODE_ZERO
)
# test_create_map depends on ESS models for aarch64
add_test(test_create_map bash ${CMAKE_CURRENT_SOURCE_DIR}/test_create_map_offline.sh ess)
set_tests_properties(test_create_map PROPERTIES DEPENDS setup_ess_models)
