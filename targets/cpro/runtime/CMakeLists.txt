cmake_minimum_required(VERSION 3.16)
project(qwen35_dflash_hiai VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(BUILD_HIAI_RUNNER "Link the HarmonyOS HIAI executable" OFF)
option(BUILD_HIAI_OBJECTS "Compile HIAI adapter against real DDK headers without linking" OFF)
option(BUILD_HOST_TESTS "Build host scheduler and HIAI lifecycle tests" ON)
set(HIAI_INCLUDE_DIR "" CACHE PATH "Directory containing HiAiModelManagerService.h")
set(HIAI_LIBRARIES "" CACHE STRING "Absolute paths to internal HarmonyOS ARM64 HIAI libraries")
add_library(dflash_core STATIC src/generation.cpp src/sha256.cpp)
target_include_directories(dflash_core PUBLIC include)
target_compile_options(dflash_core PRIVATE -Wall -Wextra -Wpedantic)
if(BUILD_HIAI_RUNNER OR BUILD_HIAI_OBJECTS)
  if(NOT EXISTS "${HIAI_INCLUDE_DIR}/HiAiModelManagerService.h")
    message(FATAL_ERROR "Set HIAI_INCLUDE_DIR to the matching DDK headers")
  endif()
  add_library(hiai_adapter OBJECT src/hiai_executor.cpp src/main.cpp)
  target_include_directories(hiai_adapter PRIVATE include "${HIAI_INCLUDE_DIR}")
  target_compile_options(hiai_adapter PRIVATE -Wall -Wextra -Wpedantic)
endif()
if(BUILD_HIAI_RUNNER)
  if(NOT HIAI_LIBRARIES)
    message(FATAL_ERROR "HIAI_LIBRARIES must name actual internal HarmonyOS ARM64 libraries")
  endif()
  foreach(lib IN LISTS HIAI_LIBRARIES)
    if(NOT IS_ABSOLUTE "${lib}" OR NOT EXISTS "${lib}")
      message(FATAL_ERROR "HIAI library missing or not absolute: ${lib}")
    endif()
  endforeach()
  add_executable(qwen35_dflash_hiai $<TARGET_OBJECTS:hiai_adapter>)
  target_link_libraries(qwen35_dflash_hiai PRIVATE dflash_core ${HIAI_LIBRARIES})
endif()
if(BUILD_HOST_TESTS)
  enable_testing()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_executable(dflash_scheduler_tests tests/test_generation.cpp)
  target_link_libraries(dflash_scheduler_tests PRIVATE dflash_core)
  add_test(NAME scheduler COMMAND dflash_scheduler_tests)
  add_executable(hiai_runner_fake src/main.cpp src/hiai_executor.cpp tests/fake_hiai/fake_hiai.cpp)
  target_include_directories(hiai_runner_fake PRIVATE tests/fake_hiai)
  target_compile_definitions(hiai_runner_fake PRIVATE QWEN35_FAKE_HIAI=1)
  target_link_libraries(hiai_runner_fake PRIVATE dflash_core)
  add_executable(hiai_model_buffer_tests tests/test_hiai_model_buffer.cpp
    src/hiai_executor.cpp tests/fake_hiai/fake_hiai.cpp)
  target_include_directories(hiai_model_buffer_tests PRIVATE tests/fake_hiai)
  target_link_libraries(hiai_model_buffer_tests PRIVATE dflash_core)
  add_test(NAME hiai_model_buffer
    COMMAND hiai_model_buffer_tests ${CMAKE_CURRENT_BINARY_DIR}/buffer-cases)
  add_executable(hiai_feature_capture_tests tests/test_hiai_feature_capture.cpp
    src/hiai_executor.cpp tests/fake_hiai/fake_hiai.cpp)
  target_include_directories(hiai_feature_capture_tests PRIVATE tests/fake_hiai)
  target_link_libraries(hiai_feature_capture_tests PRIVATE dflash_core)
  add_test(NAME hiai_feature_capture
    COMMAND hiai_feature_capture_tests ${CMAKE_CURRENT_BINARY_DIR}/feature-cases)
  add_test(NAME hiai_lifecycle_faults
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_hiai_runner.py
      $<TARGET_FILE:hiai_runner_fake> ${CMAKE_CURRENT_BINARY_DIR}/fake-cases)
  add_test(NAME hiai_io_padding
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_hiai_io_padding.py
      $<TARGET_FILE:hiai_runner_fake> ${CMAKE_CURRENT_BINARY_DIR}/padding-cases)
  add_test(NAME hiai_draft_step
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_hiai_draft_step.py
      $<TARGET_FILE:hiai_runner_fake> ${CMAKE_CURRENT_BINARY_DIR}/draft-cases)
  add_test(NAME hiai_draft_file_v2
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_hiai_draft_file_v2.py
      $<TARGET_FILE:hiai_runner_fake> ${CMAKE_CURRENT_BINARY_DIR}/draft-v2-cases)
  add_test(NAME hiai_model_description_probe
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_hiai_model_description_probe.py
      $<TARGET_FILE:hiai_runner_fake> ${CMAKE_CURRENT_BINARY_DIR}/description-cases)
  add_test(NAME hiai_model_io_probe
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_hiai_model_io_probe.py
      $<TARGET_FILE:hiai_runner_fake> ${CMAKE_CURRENT_BINARY_DIR}/model-io-cases)
  add_executable(hiai_runner_fake_no_tensor_name src/main.cpp src/hiai_executor.cpp tests/fake_hiai/fake_hiai.cpp)
  target_include_directories(hiai_runner_fake_no_tensor_name PRIVATE tests/fake_hiai)
  target_compile_definitions(hiai_runner_fake_no_tensor_name PRIVATE QWEN35_FAKE_HIAI=1 QWEN35_FAKE_NO_TENSOR_NAME=1)
  target_link_libraries(hiai_runner_fake_no_tensor_name PRIVATE dflash_core)
  add_test(NAME hiai_model_io_probe_no_tensor_name
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_hiai_model_io_probe.py
      $<TARGET_FILE:hiai_runner_fake_no_tensor_name> ${CMAKE_CURRENT_BINARY_DIR}/model-io-no-name-cases no-name-field)
  add_executable(hiai_optional_enum_tests tests/test_hiai_optional_enum.cpp)
  target_link_libraries(hiai_optional_enum_tests PRIVATE dflash_core)
  add_test(NAME hiai_optional_enum COMMAND hiai_optional_enum_tests)
endif()
