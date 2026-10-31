cmake_minimum_required(VERSION 3.16)
project(qwen35_dflash_hiai VERSION 0.1.0 LANGUAGES C CXX)
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
add_executable(qwen35_board_serial src/board_serial_main.cpp)
target_link_libraries(qwen35_board_serial PRIVATE dflash_core)
target_compile_options(qwen35_board_serial PRIVATE -Wall -Wextra -Wpedantic)
if(BUILD_HIAI_RUNNER OR BUILD_HIAI_OBJECTS)
  if(NOT EXISTS "${HIAI_INCLUDE_DIR}/HiAiModelManagerService.h")
    message(FATAL_ERROR "Set HIAI_INCLUDE_DIR to the matching DDK headers")
  endif()
  add_library(hiai_adapter OBJECT src/hiai_executor.cpp src/main.cpp)
  target_include_directories(hiai_adapter PRIVATE include "${HIAI_INCLUDE_DIR}")
  target_compile_options(hiai_adapter PRIVATE -Wall -Wextra -Wpedantic)
  add_library(hiai_resident_adapter OBJECT src/board_serial_main.cpp src/resident_session.cpp src/hiai_executor.cpp)
  target_include_directories(hiai_resident_adapter PRIVATE include "${HIAI_INCLUDE_DIR}")
  target_compile_definitions(hiai_resident_adapter PRIVATE QWEN35_RESIDENT_V2=1)
  target_compile_options(hiai_resident_adapter PRIVATE -Wall -Wextra -Wpedantic)
  add_library(hiai_resident_mixed_adapter OBJECT src/board_serial_main.cpp src/resident_session.cpp src/hiai_executor.cpp)
  target_include_directories(hiai_resident_mixed_adapter PRIVATE include "${HIAI_INCLUDE_DIR}")
  target_compile_definitions(hiai_resident_mixed_adapter PRIVATE QWEN35_RESIDENT_MIXED=1)
  target_compile_options(hiai_resident_mixed_adapter PRIVATE -Wall -Wextra -Wpedantic)
  add_library(hiai_resident_v1_adapter OBJECT src/board_serial_main.cpp src/resident_session.cpp src/hiai_executor.cpp)
  target_include_directories(hiai_resident_v1_adapter PRIVATE include "${HIAI_INCLUDE_DIR}")
  target_compile_definitions(hiai_resident_v1_adapter PRIVATE QWEN35_RESIDENT_V1=1)
  target_compile_options(hiai_resident_v1_adapter PRIVATE -Wall -Wextra -Wpedantic)
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
  add_executable(qwen35_board_resident $<TARGET_OBJECTS:hiai_resident_adapter>)
  target_link_libraries(qwen35_board_resident PRIVATE dflash_core ${HIAI_LIBRARIES})
  add_executable(qwen35_board_resident_mixed $<TARGET_OBJECTS:hiai_resident_mixed_adapter>)
  target_link_libraries(qwen35_board_resident_mixed PRIVATE dflash_core ${HIAI_LIBRARIES})
  add_executable(qwen35_board_resident_v1 $<TARGET_OBJECTS:hiai_resident_v1_adapter>)
  target_link_libraries(qwen35_board_resident_v1 PRIVATE dflash_core ${HIAI_LIBRARIES})
endif()
if(BUILD_HOST_TESTS)
  enable_testing()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_executable(board_serial_fake src/board_serial_main.cpp)
  target_compile_definitions(board_serial_fake PRIVATE QWEN35_BOARD_SERIAL_HOST_FAKE=1)
  target_link_libraries(board_serial_fake PRIVATE dflash_core)
  add_test(NAME board_local_serial
    COMMAND ${CMAKE_COMMAND} -E env
      QWEN35_BOARD_SERIAL_TEST_FAKE=$<TARGET_FILE:board_serial_fake>
      QWEN35_BOARD_SERIAL_TEST_PRODUCTION=$<TARGET_FILE:qwen35_board_serial>
      ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_board_serial.py
      -q -p no:cacheprovider --basetemp=${CMAKE_CURRENT_BINARY_DIR}/board-serial-cases)
  add_test(NAME board_local_resident_protocol
    COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_board_resident.py
      -q -p no:cacheprovider --basetemp=${CMAKE_CURRENT_BINARY_DIR}/board-resident-cases)
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
  add_executable(hiai_original_target_tests tests/test_hiai_original_target.cpp
    src/hiai_executor.cpp tests/fake_hiai/fake_hiai.cpp)
  target_include_directories(hiai_original_target_tests PRIVATE tests/fake_hiai)
  target_link_libraries(hiai_original_target_tests PRIVATE dflash_core)
  add_test(NAME hiai_original_target_resident_adapter
    COMMAND hiai_original_target_tests ${CMAKE_CURRENT_BINARY_DIR}/original-target-cases)
  add_executable(hiai_resident_session_tests tests/test_resident_session.cpp
    src/resident_session.cpp src/hiai_executor.cpp tests/fake_hiai/fake_hiai.cpp)
  target_include_directories(hiai_resident_session_tests PRIVATE tests/fake_hiai)
  target_link_libraries(hiai_resident_session_tests PRIVATE dflash_core)
  add_test(NAME hiai_resident_session
    COMMAND hiai_resident_session_tests ${CMAKE_CURRENT_BINARY_DIR}/resident-session-cases)
  add_executable(board_resident_hiai_fake src/board_serial_main.cpp src/resident_session.cpp
    src/hiai_executor.cpp tests/fake_hiai/fake_hiai.cpp)
  target_include_directories(board_resident_hiai_fake PRIVATE tests/fake_hiai)
  target_compile_definitions(board_resident_hiai_fake PRIVATE QWEN35_RESIDENT_V2=1 QWEN35_BOARD_SERIAL_HOST_FAKE=1 QWEN35_FAKE_HIAI=1)
  target_link_libraries(board_resident_hiai_fake PRIVATE dflash_core)
  add_executable(board_resident_mixed_hiai_fake src/board_serial_main.cpp src/resident_session.cpp
    src/hiai_executor.cpp tests/fake_hiai/fake_hiai.cpp)
  target_include_directories(board_resident_mixed_hiai_fake PRIVATE tests/fake_hiai)
  target_compile_definitions(board_resident_mixed_hiai_fake PRIVATE QWEN35_RESIDENT_MIXED=1 QWEN35_BOARD_SERIAL_HOST_FAKE=1 QWEN35_FAKE_HIAI=1)
  target_link_libraries(board_resident_mixed_hiai_fake PRIVATE dflash_core)
  add_executable(board_resident_v1_hiai_fake src/board_serial_main.cpp src/resident_session.cpp
    src/hiai_executor.cpp tests/fake_hiai/fake_hiai.cpp)
  target_include_directories(board_resident_v1_hiai_fake PRIVATE tests/fake_hiai)
  target_compile_definitions(board_resident_v1_hiai_fake PRIVATE QWEN35_RESIDENT_V1=1 QWEN35_BOARD_SERIAL_HOST_FAKE=1 QWEN35_FAKE_HIAI=1)
  target_link_libraries(board_resident_v1_hiai_fake PRIVATE dflash_core)
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
  add_test(NAME hiai_target_load_probe
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_hiai_target_load_probe.py
      $<TARGET_FILE:hiai_runner_fake> ${CMAKE_CURRENT_BINARY_DIR}/target-load-probe-cases)
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

if(BUILD_HOST_TESTS)
  add_executable(hiai_resident_v1_tests tests/test_resident_v1.cpp src/resident_session.cpp
    src/hiai_executor.cpp tests/fake_hiai/fake_hiai.cpp)
  target_include_directories(hiai_resident_v1_tests PRIVATE tests/fake_hiai)
  target_link_libraries(hiai_resident_v1_tests PRIVATE dflash_core)
  add_test(NAME hiai_resident_v1 COMMAND hiai_resident_v1_tests ${CMAKE_CURRENT_BINARY_DIR}/resident-v1-cases)
endif()

if(BUILD_HOST_TESTS)
  add_executable(hiai_resident_mixed_tests tests/test_resident_mixed.cpp src/resident_session.cpp
    src/hiai_executor.cpp tests/fake_hiai/fake_hiai.cpp)
  target_include_directories(hiai_resident_mixed_tests PRIVATE tests/fake_hiai)
  target_link_libraries(hiai_resident_mixed_tests PRIVATE dflash_core)
  add_test(NAME hiai_resident_mixed COMMAND hiai_resident_mixed_tests ${CMAKE_CURRENT_BINARY_DIR}/resident-mixed-cases)
endif()

# Text I/O is linked statically into the mixed supervisor. No board-side Python,
# shared tokenizer library, network fetch, or per-token helper process is used.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_TEST OFF CACHE BOOL "" FORCE)
set(INSTALL_DOCUMENTATION OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/oniguruma EXCLUDE_FROM_ALL)
add_library(qwen_utf8proc STATIC third_party/utf8proc/utf8proc.c)
target_include_directories(qwen_utf8proc PUBLIC third_party/utf8proc)
target_compile_definitions(qwen_utf8proc PUBLIC UTF8PROC_STATIC)
add_library(qwen_text_tokenizer STATIC src/tokenizer.cpp)
target_include_directories(qwen_text_tokenizer PUBLIC include)
target_link_libraries(qwen_text_tokenizer PRIVATE onig qwen_utf8proc)
add_executable(qwen35_tokenizer src/tokenizer_main.cpp)
target_link_libraries(qwen35_tokenizer PRIVATE qwen_text_tokenizer)
if(TARGET hiai_resident_mixed_adapter)
  target_compile_definitions(hiai_resident_mixed_adapter PRIVATE QWEN35_TEXT_TOKENIZER=1)
endif()
if(TARGET qwen35_board_resident_mixed)
  target_link_libraries(qwen35_board_resident_mixed PRIVATE qwen_text_tokenizer)
endif()
if(TARGET board_resident_mixed_hiai_fake)
  target_compile_definitions(board_resident_mixed_hiai_fake PRIVATE QWEN35_TEXT_TOKENIZER=1)
  target_link_libraries(board_resident_mixed_hiai_fake PRIVATE qwen_text_tokenizer)
endif()
