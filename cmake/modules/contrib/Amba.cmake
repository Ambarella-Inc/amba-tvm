# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

if(USE_AMBA_RUNTIME)
    message(STATUS "Build with Ambarella runtime")
    if(IS_DIRECTORY ${USE_AMBA_RUNTIME})
      set(AMBA_ROOT_DIR ${USE_AMBA_RUNTIME})
      message(STATUS "Custom Ambarella path: " ${AMBA_ROOT_DIR})
    endif()

    find_path(AMBA_INCLUDE_DIR amba_tvm.h HINTS ${AMBA_ROOT_DIR} PATH_SUFFIXES include)
    if(EXISTS ${AMBA_INCLUDE_DIR})
      include_directories(${AMBA_INCLUDE_DIR})
    else(AMBA_INCLUDE_DIR)
      message(ERROR " Could not find Ambarella include files.")
    endif()

    if(NOT USE_AMBA_TOOLCHAIN)
      find_library(AMBA_LIB_DIR famba_tvm HINTS ${AMBA_ROOT_DIR} PATH_SUFFIXES lib)
      if (EXISTS ${AMBA_LIB_DIR})
        list(APPEND TVM_RUNTIME_LINKER_LIBS ${AMBA_LIB_DIR})
      else (AMBA_LIB_DIR)
        message(ERROR " Could not find Ambarella lib files.")
      endif()
    endif()

    file(GLOB AMBA_CONTRIB_SRC src/runtime/contrib/amba/amba_module.cc
        src/runtime/contrib/amba/amba_device_api.cc)
    list(APPEND RUNTIME_SRCS ${AMBA_CONTRIB_SRC})

endif()

if (USE_AMBA_TOOLCHAIN)
    set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++ CACHE FILEPATH "CXX compiler." FORCE)
    set(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc CACHE FILEPATH "C compiler." FORCE)
endif()

