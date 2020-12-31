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

# This is Makefile to build fake AmbaTVM library.
# This fake library is for Emulator mode only

LOCAL_AMBA_SRC := fake_amba_tvm.cc
LOCAL_AMBA_TARGET := libfamba_tvm.so
LOCAL_AMBA_INC ?= /home/dlr/prebuild/amba/include

LOCAL_CFLAGS    := -std=c++11 -shared -fPIC -I${LOCAL_AMBA_INC}

CXX := /usr/bin/c++
STRIP := /usr/bin/strip

.PHONY: ${LOCAL_AMBA_TARGET} clean

${LOCAL_AMBA_TARGET}:
	@${CXX} -o $@ ${LOCAL_AMBA_SRC} ${LOCAL_CFLAGS}
	@${STRIP} ${LOCAL_AMBA_TARGET}

clean:
	@rm -rf ${LOCAL_AMBA_TARGET}
