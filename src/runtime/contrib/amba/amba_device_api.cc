/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */


 /*!
  * \file amba_device_api.cc
  */

#include <dmlc/logging.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>

#include "amba_tvm.h"

namespace tvm {
namespace runtime {

class AmbaDeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(Device dev) final {}
  void GetAttr(Device dev, DeviceAttrKind kind, TVMRetValue* rv) final {
    if (kind == kExist) {
      *rv = 1;
    }
  }
  void* AllocDataSpace(Device dev, size_t nbytes, size_t alignment,
                       DLDataType type_hint) final {
    return AmbaDeviceAlloc(nbytes, alignment);
  }

  void FreeDataSpace(Device dev, void* ptr) final {
    AmbaDeviceFree(ptr);
  }

  void CopyDataFromTo(const void* from, size_t from_offset, void* to, size_t to_offset, size_t size,
                      Device dev_from, Device dev_to, DLDataType type_hint,
                      TVMStreamHandle stream) final {
    memcpy(static_cast<char*>(to) + to_offset, static_cast<const char*>(from) + from_offset, size);
  }

  void StreamSync(Device dev, TVMStreamHandle stream) final {}

  static AmbaDeviceAPI* Global() {
    // NOTE: explicitly use new to avoid exit-time destruction of global state
    // Global state will be recycled by OS as the process exits.
    static auto* inst = new AmbaDeviceAPI();
    return inst;
  }
};

TVM_REGISTER_GLOBAL("device_api.amba").set_body([](TVMArgs args, TVMRetValue* rv) {
  DeviceAPI* ptr = AmbaDeviceAPI::Global();
  *rv = static_cast<void*>(ptr);
});

}  // namespace runtime
}  // namespace tvm

