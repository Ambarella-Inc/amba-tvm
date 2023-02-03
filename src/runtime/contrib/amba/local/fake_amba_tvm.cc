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

#include "amba_tvm.h"

int GetAmbaTVMLibVersion(void)
{
  return AMBA_TVM_LIB_VERSION;
}
int InitAmbaTVM(void) {
  return 0;
}
int InitAmbaEngine(amba_engine_cfg_t * engine_cfg,
  amba_engine_io_t * engine_input,amba_engine_io_t * engine_output) {
  return 0;
}
int SetAmbaEngineInput(amba_engine_cfg_t *engine_cfg,
	const char *input_name, AmbaDLTensor *input) {
  return 0;
}
int RunAmbaEngine(amba_engine_cfg_t * engine_cfg,
	amba_perf_t *perf){
  return 0;
}
int GetAmbaEngineOutput(amba_engine_cfg_t *engine_cfg,
	const char *output_name, AmbaDLTensor *output){
  return 0;
}
int DeleteAmbaTVM(amba_engine_cfg_t *engine_cfgs, uint32_t num) {
  return 0;
}
int CheckAmbaEngineInputName(amba_engine_cfg_t * engine_cfg,
  const char * input_name) {
  return 0;
}
int CheckAmbaEngineOutputName(amba_engine_cfg_t * engine_cfg,
  const char * output_name) {
  return 0;
}
void* AmbaDeviceAlloc(unsigned long nbytes, unsigned long alignment)
{
  return nullptr;
}
int AmbaDeviceFree(void* ptr) {
  return 0;
}

