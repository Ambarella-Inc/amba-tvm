/* * Licensed to the Apache Software Foundation (ASF) under one
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
 * \file runtime/contrib/cv22/cv22_module.h
 * \brief CV22Module is the runtime module for cv22 backend.
 */

#ifndef TVM_RUNTIME_CONTRIB_CV22_CV22_MODULE_H_
#define TVM_RUNTIME_CONTRIB_CV22_Cv22_MODULE_H_

#include <tvm/ir/module.h>

#include <string>
#include <vector>
#include <unordered_map>

typedef struct {
    std::string filename;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
} subgraph_attr_t;

namespace tvm {
namespace runtime {

/*!
 * \brief Create a CV22Module.
 * \param cv22_subgraphs <TBD>
 * \return CV22Module created from subgraphs.
 */
Module CV22ModuleCreate(
    const std::unordered_map<std::string, subgraph_attr_t>& cv22_subgraphs);

/*!
 * \brief Create a AmbaModule.
 * \param dag_subgraphs <TBD>
 * \return AmbaModule created from subgraphs.
 */
Module AmbaModuleCreate(
  const std::unordered_map<std::string, subgraph_attr_t>& amba_subgraphs);

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_CV22_CV22_MODULE_H_
