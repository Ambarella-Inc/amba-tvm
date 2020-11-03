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
 * \file src/relay/backend/contrib/cv22/codegen_cv22.cc
 * \brief Implementation of CV22 codegen APIs.
 */

#include <tvm/node/serialization.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <sstream>
#include <unordered_map>

#include "../../utils.h"

#include "../../../../runtime/contrib/cv22/cv22_module.h"
#include "../codegen_c/codegen_c.h"

namespace tvm {
namespace relay {
namespace contrib {

using namespace backend;

/*!
 * \brief Generates a TensorRTModule from a relay expression. This "compilation"
 * does not require TensorRT since the actual conversion using TensorRT APIs is
 * deferred until runtime. This step simply serializes the relay functions into
 * strings.
 */
class CV22ModuleCodegen : public CSourceModuleCodegenBase {
 public:
  /*!
   * \brief Serializes a function and stores it in serialized_subgraphs_ so that
   * it can be included in the TensorRT module.
   * \param func A relay function to add to the TensorRT module.
   */
  void GenFunc(const Function& func) {
    CHECK(func.defined()) << "Input error: expect a Relay function.";
    // Record the external symbol for runtime lookup.
    auto sid = GetExtSymbol(func);
    LOG(INFO) << "Running GenFunc for " << sid;

    subgraph_attr_t attr = {};

    // CVFlow compiler is expected to create ambapb and cavalry binaries for each cv22 subgraph
    // CV22_RAND_ID is a random session id appended to all filenames
    std::string cv22_rand_id = this->getEnvVar("CV22_RAND_ID");
    if (cv22_rand_id.empty()) {
        LOG(ERROR) << "Env variable CV22_RAND_ID not found";
        exit(-1);
    }
    else {
        LOG(INFO) << "Env variable CV22_RAND_ID set to " << cv22_rand_id << " by cvflow compiler";
    }
    attr.filename = sid + "_" + cv22_rand_id;

    // input list
    for (size_t i = 0; i < func->params.size(); ++i) {
        attr.inputs.push_back(func->params[i]->name_hint());
    }

    // get outputs from env variable created during cvflow compilation
    std::string output_list = this->getEnvVar("CV22_OUTPUTS_LIST");
    if (output_list.empty()) {
        LOG(ERROR) << "Env variable CV22_OUTPUTS_LIST not found";
        exit(-1);
    }
    else {
        LOG(INFO) << "Env variable CV22_OUTPUTS_LIST set to " << output_list << " by cvflow compiler";
    }

    std::stringstream ss(output_list);
    while (ss.good()) {
        std::string substr;
        getline(ss, substr, ',');
        attr.outputs.push_back(substr);
    }

    cv22_subgraphs_[sid] = attr;
  }

  /*!
   * \brief Creates the TensorRT module from the Relay function or IRModule.
   * \param ref An object ref that could be either a Relay function or IRModule.
   * \return The TensorRT runtime module.
   */
  runtime::Module CreateCSourceModule(const ObjectRef& ref) override {
    LOG(INFO) << "In function CreateCSourceModule ";
    if (ref->IsInstance<FunctionNode>()) {
      GenFunc(Downcast<Function>(ref));
    } else if (ref->IsInstance<IRModuleNode>()) {
      IRModule mod = Downcast<IRModule>(ref);
      for (const auto& it : mod->functions) {
        GenFunc(Downcast<Function>(it.second));
      }
    } else {
      LOG(FATAL)
          << "The input ref is expected to be a Relay function or module.";
    }

    // check for emulator or target build
    std::string emulator_env = this->getEnvVar("CV22_EMULATOR");

    if (emulator_env.empty()) {
        return runtime::AmbaModuleCreate(cv22_subgraphs_);
    }
    else {
        return runtime::CV22ModuleCreate(cv22_subgraphs_);
    }

  }

 private:
  /*! \brief Map of external symbol to serialized Relay functions. */
  std::unordered_map<std::string, subgraph_attr_t> cv22_subgraphs_;

  std::string getEnvVar(std::string const& key) {
    char * val = getenv( key.c_str() );

    // clear variable
    unsetenv(key.c_str());

    return val == NULL ? std::string("") : std::string(val);
  }

};

/*!
 * \brief The external compiler/codegen tool. It takes a Relay expression/module
 * and compiles it into a runtime module.
 */
runtime::Module CV22Compiler(const ObjectRef& ref) {
  CV22ModuleCodegen cv22_codegen;
  return cv22_codegen.CreateCSourceModule(ref);
}

TVM_REGISTER_GLOBAL("relay.ext.cv22").set_body_typed(CV22Compiler);

/*!
 * \brief Get TensorRT version that TVM was compiled against.
 * \return TensorRT version as a list of [major, minor, patch], or an empty list
 * if not compiled against TensorRT.
 */
Array<Integer> GetCV22Version() {
#if TVM_GRAPH_RUNTIME_TENSORRT
  return {Integer(NV_TENSORRT_MAJOR), Integer(NV_TENSORRT_MINOR),
          Integer(NV_TENSORRT_PATCH)};
#else
  return {};
#endif  // TVM_GRAPH_RUNTIME_TENSORRT
}

TVM_REGISTER_GLOBAL("relay._transform.GetCV22Version")
    .set_body_typed(GetCV22Version);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm

