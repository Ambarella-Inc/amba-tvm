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
 * \file src/runtime/contrib/amba/amba_module.cc
 * \brief Ambarella runtime module for TVM.
 */
#include <stdlib.h>
#include <tvm/node/serialization.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/ndarray.h>

#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include "../../file_utils.h"

#include "../cv22/cv22_module.h"
#include "amba_tvm.h"


namespace tvm {
namespace runtime {

/*!
 * \brief Engine for target runtime.
 * \param engine_id_ The id of engine.
 * \param engine_name_ The func_name of ubgraph.
 * \param engine_in_ The input pairs of name and buffer for engine.
 * \param engine_out_ The output paris of name and buffer for engine.
 *
 * To run one subgraph on Ambarella target, one Ambarella engine
 * should be built first. Then this built engine could be executed
 * on Ambarella target.
 */

typedef struct {
  unsigned long engine_id_;
  std::string engine_name_;
  std::vector<std::pair<std::string, DLTensor*> > engine_in_;
  std::vector<std::pair<std::string, DLTensor*> > engine_out_;
} AmbaEngineContext;

/*!
 * \brief AmbaModule is for Ambarella target backend.
 */
class AmbaModule : public runtime::ModuleNode {
 public:
  explicit AmbaModule(
    const std::unordered_map<std::string, subgraph_attr_t>& amba_subgraphs)
    : amba_subgraphs_(amba_subgraphs) {
    CHECK(GetAmbaTVMLibVersion() >= 0x1)
      << "AmbaTVM library version should not be less than " << 0x1;
    // general init
    InitAmbaTVM();
  }
  ~AmbaModule() {
    std::vector <amba_engine_cfg_t> engine_cfgs;
    for (const auto &engine_it: amba_engine_cache_) {
      amba_engine_cfg_t engine_cfg;
      engine_cfg.engine_id = engine_it.second.engine_id_;
      engine_cfg.engine_name = engine_it.second.engine_name_.c_str();
      engine_cfg.engine_filepath = nullptr;
      engine_cfgs.push_back(engine_cfg);
    }
    // deinit general mem and network mem
    CHECK(GetAmbaTVMLibVersion() >= 0x1)
      << "AmbaTVM library version should not be less than " << 0x1;
    DeleteAmbaTVM(engine_cfgs.data(), engine_cfgs.size());
  }

  PackedFunc GetFunction(const std::string& name,
                         const ObjectPtr<Object>& sptr_to_self) final {
    // Returning nullptr tells TVM that the function is not in this module, so
    // it can look for the correct one.
    auto it_subgraph = amba_subgraphs_.find(name);
    if (it_subgraph == amba_subgraphs_.end()) {
      return PackedFunc(nullptr);
    }

    // Generate an external packed function
    return PackedFunc([this, name](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      CHECK(GetAmbaTVMLibVersion() >= 0x1)
        << "AmbaTVM library version should not be less than " << 0x1;
      RunAmbaCVFlow(name, args, rv);
      /*fprintf(stderr, "[ AmbaModule ] [ %s ] cvflow time: %d us\n",
        name.c_str(), (*rv).operator int());*/
    });
  }

  const char* type_key() const { return "amba"; }

  void SaveToFile(const std::string& file_name,
                  const std::string& format) final {
    std::string fmt = runtime::GetFileFormat(file_name, format);
    CHECK_EQ(fmt, type_key()) << "Can only save to format=" << type_key();
    SaveBinaryToFile(file_name, SerializeAmbaModuleToString());
  }

  void SaveToBinary(dmlc::Stream* stream) final {
    stream->Write(SerializeAmbaModuleToString());
  }

  static Module LoadFromFile(const std::string& path) {
    std::ifstream filep(path);
    filep.seekg(0, std::ios::end);
    size_t size = filep.tellg();
    std::string amba_module(size, ' ');
    filep.seekg(0);
    filep.read(&amba_module[0], size);
    return CreateAmbaModuleFromString(amba_module);
  }

  static Module LoadFromBinary(void* strm) {
    dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
    std::string amba_module;
    stream->Read(&amba_module);
    return CreateAmbaModuleFromString(amba_module);
  }

 private:
  // map func_name to Ambarella binary
  std::unordered_map<std::string, subgraph_attr_t> amba_subgraphs_;
  // map func_name to Ambarella Engine
  std::unordered_map<std::string, AmbaEngineContext> amba_engine_cache_;

  std::vector<DLTensor*> ConvertArgs(tvm::TVMArgs args) {
    std::vector<DLTensor*> io_args(args.size(), nullptr);
    for (int32_t i = 0; i < args.size(); ++i) {
      if (args[i].type_code() == kTVMNDArrayHandle) {
        // Relay Debug/VM uses NDArray
        runtime::NDArray array = args[i];
        io_args[i] = const_cast<DLTensor*>(array.operator->());
      } else if (args[i].type_code() == kTVMDLTensorHandle) {
        // Graph runtime uses DLTensors
        io_args[i] = args[i];
      } else {
        LOG(FATAL) << "Invalid TVMArgs type.";
      }
    }
    return io_args;
  }

  /* AmbaDLTensor is for AmbaDevice */
  void ConvertToAmbaTensor(DLTensor *dl_arg, AmbaDLTensor *amba_arg) {
    int size = 1;
    amba_arg->data_virt = dl_arg->data;
    amba_arg->device_type = dl_arg->ctx.device_type;
    amba_arg->device_id = dl_arg->ctx.device_id;
    amba_arg->ndim = dl_arg->ndim;
    amba_arg->dtype_code = dl_arg->dtype.code;
    amba_arg->dtype_bits = dl_arg->dtype.bits;
    amba_arg->dtype_lanes = dl_arg->dtype.lanes;
    amba_arg->shape = dl_arg->shape;
    amba_arg->strides = dl_arg->strides;
    amba_arg->byte_offset = dl_arg->byte_offset;

    /* Tensor with no padding */
    size = 1;
    for (int i = 0; i < dl_arg->ndim; ++i) {
      size *= dl_arg->shape[i];
    }
    size *= (dl_arg->dtype.bits * dl_arg->dtype.lanes + 7) / 8;
    amba_arg->size = size;
  }

  AmbaEngineContext BuildAmbaEngine(tvm::TVMArgs args,
    const std::string& name, subgraph_attr_t amba_subgraph) {
    AmbaEngineContext engine_ctx;
    amba_engine_cfg_t engine_cfg;
    amba_engine_io_t  engine_input, engine_output;
    std::vector<AmbaDLTensor> in_tensors, out_tensors;
    std::vector<const char*> in_names, out_names;
    int num_inputs  = amba_subgraph.inputs.size();
    int num_outputs  = amba_subgraph.outputs.size();

    auto io_args = ConvertArgs(args);

    // engine cfg
    engine_cfg.engine_name = name.c_str();
    engine_cfg.engine_filepath = amba_subgraph.filename.c_str();

    // engine input
    in_tensors.resize(num_inputs);
    in_names.resize(num_inputs);
    auto input_it = amba_subgraph.inputs.begin();
    for (int i = 0; i < num_inputs; ++ i, ++ input_it) {
      ConvertToAmbaTensor(io_args[i], &in_tensors[i]);
      in_names[i] = input_it->c_str();
    }
    engine_input.num = num_inputs;
    engine_input.tensors = in_tensors.data();
    engine_input.names = in_names.data();

    // engine output
    out_tensors.resize(num_outputs);
    out_names.resize(num_outputs);
    auto output_it = amba_subgraph.outputs.begin();
    const int entry_in_storage = io_args.size() - num_outputs;
    for (int i = 0; i < num_outputs; ++ i, ++ output_it) {
      ConvertToAmbaTensor(io_args[entry_in_storage + i], &out_tensors[i]);
      out_names[i] = output_it->c_str();
    }
    engine_output.num = num_outputs;
    engine_output.tensors = out_tensors.data();
    engine_output.names = out_names.data();

    InitAmbaEngine(&engine_cfg, &engine_input, &engine_output);
    engine_ctx.engine_id_ = engine_cfg.engine_id;
    engine_ctx.engine_name_ = name;

    // vector inputs has the same order as in args
    input_it = amba_subgraph.inputs.begin();
    for (int i = 0; i < num_inputs; ++ i, ++ input_it) {
      // check for safety
      if (CheckAmbaEngineInputName(&engine_cfg,
        input_it->c_str()) == 0) {
        engine_ctx.engine_in_.push_back(std::make_pair(*input_it,
          io_args[i]));
      }
    }

    // vector outputs has the same order as in args
    output_it = amba_subgraph.outputs.begin();
    for (int i = 0; i < num_outputs; ++ i, ++output_it) {
      // check for safety
      if (CheckAmbaEngineOutputName(&engine_cfg,
        output_it->c_str()) == 0) {
        engine_ctx.engine_out_.push_back(std::make_pair(*output_it,
          io_args[entry_in_storage + i]));
      }
    }

    return engine_ctx;
  }

  void ExecuteAmbaEngine(const AmbaEngineContext& engine_and_context,
                     tvm::TVMArgs args, tvm::TVMRetValue* rv) {
    int rval = 0;
    AmbaDLTensor amba_arg;
    amba_perf_t perf;
    amba_engine_cfg_t engine_cfg;
    engine_cfg.engine_id = engine_and_context.engine_id_;
    engine_cfg.engine_name = engine_and_context.engine_name_.c_str();
    engine_cfg.engine_filepath = nullptr;

    do {
      for (const auto& input_it: engine_and_context.engine_in_) {
        ConvertToAmbaTensor(input_it.second, &amba_arg);
        rval = SetAmbaEngineInput(&engine_cfg,
          input_it.first.c_str(), &amba_arg);
        if (rval < 0) break;
      }
      if (rval < 0) break;

      rval = RunAmbaEngine(&engine_cfg, &perf);
      if (rval < 0) break;

      for (const auto& output_it: engine_and_context.engine_out_) {
        ConvertToAmbaTensor(output_it.second, &amba_arg);
        rval = GetAmbaEngineOutput(&engine_cfg,
          output_it.first.c_str(), &amba_arg);
        if (rval < 0) break;
      }
      if (rval < 0) break;
      *rv = static_cast<int>(perf.cvflow_time_us);
    } while(0);
  }

  void RunAmbaCVFlow(const std::string& name,
    tvm::TVMArgs args, tvm::TVMRetValue* rv){
      auto it = amba_engine_cache_.find(name);
      if (it == amba_engine_cache_.end()) {
        auto engine_and_context = BuildAmbaEngine(args, name, amba_subgraphs_[name]);
        amba_engine_cache_[name] = engine_and_context;
        ExecuteAmbaEngine(engine_and_context, args, rv);
      } else {
        ExecuteAmbaEngine(it->second, args, rv);
      }
  }

  /*! \brief Serialize this module to a string. To be used during codegen. */
  std::string SerializeAmbaModuleToString() {
    // split amba_subgraphs_ to three groups:
    // func_name : filename std::unordered_map<std::string, std::string>
    // func_name: input names std::unordered_map<std::string, std::vector<std::string> >
    // func_name: output names std::unordered_map<std::string, std::vector<std::string> >
    std::unordered_map<std::string, std::string> amba_fnames;
    std::unordered_map<std::string, std::vector<std::string> > amba_input_names;
    std::unordered_map<std::string, std::vector<std::string> > amba_output_names;
    for (const auto& graph_it: amba_subgraphs_) {
      const subgraph_attr_t& attr = graph_it.second;
      amba_fnames.emplace(graph_it.first, attr.filename);
      amba_input_names.emplace(graph_it.first, attr.inputs);
      amba_output_names.emplace(graph_it.first, attr.outputs);
    }
    std::ostringstream os;
    dmlc::JSONWriter writer(&os);
    writer.BeginObject();
    writer.WriteObjectKeyValue("filenames", amba_fnames);
    writer.WriteObjectKeyValue("inputnames", amba_input_names);
    writer.WriteObjectKeyValue("outputnames", amba_output_names);
    writer.EndObject();
    return os.str();
  }

  /*! \brief Load serialized module from string created by SerializeAmbaModuleToString. */
  static Module CreateAmbaModuleFromString(const std::string& str) {
    // combine amba_subgraphs_ by three groups:
    // func_name : filename std::unordered_map<std::string, std::string>
    // func_name: input names std::unordered_map<std::string, std::vector<std::string> >
    // func_name: output names std::unordered_map<std::string, std::vector<std::string> >
    std::unordered_map<std::string, std::string> amba_fnames;
    std::unordered_map<std::string, std::vector<std::string> > amba_input_names;
    std::unordered_map<std::string, std::vector<std::string> > amba_output_names;
    std::unordered_map<std::string, subgraph_attr_t> amba_subgraphs;
    std::istringstream is(str);
    dmlc::JSONReader reader(&is);
    dmlc::JSONObjectReadHelper helper;
    helper.DeclareField("filenames", &amba_fnames);
    helper.DeclareField("inputnames", &amba_input_names);
    helper.DeclareField("outputnames", &amba_output_names);
    helper.ReadAllFields(&reader);
    for(const auto& it: amba_fnames) {
      std::string func_name = it.first;
      amba_subgraphs[func_name].filename = it.second;
      amba_subgraphs[func_name].inputs = amba_input_names[func_name];
      amba_subgraphs[func_name].outputs = amba_output_names[func_name];
    }
    auto n = make_object<AmbaModule>(amba_subgraphs);
    return Module(n);
  }
};

Module AmbaModuleCreate(
  const std::unordered_map<std::string, subgraph_attr_t>& amba_subgraphs) {
  // TBD, rewrite filename to help locate Ambarella binaries on EVK
  std::unordered_map<std::string, subgraph_attr_t> subgraphs;
  for (const auto &graph_it: amba_subgraphs) {
    std::string func_name = graph_it.first;
    subgraphs[func_name] =  graph_it.second;
    subgraphs[func_name].filename = subgraphs[func_name].filename + ".amba";
  }
  auto n = make_object<AmbaModule>(subgraphs);
  return Module(n);
}

TVM_REGISTER_GLOBAL("runtime.module.loadfile_amba")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = AmbaModule::LoadFromFile(args[0]);
});

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_amba")
.set_body_typed(AmbaModule::LoadFromBinary);

}  // namespace runtime
}  // namespace tvm


