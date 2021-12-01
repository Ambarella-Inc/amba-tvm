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
 * \file runtime/contrib/cv22/cv22_module.cc
 * \brief CV22Module is the runtime module for cv22 backend.
 */

#include <stdlib.h>
#include <sys/stat.h>
#include <tvm/node/serialization.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/ndarray.h>

#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include "../../file_utils.h"
#include "cv22_module.h"

namespace tvm {
namespace runtime {

/*! \brief A module for CV22 runtime. */
class CV22Module : public runtime::ModuleNode {
 public:
  explicit CV22Module(
      const std::unordered_map<std::string, subgraph_attr_t>& cv22_subgraphs)
      : cv22_subgraphs_(cv22_subgraphs) {
      LOG(INFO) << "CV22Module Constructor";
  }

  ~CV22Module() {
      LOG(INFO) << "CV22Module Destructor";
  }

  PackedFunc GetFunction(const std::string& name,
                         const ObjectPtr<Object>& sptr_to_self) final {
    // Returning nullptr tells TVM that the function is not in this module, so
    // it can look for the correct one.
    auto it_subgraph = cv22_subgraphs_.find(name);
    if (it_subgraph == cv22_subgraphs_.end()) {
      return PackedFunc(nullptr);
    }
    // Generate an external packed function
    return PackedFunc([this, name](tvm::TVMArgs args, tvm::TVMRetValue* rv) {

      LOG(INFO) << "CV22Module GetFunction PackedFunc";

      std::string out_dir = "/tmp/test_amba/eval/";

      std::string ambapb_fpath = out_dir + cv22_subgraphs_[name].filename + ".ambapb.ckpt.onnx";
      LOG(INFO) << "Filename: " << ambapb_fpath;
      std::string cmd = "evaluate.py --metagraph " + ambapb_fpath;

      // Save inputs to file
      std::vector<std::string>& inputs = cv22_subgraphs_[name].inputs;
      for (size_t i = 0; i < inputs.size(); ++i) {
          LOG(INFO) << "Input " << i << ": " << inputs[i];

          DLTensor* arg = args[i];
          float* data = reinterpret_cast<float*>(arg->data);

          int buf_size = 1;
          LOG(INFO) << "Shape:";
          for (int j = 0; j < arg->ndim; ++j) {
               LOG(INFO) << arg->shape[j];
               buf_size *= arg->shape[j];
          }
          LOG(INFO) << "Size: " << buf_size;

          std::string in_fname = out_dir + inputs[i] + ".bin";
          std::ofstream fout;
          fout.open(in_fname, std::ios::binary);
          if (fout.is_open()) {
              fout.write((char*) data, buf_size*sizeof(float));
          }
          else {
              LOG(ERROR)  << "Unable to open file";
              exit(-1);
          }
          fout.close();

          cmd += " --inputdata " + inputs[i] + "=" + in_fname;
      }

      // Run ades
      cmd += " --output_folder /tmp/test_amba/eval/outputs --log_dir /tmp/test_amba/eval/logs";
      LOG(INFO) << "Cmd: " << cmd;
      int ret = system(cmd.c_str());
      if (ret == -1) {
          LOG(ERROR) << "evaluate.py failed!";
          exit(-1);
      }

      // Read outputs from file
      std::vector<std::string>& outputs = cv22_subgraphs_[name].outputs;
      int out_idx = inputs.size();
      for (size_t o = 0; o < outputs.size(); ++o, ++out_idx) {
          LOG(INFO) << "Output " << o << ": " << outputs[o];

          // mangle tensor name
          std::string out_tname, pat, rep;
          size_t start_pos;

          out_tname = outputs[o];

          // replace "." with "_dt_"
          pat = std::string(".");
          rep = std::string("_dt_");
          start_pos = out_tname.find(pat);
          out_tname.replace(start_pos, pat.length(), rep);

          std::string out_fname = "/tmp/test_amba/eval/outputs/" + out_tname + "_iter0.bin";
          std::ifstream fin;
          fin.open(out_fname, std::ios::binary);

          if (fin.is_open()) {
              DLTensor* arg = args[out_idx];
              float* data = reinterpret_cast<float*>(arg->data);

              // get length of the file
              fin.seekg(0, std::ios::end);
              int size = fin.tellg();
              fin.seekg(0, std::ios::beg);

              fin.read(reinterpret_cast<char*>(data), size);
              LOG(INFO) << "Number of bytes read: " << fin.gcount();

              fin.close();
          }

          else {
              LOG(ERROR) << "Unable to open file " << out_fname;
              exit(-1);
          }
      }
    });
  }

  const char* type_key() const { return "cv22"; }

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
    std::string serialized_module(size, ' ');
    filep.seekg(0);
    filep.read(&serialized_module[0], size);
    return CreateAmbaModuleFromString(serialized_module);
  }

  static Module LoadFromBinary(void* strm) {
    dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
    std::string serialized_module;
    stream->Read(&serialized_module);
    return CreateAmbaModuleFromString(serialized_module);
  }

 private:
  std::unordered_map<std::string, subgraph_attr_t> cv22_subgraphs_;

  /*! \brief Serialize this module to a string. To be used during codegen. */
  std::string SerializeAmbaModuleToString() {
    // split cv22_subgraphs_ to three groups:
    // func_name : filename std::unordered_map<std::string, std::string>
    // func_name: input names std::unordered_map<std::string, std::vector<std::string> >
    // func_name: output names std::unordered_map<std::string, std::vector<std::string> >
    std::unordered_map<std::string, std::string> amba_fnames;
    std::unordered_map<std::string, std::vector<std::string> > amba_input_names;
    std::unordered_map<std::string, std::vector<std::string> > amba_output_names;
    for (const auto& graph_it: cv22_subgraphs_) {
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
    // combine cv22_subgraphs_ by three groups:
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
    auto n = make_object<CV22Module>(amba_subgraphs);
    return Module(n);
  }
 };

Module CV22ModuleCreate(
    const std::unordered_map<std::string, subgraph_attr_t>& cv22_subgraphs) {
  LOG(INFO) << "In CV22ModuleCreate";
  auto n = make_object<CV22Module>(cv22_subgraphs);
  return Module(n);
}

TVM_REGISTER_GLOBAL("runtime.module.loadfile_cv22")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = CV22Module::LoadFromFile(args[0]);
});

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_cv22")
.set_body_typed(CV22Module::LoadFromBinary);

}  // namespace runtime
}  // namespace tvm
