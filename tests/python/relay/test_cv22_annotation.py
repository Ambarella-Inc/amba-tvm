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
"""Unit tests for graph partitioning."""
import os
import numpy as np

import tvm
import tvm.relay.testing
from tvm import relay
from tvm.relay import transform
from tvm.relay.op.annotation import compiler_begin, compiler_end

# Imports for expr
from tvm.relay.expr_functor import ExprMutator, ExprVisitor
from tvm.relay.function import Function

# Onnx imports
#from tvm.relay.converter import to_onnx
from tvm.contrib.target.onnx import to_onnx

import onnx
import mxnet as mx
from gluoncv import model_zoo, data, utils
import gluoncv
import tvm.relay.op.contrib.cv22
from tvm.relay.backend.contrib.cv22 import PruneSubgraphs

def partitions_to_modules(mod):
    module_dict = {}
    original_mod = mod
    for func in mod.get_global_vars():
        name = func.name_hint
        if "cv22" in name:
            mod = tvm.IRModule.from_expr(original_mod[name])
            new_mod = tvm.ir.module.IRModule()
            new_mod["main"] = mod[name]
            module_dict[name] = new_mod

    return module_dict


def is_cv22_first_subgraph(mod):
    class CV22SubgraphChecker(ExprVisitor):
        def __init__(self):
            print('init')
            ExprVisitor.__init__(self)
            self.is_cv22_first_subgraph = False
            self.visit_counter = 0

        def visit_call(self, call):
            super().visit_call(call)
            if self.visit_counter == 0 and isinstance(call.op, tvm.ir.expr.GlobalVar):
                name = call.op.name_hint
                if "cv22" in name:
                    self.is_cv22_first_subgraph = True
            self.visit_counter += 1

    checker = CV22SubgraphChecker()
    checker.visit(mod["main"])
    return checker.is_cv22_first_subgraph

def test_manual():
    #============= Constructing a simple graph ==============
    dtype = "float32"
    i0_shape = (1, 3, 224, 224) # NCHW
    w0_shape = (32, 3, 3, 3)    # OIHW
    i1_shape = (1, 32, 224, 224) # NCHW
    out_shape = (1, 32, 224, 224) # NCHW

    data0 = relay.var('data0', shape=(i0_shape), dtype=dtype)
    weight0 = relay.var('weight0', shape=(w0_shape), dtype=dtype)
    data1 = relay.var('data1', shape=(i1_shape), dtype=dtype)

    begin0 = relay.annotation.compiler_begin(data0, "cv22")
    begin1 = relay.annotation.compiler_begin(weight0, "cv22")
    begin2 = relay.annotation.compiler_begin(data1, "cv22")

    node0  = relay.nn.conv2d(begin0,
                             begin1,
                             kernel_size=(3, 3),
                             padding=(1, 1),
                             kernel_layout = 'OIHW')
    node1  = relay.add(node0, begin2)

    # whole graph offload
    out2 = relay.annotation.compiler_end(node1, "cv22")
    f2   = relay.Function([data0, weight0, data1], out2)
    mod2 = tvm.IRModule.from_expr(f2)

    print('---------- Annotated graph ----------')
    print(mod2.astext(show_meta_data=False))

    # graph partitioning
    mod2_partition = transform.PartitionGraph()(mod2)
    print('---------- Partitioned graph ----------')
    print(mod2_partition.astext(show_meta_data=False))

    # conver to onnx
    module_list = partitions_to_modules(mod2_partition)
    module_list = partitions_to_modules(mod)
    for name, module in module_list.items():
        onnx_path = name+".onnx"
        onnx_model = to_onnx(module, {}, name, path=onnx_path)

def test_classification():
    # TODO: Debug errors. ## == TVM ERROR in parsing. # == Some error in onnx or partitioning

    model_list = {
        'ResNet18_v1' : 'ResNet18_v1',
        'MobileNet1.0' : 'MobileNet1',
        'VGG11' : 'VGG11',
        'SqueezeNet1.0' : 'SqueezeNet1.0',
        'DenseNet121' : 'DenseNet121',
        'AlexNet' : 'AlexNet',
        'InceptionV3' : 'InceptionV3',

    }

    for model_name in model_list:
        print(model_name)
        model = gluoncv.model_zoo.get_model(model_name, pretrained=True)
        mod, params = relay.frontend.from_mxnet(model, {'data':(1,3,224,224)})

        print('---------- Original Graph ----------')
        mod = relay.transform.RemoveUnusedFunctions()(mod)
        print(mod.astext(show_meta_data=False))
        print('---------- Bound Graph ----------')
        from tvm.relay.build_module import bind_params_by_name
        mod['main'] = bind_params_by_name(mod['main'], params)
        print(mod.astext(show_meta_data=False))
        print("---------- Annotated Graph ----------")
        mod = transform.AnnotateTarget("cv22")(mod)
        print(mod.astext(show_meta_data=False))
        print("---------- Merge Compiler Regions ----------")
        mod = transform.MergeCompilerRegions()(mod)
        print(mod.astext(show_meta_data=False))
        print("---------- Partioned Graph ----------")
        mod = transform.PartitionGraph()(mod)
        print(mod.astext(show_meta_data=False))
        print("---------- Pruned Graph ----------")
        mod = PruneSubgraphs(mod, "cv22", 2)
        print(mod.astext(show_meta_data=False))

        module_list = partitions_to_modules(mod)
        for name, module in module_list.items():
            onnx_path = name+".onnx"
            onnx_model = to_onnx(module, {}, name, path=onnx_path)

def test_mxnet_mobilenet_ssd():

    from tvm.contrib.download import download_testdata
    im_fname = download_testdata('https://github.com/dmlc/web-data/blob/master/' +
                                 'gluoncv/detection/street_small.jpg?raw=true',
                                 'street_small.jpg', module='data')

    #model = 'ssd_512_mobilenet1.0_coco'
    model = 'ssd_512_mobilenet1.0_voc'
    image_size = 512

    input_name = 'data'
    input_shape = (1, 3, image_size, image_size)
    dtype = 'float32'
    x, img = data.transforms.presets.ssd.load_test(im_fname, short=image_size)
    block = model_zoo.get_model(model, pretrained=True)

    block.hybridize()
    block.forward(x)
    block.export('temp')
    #with open('temp-symbol.json') as f:
    #    model_json = json.load(f)
    #model_json["heads"] = [[700, 0, 0]]
    #with open('temp-symbol.json', 'w') as json_file:
    #    json.dump(model_json, json_file)

    model_json = mx.symbol.load('temp-symbol.json')
    save_dict = mx.ndarray.load('temp-0000.params')
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        elif tp == 'aux':
            aux_params[name] = v
    mod, params = relay.frontend.from_mxnet(model_json, {input_name: input_shape}, arg_params=arg_params, aux_params=aux_params)

    print('---------- Original Graph ----------')
    mod = relay.transform.RemoveUnusedFunctions()(mod)
    print(mod.astext(show_meta_data=False))
    print('---------- Bound Graph ----------')
    from tvm.relay.build_module import bind_params_by_name
    mod['main'] = bind_params_by_name(mod['main'], params)
    print(mod.astext(show_meta_data=False))
    print("---------- Annotated Graph ----------")
    mod = transform.AnnotateTarget("cv22")(mod)
    print(mod.astext(show_meta_data=False))
    print("---------- Merge Compiler Regions ----------")
    mod = transform.MergeCompilerRegions()(mod)
    print(mod.astext(show_meta_data=False))
    print("---------- Partioned Graph ----------")
    mod = transform.PartitionGraph()(mod)
    print(mod.astext(show_meta_data=False))
    print("---------- Pruned Graph ----------")
    mod = PruneSubgraphs(mod, "cv22", 2)
    print(mod.astext(show_meta_data=False))

    module_list = partitions_to_modules(mod)
    for name, module in module_list.items():
        onnx_path = name+".onnx"
        print('converting subgraph '+onnx_path)
        onnx_model = to_onnx(module, {}, name, path=onnx_path)


if __name__ == '__main__':
    #test_manual()
    #test_classification()
    test_mxnet_mobilenet_ssd()
