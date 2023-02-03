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

"""Unit tests for cv22 compilation"""

import os
import numpy as np
from shutil import rmtree
from matplotlib import pyplot as plt

# tvm imports
import tvm
import tvm.relay.testing
from tvm import relay
from tvm.relay import transform
from tvm.relay.op.annotation import compiler_begin, compiler_end
from tvm.relay.build_module import bind_params_by_name

# expr imports
from tvm.relay.expr_functor import ExprMutator, ExprVisitor
from tvm.relay.function import Function

# onnx imports
from tvm.contrib.target.onnx import to_onnx
import onnx

# mxnet imports
import mxnet as mx
from gluoncv import model_zoo, data, utils

def create_cvflow_dir(dir_name):
    '''
    Remove existing dir and create new one
    '''
    rmtree(dir_name, ignore_errors=True)

    dir_name += 'prepare/'
    os.makedirs(dir_name)

    return dir_name

def compile_model(mod, params, compiler, output_basename='compiled'):
    '''
    1) Annotate ops belonging to cv22 white list as "cv22"
    2) Partition graph to create multiple subgraphs
    3) Prune subgraphs to retain only one cv22 subgraph (Note: currently it is the first cv22 subgraph)
    4) Convert cv22 subgraphs to onnx
    5) Compile onnx models using cvflow compiler
    6) Call relay.build
    7) Serialize to files
    '''
    try:
        # cvflow imports
        import tvm.relay.op.contrib.cv22
        from tvm.relay.backend.contrib.cv22 import PruneSubgraphs, PartitionsToModules, GetCvflowExecutionMode, CvflowCompilation, CVFlowTVMWrapper

        print('---------- Original Graph ----------')
        mod = transform.RemoveUnusedFunctions()(mod)
        print(mod.astext(show_meta_data=False))

        print('---------- NHWC -> NCHW ----------')
        mod = transform.ConvertLayout({'nn.conv2d': ['NCHW', 'default']})(mod)
        print(mod.astext(show_meta_data=False))

        print('---------- Bound Graph ----------')
        if params:
            mod['main'] = bind_params_by_name(mod['main'], params)
        print(mod.astext(show_meta_data=False))

        print("---------- Annotated Graph ----------")
        mod = transform.AnnotateTarget(compiler)(mod)
        print(mod.astext(show_meta_data=False))

        print("---------- Merge Compiler Regions ----------")
        mod = transform.MergeCompilerRegions()(mod)
        print(mod.astext(show_meta_data=False))

        print("---------- Partioned Graph ----------")
        mod = transform.PartitionGraph()(mod)
        print(mod.astext(show_meta_data=False))

        print("---------- Pruned Graph ----------")
        mod = PruneSubgraphs(mod, compiler, 1)
        print(mod.astext(show_meta_data=False))

        output_folder = os.path.join('/tmp/test_amba/', 'prepare')
        rmtree(output_folder, ignore_errors=True)
        os.makedirs(output_folder)

        module_list = PartitionsToModules(mod, compiler)
        for name, module in module_list.items():
            print("---------- Converting subgraph %s to onnx ----------" % name)
            onnx_model = to_onnx(module, {}, name)

            print("---------- Invoking Cvflow Compilation ----------")
            save_path = CvflowCompilation(model_proto=onnx_model, \
                                          output_name=name, \
                                          output_folder=output_folder)
            print('Saved compiled model to: %s\n' % save_path)

        # mode: EMULATOR or TARGET
        exe_mode = GetCvflowExecutionMode()

        # initialize wrapper
        ct = CVFlowTVMWrapper(exe_mode)

        # tvm compilation
        ct.relay_build(mod)

        # serialize
        json_fname, lib_fname, params_fname = ct.serialize(basename=output_basename)

        return json_fname, lib_fname, params_fname

    except:
        raise ValueError('Cvflow compilation failed')

def run_model(inputs_map, json_fname, lib_fname, params_fname):
    try:
        # cvflow imports
        from tvm.relay.backend.contrib.cv22 import GetCvflowExecutionMode, CVFlowTVMWrapper

        # mode: EMULATOR or TARGET
        exe_mode = GetCvflowExecutionMode()

        # initialize wrapper
        ct = CVFlowTVMWrapper(exe_mode)

        # deserialize
        ct.deserialize(json_fname, lib_fname, params_fname)

        # run
        outputs = ct.tvm_runtime(inputs_map)

        return outputs

    except:
        raise ValueError('run_model failed')

def test_manual():
    #============= Constructing a simple graph ==============
    dtype = "float32"
    i_shape = (1,3,224,224) # NCHW

    compiler = 'cv22'

    data0_arr = np.random.randint(0, 255, i_shape).astype(dtype)
    data0 = relay.var('data0', shape=(i_shape), dtype=dtype)

    # cv22
    begin0 = relay.annotation.compiler_begin(data0, compiler)
    node0  = relay.multiply(begin0, begin0)
    final  = relay.annotation.compiler_end(node0, compiler)
    # arm
    out    = relay.add(final, final)

    func = relay.Function([data0], out)
    mod = tvm.IRModule.from_expr(func)

    json_fname, lib_fname, params_fname = compile_model(mod, None, compiler=compiler, output_basename='compiled')

    # test runtime
    inputs_map = {"data0": data0_arr}
    output = run_model(inputs_map, json_fname, lib_fname, params_fname)[0]

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def test_classification():
    # TODO: Debug errors. ## == TVM ERROR in parsing. # == Some error in onnx or partitioning

    model_list = {
        'ResNet18_v1' : 'ResNet18_v1',
        'MobileNet1.0' : 'MobileNet1',
        'VGG11' : 'VGG11',
        'SqueezeNet1.0' : 'SqueezeNet1.0',
        'DenseNet121' : 'DenseNet121',
        'AlexNet' : 'AlexNet',
        #'InceptionV3' : 'InceptionV3',
    }

    dtype = "float32"
    i_shape = (1,3,224,224)

    data = np.random.randint(0, 255, i_shape).astype(dtype)
    #data = np.fromfile('Penguins_fp32_preproc_224.bin', count=-1, dtype=dtype)
    #data = np.reshape(data, i_shape)

    for model_name in model_list:
        print(model_name)
        model = model_zoo.get_model(model_name, pretrained=True)
        mod, params = relay.frontend.from_mxnet(model, {'data':i_shape})

        json_fname, lib_fname, params_fname = compile_model(mod, params, compiler='cv22', output_basename='compiled')

        # test runtime
        inputs_map = {"data": data}
        scores = run_model(inputs_map, json_fname, lib_fname, params_fname)[0]

        norm = softmax(scores[0,:])
        idx  = np.argsort(norm)[::-1]
        print()
        print(model_name)
        print('Top 5 categories: %s' % idx[:5])
        print('Top 5 scores: %s' % norm[idx[:5]])
        #scores.tofile(model_name+'_output.bin')

def test_object_detection():

    from tvm.contrib.download import download_testdata
    im_fname = download_testdata('https://github.com/dmlc/web-data/blob/master/' +
                                 'gluoncv/detection/street_small.jpg?raw=true',
                                 'street_small.jpg', module='data')

    image_size = 512
    input_shape = (1, 3, image_size, image_size)
    dtype = 'float32'
    x, img = data.transforms.presets.ssd.load_test(im_fname, short=image_size)

    model_list = ['ssd_512_mobilenet1.0_voc', 'ssd_512_resnet50_v1_voc']

    for model in model_list:
        print(model)

        block = model_zoo.get_model(model, pretrained=True)
        block.hybridize()
        block.forward(x)
        block.export('temp')

        input_name = 'data'
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

        json_fname, lib_fname, params_fname = compile_model(mod, params, compiler='cv22', output_basename='compiled')

        # test runtime
        inputs_map = {"data": x.asnumpy()}
        class_IDs, scores, bounding_boxes = run_model(inputs_map, json_fname, lib_fname, params_fname)

        # draw bbox
        ax = utils.viz.plot_bbox(img, bounding_boxes[0], scores[0], class_IDs[0], class_names=block.classes)
        out_img_fname = model + '_output.jpg'
        plt.savefig(out_img_fname)

if __name__ == '__main__':
    #test_manual()
    #test_classification()
    test_object_detection()
