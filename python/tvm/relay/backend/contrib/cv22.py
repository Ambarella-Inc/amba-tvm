# licensed to the apache software foundation (asf) under one
# or more contributor license agreements.  see the notice file
# distributed with this work for additional information
# regarding copyright ownership.  the asf licenses this file
# to you under the apache license, version 2.0 (the
# "license"); you may not use this file except in compliance
# with the license.  you may obtain a copy of the license at
#
#   http://www.apache.org/licenses/license-2.0
#
# unless required by applicable law or agreed to in writing,
# software distributed under the license is distributed on an
# "as is" basis, without warranties or conditions of any
# kind, either express or implied.  see the license for the
# specific language governing permissions and limitations
# under the license.
"""cvflow compilation code"""

import os
import sys
import subprocess
import onnx
import copy
import numpy as np
import json
import ctypes
import re
from enum import Enum

import tvm
from tvm import relay
from tvm import topi
from tvm.topi.util import get_const_tuple

from tvm.relay.expr_functor import ExprMutator, ExprVisitor
from tvm.relay.expr import Call, Constant, Tuple, GlobalVar
from tvm.ir import Op
from tvm.relay.function import Function
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
from tvm.contrib import graph_runtime

# import cvtools package
cnn_utils_path = subprocess.check_output(['tv2', '-basepath', 'CnnUtils'])
cnn_utils_path = cnn_utils_path.decode().rstrip('\n')

tv2_p = cnn_utils_path + '/packages/'
if tv2_p not in sys.path:
    sys.path.append(tv2_p)
else:
    raise Exception('%s not found' % tv2_p)

tv2_p = cnn_utils_path + '/onnx/common'
if tv2_p not in sys.path:
    sys.path.append(tv2_p)
else:
    raise Exception('%s not found' % tv2_p)

import cvflow_backend
from cvflow_backend.ir_utils import ir_helper
import onnx_graph_utils as OnnxGraphUtils

class VarReplacer(ExprMutator):
    def __init__(self, var_map):
        ExprMutator.__init__(self)
        self.var_map = var_map

    def visit_var(self, var):
        if var in self.var_map:
            return self.var_map[var]
        return super().visit_var(var)

class VarRenamer(ExprMutator):
    def __init__(self, new_subgraph_name):
        ExprMutator.__init__(self)
        self.new_subgraph_name = new_subgraph_name

    def visit_var(self, var):
        if "_".join(var.name_hint.split('_')[:2]) != self.new_subgraph_name:
            new_var_name = self.new_subgraph_name + "_" + var.name_hint.split('_')[2]
            return relay.Var(new_var_name, var.checked_type)
        return super().visit_var(var)

class SubgraphRemover(ExprMutator):
    def __init__(self, subgraphs_to_remove, mod, new_mod, rename_starting_from_0=True):
        ExprVisitor.__init__(self)
        self.subgraphs_to_remove = subgraphs_to_remove
        self.mod = mod
        self.new_mod = new_mod
        self.rename_starting_from_0 = rename_starting_from_0
        self.count = 0

    def visit_call(self, call):
        if isinstance(call.op, GlobalVar):
            name = call.op.name_hint
            if name in self.subgraphs_to_remove:
                # "Inline" the subgraph back into new main function.
                func = self.mod[name]
                var_map = {}
                for arg, param in zip(call.args, func.params):
                    var_map[param] = super().visit(arg)
                new_body = VarReplacer(var_map).visit(func.body)
                return new_body
            elif name != "main":
                # Copy the GlobalVar (subgraph function) to the new module and call.
                if self.rename_starting_from_0:
                    new_name = name.split('_')[0] + "_" + str(self.count)
                    self.count += 1
                else:
                    new_name = name
                args = []
                for arg in call.args:
                    args.append(super().visit(arg))
                subgraph_gv = relay.GlobalVar(new_name)
                if self.rename_starting_from_0:
                    subgraph_func = VarRenamer(new_name).visit(self.mod[name])
                    subgraph_func = subgraph_func.with_attr("global_symbol", new_name)
                    self.new_mod[subgraph_gv] = subgraph_func
                else:
                    self.new_mod[subgraph_gv] = self.mod[name]
                return subgraph_gv(*args)
        return super().visit_call(call)

'''
def PruneSubgraphsWithMoreThanOneInput(mod, compiler="cv22"):
    subgraph_names_to_remove = []
    # Remove subgraphs with more than 1 input or tuple inputs.
    for subgraph in mod.get_global_vars():
        name = subgraph.name_hint
        if not mod[name].attrs or mod[name].attrs["Compiler"] != compiler:
            continue
        logging.debug("Sugraph params: %s" % mod[name].params)
        if len(mod[name].params) != 1 or isinstance(mod[name].params[0].checked_type, relay.TupleType):
            subgraph_names_to_remove.append(name)
    logging.debug("Removing subgraphs due to having more than one input: %s" % subgraph_names_to_remove)
    new_mod = tvm.IRModule()
    new_mod["main"] = SubgraphRemover(subgraph_names_to_remove, mod, new_mod).visit(mod["main"])
    return new_mod
'''

def PruneSubgraphs(mod, compiler, num_subgraphs_to_keep, logger=None):
    subgraph_with_macs = []
    for subgraph in mod.get_global_vars():
        name = subgraph.name_hint
        if not mod[name].attrs or mod[name].attrs["Compiler"] != compiler:
            continue
        num_macs = relay.analysis.get_total_mac_number(mod[name])
        subgraph_with_macs.append([name, num_macs])
    subgraph_with_macs = sorted(subgraph_with_macs, key=lambda x: int(x[1]))

    if logger is not None:
        logger.debug("[PruneSubgraphs] Subgraphs with computed # of MACS: %s" % subgraph_with_macs)
    else:
        print("[PruneSubgraphs] Subgraphs with computed # of MACS: %s" % subgraph_with_macs)

    subgraphs_to_remove = subgraph_with_macs[:-num_subgraphs_to_keep]
    '''
    subgraphs_to_remove = []
    for s in subgraph_with_macs:
        if not s[0] == 'cv22_0':
            subgraphs_to_remove.append(s)
    '''

    if logger is not None:
        logger.debug("[PruneSubgraphs] Will remove these subgraphs: %s" % subgraphs_to_remove)
    else:
        print("[PruneSubgraphs] Will remove these subgraphs: %s" % subgraphs_to_remove)

    subgraph_names_to_remove = set([x[0] for x in subgraphs_to_remove])
    # Create new pruned module
    new_mod = tvm.IRModule()
    new_mod["main"] = SubgraphRemover(subgraph_names_to_remove, mod, new_mod).visit(mod["main"])
    return new_mod

def PartitionsToModules(mod, compiler):
        module_dict = {}
        for func in mod.get_global_vars():
                name = func.name_hint
                if compiler in name:
                        mod = tvm.IRModule.from_expr(mod[name])
                        new_mod = tvm.ir.module.IRModule()
                        new_mod['main'] = mod[name]
                        module_dict[name] = new_mod

        return module_dict

class tags(Enum):
    SHAPE = 'shape'
    FPATH = 'filepath'
    FILE  = 'file'
    DTYPE = 'dtype'
    EXTN  = 'extn'

def create_rand_dra(primary_inputs, output_folder):
    '''
    Create DRA files with random float32 values
    Inputs:
    primary_inputs: {input tensor name: shape}
    output_folder:  folder to store dra files in
    Outputs:
    dra_dict: {input tensor name: (shape, dra bin filename)}
    '''

    dra_dict = {}
    for i in primary_inputs:
        ishape = primary_inputs[i]

        data = np.random.randint(0, 100, ishape).astype(np.float32)

        fname = output_folder + i + '_dra.bin'
        data.tofile(fname)

        dra_fname = output_folder + i + '_dra.txt'
        with open(dra_fname, 'w') as dra_file:
            dra_file.write(fname)

        dra_dict[i] = {
                       tags.SHAPE.value: ishape,
                       tags.FPATH.value: dra_fname
                      }

    return dra_dict

def create_splits_json(dra_dict, primary_outputs, vp_name, output_folder, gs_recs):
    '''
    Create splits json for cvflow backend prepare
    Inputs:
    dra_dict: {input tensor name: {shape:shape, filepath:dra bin filename} }
    vp_name:  vp subgraph name
    primary_outputs: {output tensor name: shape}
    Outputs:
    splits_json_dict: See spec
    '''

    begin_dict = {}
    for i in dra_dict:
        inp_dict = {}
        inp_dict[tags.SHAPE.value] = dra_dict[i][tags.SHAPE.value]
        inp_dict[tags.DTYPE.value] = 'float32'
        inp_dict[tags.FILE.value]  = dra_dict[i][tags.FPATH.value]
        inp_dict[tags.EXTN.value]  = 'bin'

        begin_dict[i] = inp_dict.copy()

    end_dict = {}
    for o in primary_outputs:
        end_dict[o] = {tags.DTYPE.value: 'float32'}

    attr_dict = {}
    #attr_dict['cnngen_flags'] = '-c coeff-force-fx16,act-force-fx16'
    attr_dict['vas_flags'] = '-v'

    graph_surgery_transforms = []
    if gs_recs['MOD_NODE_NAMES']:
        graph_surgery_transforms.append('ModNodeNames')
    if gs_recs['FOLD_CONSTANTS']:
        graph_surgery_transforms.append('FoldConstants')
    attr_dict['graph_surgery_transforms'] = ','.join(graph_surgery_transforms)

    vp_dict = {}
    vp_dict['type']  = 'ORCVP'
    vp_dict['begin'] = begin_dict
    vp_dict['end']   = end_dict
    vp_dict['attr']  = attr_dict

    splits_json = {}
    splits_json[vp_name] = copy.deepcopy(vp_dict)

    splits_json_fname = output_folder + vp_name + '_splits.json'
    with open(splits_json_fname, 'w') as json_file:
        json.dump(splits_json, json_file, indent=1)

    return splits_json_fname

def set_env_variable(key, value):
    os.environ[key] = value

def CvflowCompilation(model_proto, output_name, output_folder, metadata, input_config=None):

    if not output_folder.endswith('/'):
         output_folder += '/'

    # get graph info
    model_summary, gs_recs = OnnxGraphUtils.determine_graph_info(model_proto, None)

    primary_inputs  = model_summary['PRIMARY_INPUTS']
    primary_outputs = model_summary['PRIMARY_OUTPUTS']

    # (DEBUG) use random images for DRA if not provided by user
    if input_config is None:
        dra_dict = create_rand_dra(primary_inputs, output_folder)

    # remap input_config keys to that in the relay -> onnx converted tensor names
    # (TBD) map tensor names in original model (input_config) with those in relay -> onnx converted model (primary *puts)
    else:
        if len(primary_inputs) != 1:
            raise ValueError('[CvflowCompilation] unsupported case: only single input is supported')

        dra_dict = {}
        for i in primary_inputs:
            dra_dict[i] = {}
            dra_dict[i][tags.SHAPE.value] = primary_inputs[i]
            dra_dict[i][tags.FPATH.value] = input_config[list(input_config.keys())[0]][tags.FPATH.value]

    # create splits json file
    graphdesc_path = create_splits_json(dra_dict, primary_outputs, output_name, output_folder, gs_recs)

    # update metadata (non service case) with the correct input and output info
    if metadata:
        in_dtype = metadata['Model']['Inputs'][0]['dtype']
        metadata['Model']['Inputs'] = []
        for i,sh in primary_inputs.items():
            metadata['Model']['Inputs'].append({'name':i, 'shape':sh, 'dtype':in_dtype})

        out_dtype = metadata['Model']['Outputs'][0]['dtype']
        metadata['Model']['Outputs'] = []
        for o,sh in primary_outputs.items():
            metadata['Model']['Outputs'].append({'name':o, 'shape':sh, 'dtype':out_dtype})

    # set outputs list as env variable
    # this will be used by codegen
    pr_outputs_list = list(primary_outputs.keys())

    set_env_variable('CV22_OUTPUTS_LIST', ','.join(pr_outputs_list))

    graphdesc_bytes = None
    with open(graphdesc_path, mode='rb') as f:
        graphdesc_bytes = f.read()

    ckpt_ambapb = cvflow_backend.prepare(model_bytes=model_proto.SerializeToString(), \
                                         graph_desc_bytes=graphdesc_bytes, \
                                         framework='onnx', \
                                         metagraph_type='fast_checkpoint', \
                                         output_name=output_name, \
                                         output_folder=output_folder, \
                                         log_dir=output_folder+'/logs')

    save_path = ir_helper.save_model(ckpt_ambapb, \
                                     output_name, \
                                     output_folder)

    # generate cavalry bin

    # roundabout way - convert fast checkpoint to checkpoint
    # this generates vas artifacts which can be used to generate 
    # cavalry bin
    _output_folder = output_folder + 'cavalry/'
    cvflow_backend.prepare(model_bytes=ckpt_ambapb.SerializeToString(),
                           metagraph_type='checkpoint',
                           output_name=output_name,
                           output_folder=_output_folder)

    vas_out_dir = _output_folder + 'ambacnn_out/' + output_name + '/vas_output/'
    if not os.path.isdir(vas_out_dir):
        raise ValueError('Vas output folder (%s) not found' % vas_out_dir)

    cavalry_bin_fname = output_folder + output_name + '.amba'

    cavalry_cmd = 'cavalry_gen -d ' + vas_out_dir + ' -f ' + cavalry_bin_fname
    os.system(cavalry_cmd)

    # check if bin was generated
    if not os.path.isfile(cavalry_bin_fname):
        raise ValueError('Cavalry bin (%s) not generated' % cavalry_bin_fname)

    return save_path

def GetCvflowExecutionMode():
        env_var = os.environ.get('CV22_EMULATOR')

        if (env_var is None) or (env_var and int(env_var) == 0):
                mode = 'TARGET'
        else:
                mode = 'EMULATOR'

        return mode

class CVFlowTVMWrapper():

        # mode: ['EMULATOR', 'TARGET']
        def __init__(self, mode, logger=None):

                if mode not in ['EMULATOR', 'TARGET']:
                        raise ValueError('[CVFlowTVMWrapper] error: unknown mode (%s). Supported values: [EMULATOR, TARGET]' % mode)

                self._mode = mode

                if logger is None:
                    self._logger = self.init_logger(debuglevel=1)
                else:
                    self._logger = logger

                self._json   = None
                self._lib    = None
                self._params = None

                self._json_fname   = None
                self._lib_fname    = None
                self._params_fname = None

        def relay_build(self, mod, opt_level=3):

                with relay.build_config(opt_level=opt_level, disabled_pass=["AlterOpLayout"]):

                        if self._mode == 'EMULATOR':
                                self._json, self._lib, self._params =\
                                        relay.build(mod, target='llvm')
                                self._logger.info('[CVFlowTVMWrapper] info: relay build for emulator complete')

                        else:
                                self._json, self._lib, self._params = \
                                        relay.build(mod, target='llvm -device=arm_cpu -mtriple=aarch64-linux-gnu')
                                self._logger.info('[CVFlowTVMWrapper] info: relay build for target complete')

        def serialize(self, basename='compiled'):

                self._json_fname   = basename + '.json'
                self._lib_fname    = basename + '.so'
                self._params_fname = basename + '.params'

                with open(self._json_fname, 'w') as f_graph_json:
                        f_graph_json.write(self._json)

                with open(self._params_fname, 'wb') as f_params:
                        f_params.write(relay.save_param_dict(self._params))

                if self._mode == 'EMULATOR':
                        self._lib.export_library(self._lib_fname)
                        self._logger.info('[CVFlowTVMWrapper] info: serialize for emulator complete')

                else:
                        self._lib.export_library(self._lib_fname, \
                                                 cc='/usr/bin/aarch64-linux-gnu-g++', \
                                                 options=["-v"])
                        self._logger.info('[CVFlowTVMWrapper] info: serialize for target complete')

                return self._json_fname, self._lib_fname, self._params_fname

        def deserialize(self, json_fname=None, lib_fname=None, params_fname=None):

                if json_fname is None:
                    json_fname = self._json_name
                assert json_fname is not None, 'json_fname cannot be None. Serialize or provide json_fname arg'

                if lib_fname is None:
                    lib_fname = self._lib_name
                assert lib_fname is not None, 'lib_fname cannot be None. Serialize or provide lib_fname arg'

                if params_fname is None:
                    params_fname = self._params_name
                assert params_fname is not None, 'params_fname cannot be None. Serialize or provide params_fname arg'

                if self._mode == 'EMULATOR':
                        with open(json_fname, 'r') as f_graph_json:
                                self._json = f_graph_json.read()

                        with open(params_fname, 'rb') as f_params:
                                self._params = tvm.relay.load_param_dict(f_params.read())

                        self._lib = tvm.runtime.load_module(lib_fname)

                        self._logger.info('[CVFlowTVMWrapper] info: deserialize for emulator complete')

                else:
                        self._logger.info('[CVFlowTVMWrapper] info: nothing to be done for deserialize for target')

        def tvm_runtime(self, inputs):

                if self._mode == 'EMULATOR':

                        json   = self._json
                        lib    = self._lib
                        params = self._params

                        rt_mod = tvm.contrib.graph_runtime.create(json, lib, ctx=tvm.cpu())
                        rt_mod.set_input(**params)

                        cnt = 0
                        for name, data in inputs.items():
                            # get shape from graph runtime
                            # (TBD) input order assumed
                            tmp = rt_mod.get_input(cnt)
                            data = np.reshape(data, tmp.shape)

                            rt_mod.set_input(name, data)

                            cnt += 1

                        rt_mod.run()
                        num_outputs = rt_mod.get_num_outputs()

                        rt_outs = []
                        for i in range(num_outputs):
                                rt_outs.append(rt_mod.get_output(i).asnumpy())

                else:
                        rt_outs = None
                        self._logger.info('[CVFlowTVMWrapper] info: nothing to be done for tvm_runtime for target')

                return rt_outs

        def init_logger(self, debuglevel):

            import subprocess
            libpath = subprocess.check_output(['tv2', '-basepath', 'AmbaCnnUtils'])
            libpath = libpath.decode().rstrip('\n')
            tv2_p = os.path.join(libpath, 'parser/common/')
            if os.path.isdir(tv2_p):
                if tv2_p not in sys.path:
                    sys.path.append(tv2_p)
            else:
                raise Exception('%s not found' % tv2_p)

            from logger import ModifiedABSLLogger
            log = ModifiedABSLLogger(program_name="CVFlowTVMWrapper", amba_verbosity_level=debuglevel)

            return log
