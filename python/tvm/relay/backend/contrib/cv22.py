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
from os.path import join, isdir, isfile
import sys
import subprocess
import numpy as np
from enum import Enum
from queue import Queue
from shutil import copy
from onnx import save as onnx_save

# tvm imports
import tvm
from tvm import relay
from tvm.relay.expr_functor import ExprMutator
from tvm.relay.expr import GlobalVar

# import cvflowbackend package
cvb_path = subprocess.check_output(['tv2', '-basepath', 'cvflowbackend'])
cvb_path = cvb_path.decode().rstrip('\n')
cvb_path = join(cvb_path, 'lib')
if cvb_path not in sys.path:
    sys.path.append(cvb_path)
else:
    raise Exception('%s not found' % cvb_path)

# cvflow imports
import cvflowbackend
from cvflowbackend.ir_utils import ir_helper
from cvflowbackend.prepare_stage import schema

from frameworklibs.onnx import onnx_graph_utils as OnnxGraphUtils


class VarReplacer(ExprMutator):

    def __init__(self, var_map):
        ExprMutator.__init__(self)
        self.var_map = var_map

    def visit_var(self, var):
        if var in self.var_map:
            return self.var_map[var]
        return super().visit_var(var)


def get_first_subgraph(mod):
    temp_strmod = mod['main'].__str__()
    callindex = temp_strmod.find('@cv22')
    callendindex = temp_strmod.find('(', callindex)
    name = temp_strmod[callindex+1:callendindex]
    return name

    
def PruneSubgraphs(mod, prune_first=False):
    """
    Removes invalid subgraphs and those with no multiply-accumulates (if remove_no_max_subgraphs
    is set).
    """

    class SubgraphRemover(ExprMutator):
        """
        Reverts subgraphs in subgraphs_to_remove back to TVM instead of using an external codegen.
        """

        def __init__(self, subgraphs_to_remove, mod, new_mod):
            ExprMutator.__init__(self)
            self.subgraphs_to_remove = subgraphs_to_remove
            self.mod = mod
            self.new_mod = new_mod

        def visit_call(self, call):
            if isinstance(call.op, GlobalVar):
                name = call.op.name_hint
                if name in self.subgraphs_to_remove:
                    # "Inline" the subgraph back into new main function.
                    func = self.mod[name]
                    var_map = {}
                    for arg, param in zip(call.args, func.params):
                        var_map[param] = super().visit(arg)
                    new_body = relay.bind(func.body, var_map)
                    return new_body
                if name != "main":
                    args = []
                    for arg in call.args:
                        args.append(super().visit(arg))
                    return call.op(*args)
            return super().visit_call(call)

    subgraphs_with_macs = []
    # Remove invalid subgraphs

    print('in prune subs')
    print(mod.get_global_vars())
    
    for subgraph in mod.get_global_vars():
        name = subgraph.name_hint

        if (not mod[name].attrs) or (mod[name].attrs["Compiler"] != "cv22") or ("cv22" not in name):
            continue
        else:
            num_macs = relay.analysis.get_total_mac_number(mod[name])
            subgraphs_with_macs.append([name, num_macs])

    if prune_first:
        first_subgraph = get_first_subgraph(mod)
        subgraphs_names_to_remove = {x[0] for x in subgraphs_with_macs}
        print(type(subgraphs_names_to_remove))
        subgraph_to_keep = subgraphs_names_to_remove.remove(first_subgraph)
        print(subgraphs_names_to_remove)
        print(subgraph_to_keep)
    else:
        subgraphs_with_macs = sorted(subgraphs_with_macs, key=lambda x: int(x[1]))
        subgraphs_to_remove = subgraphs_with_macs[:-1]
        subgraphs_names_to_remove = {x[0] for x in subgraphs_to_remove}
    
    # Create new pruned module
    new_mod = tvm.IRModule(mod.functions, mod.type_definitions)
    new_mod["main"] = SubgraphRemover(subgraphs_names_to_remove, mod, new_mod).visit(mod["main"])
    return new_mod


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


def PartitionsToModules(mod, compiler):
        module_dict = {}
    
        for func in mod.get_global_vars():
            try:
                name = func.name_hint
                name = str(name)
                                
                #if (not mod[name].attrs) or (mod[name].attrs["Compiler"] == compiler)) and (not mod[name].attrs["Inline"]):
                if (not mod[name].attrs) or (mod[name].attrs["Compiler"] == compiler):
                    if compiler in name:                        
                        tempmod = tvm.IRModule.from_expr(mod[name])
                        new_mod = tvm.ir.module.IRModule()
                        new_mod['main'] = tempmod[name]
                        module_dict[name] = new_mod
                        
            except Exception as e:
                print(e)
                        
        return module_dict


def PartitionOneToModule(mod, compiler):
    module_dict = {}
    
    temp_strmod = mod['main'].__str__()
    callindex = temp_strmod.find('@cv22')
    callendindex = temp_strmod.find('(', callindex)

    name = temp_strmod[callindex+1:callendindex]
    tempmod = tvm.IRModule.from_expr(mod[name])
    new_mod = tvm.ir.module.IRModule()
    new_mod['main'] = tempmod[name]
    module_dict[name] = new_mod
        
    return module_dict


class tags(Enum):
    SHAPE  = 'shape'
    FPATH  = 'filepath'
    FILE   = 'file'
    DTYPE  = 'dtype'
    EXTN   = 'extn'
    CFMT   = 'colorformat'
    MEAN   = 'mean'
    SCALE  = 'scale'
    SDK    = 'sdk'


def set_env_variable(key, value):
    os.environ[key] = value


def CvflowCompilation(model_proto, output_name, output_folder, metadata, input_config, sdk, ambalink_cfg={}):


    def run_command(cmd):
        """Run command, return output as string."""
        output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
        return output.decode("ascii")


    def create_metablock(name, block_type, inputs, outputs, attrs={}):
        return schema.Block(name, block_type, inputs, outputs, attrs)


    def create_metatensor(name, shape=None, dtype=None, data=None, extn=None, init=None):
        meta_tensor = schema.Tensor(
            name=name,
            shape=shape,
            dtype=dtype,
            data=data,
            extn=extn,
            init=init
        )

        return meta_tensor


    def create_input_tensor(input_tensor_name, shape, dtype, dra_file, extn, first_node):
        if first_node:
            meta_tensor = create_metatensor(
                name=input_tensor_name,
                shape=shape,
                dtype=dtype,
                data=dra_file,
                extn=extn
            )

        else:
            meta_tensor = create_metatensor(name=input_tensor_name)

        return meta_tensor


    def flatten_if_needed(const_arr, orig_inp_sh, flattened_sh):

        if orig_inp_sh == flattened_sh:
            return const_arr

        # broadcast to original shape
        const_arr = np.broadcast_to(const_arr, orig_inp_sh)

        # flatten
        const_arr_flat = const_arr.flatten()

        return const_arr_flat


    def transpose_if_needed(const_arr, flattened_sh):

        # (TBD): remove this once schema validator is added
        if len(const_arr.shape) != 1:
            raise ValueError("Unsupported mean/scale. Only 1D is supported, got %s" % const_arr.shape)

        num_elems = const_arr.shape[0]
        if num_elems == 1:
            return const_arr

        unhandled_case = False

        if len(flattened_sh) == 1:
            pass

        elif len(flattened_sh) == 2:
            unhandled_case = True

        elif len(flattened_sh) == 3:
            if flattened_sh[0] == num_elems:
                # matches depth
                const_arr = np.reshape(const_arr, (num_elems,1,1))
            elif flattened_sh[2] == num_elems:
                pass
            else:
                unhandled_case = True

        elif len(flattened_sh) == 4:
            if flattened_sh[1] == num_elems:
                # matches depth
                const_arr = np.reshape(const_arr, (1,num_elems,1,1))
            elif flattened_sh[3] == num_elems:
                pass
            else:
                unhandled_case = True

        if unhandled_case:
            raise ValueError(
                "Unsupported case in transpose_if_needed: flattened_sh (%s), const_arr sh (%s)" %
                (flattened_sh, const_arr.shape)
            )

        return const_arr


    def create_const_tensor(const_tensor_name, const_list, orig_inp_sh, flattened_sh, output_folder):

        updated_const_arr = flatten_if_needed(np.array(const_list), orig_inp_sh, flattened_sh).astype(np.float32)
        updated_const_arr = transpose_if_needed(updated_const_arr, flattened_sh)
        shape = updated_const_arr.shape

        const_fname = join(output_folder, const_tensor_name + ".bin")
        updated_const_arr.tofile(const_fname)

        const_meta_tensor = create_metatensor(
            name=const_tensor_name,
            shape=shape,
            dtype="float32",
            init=const_fname
        )

        return const_meta_tensor


    def create_preproc_node(sgl_type, const_list, tensor_names_q, inp_cfg, pr_inp_shape, output_folder, first_node):

        # input tensor names
        input_tensor_name = tensor_names_q.get()
        const_tensor_name = tensor_names_q.get()

        # output tensor name
        # do not pop since it will be used as the input of next node
        output_tensor_name = tensor_names_q.queue[0]

        # input tensor
        input_meta_tensor = create_input_tensor(
            input_tensor_name,
            pr_inp_shape,
            inp_cfg[tags.DTYPE.value],
            inp_cfg[tags.FPATH.value],
            inp_cfg[tags.EXTN.value],
            first_node
        )
        first_node = False

        # const tensor
        const_meta_tensor = create_const_tensor(
            const_tensor_name,
            const_list,
            inp_cfg[tags.SHAPE.value],
            pr_inp_shape,
            output_folder
        )

        # output tensor
        output_meta_tensor = create_metatensor(name=output_tensor_name)

        pp_node = create_metablock(
            output_tensor_name,
            sgl_type,
            [input_meta_tensor, const_meta_tensor],
            [output_meta_tensor],
        )

        return pp_node, first_node


    def create_vp_node(node_name, tensor_names_q, inp_cfg, pr_inp_shape, first_node, gs_recs):

        # NOTE: only single input is currently supported
        input_tensor_name = tensor_names_q.get()

        input_meta_tensor = create_input_tensor(
            input_tensor_name,
            pr_inp_shape,
            inp_cfg[tags.DTYPE.value],
            inp_cfg[tags.FPATH.value],
            inp_cfg[tags.EXTN.value],
            first_node
        )

        outputs = []
        while not tensor_names_q.empty():
            outputs.append(create_metatensor(name=tensor_names_q.get(), dtype='float32'))

        attr_dict = {}
        attr_dict['cnngen_flags'] = ''

        graph_surgery_transforms = []
        if gs_recs['MOD_NODE_NAMES']:
            graph_surgery_transforms.append('ModNodeNames')
        if gs_recs['FOLD_CONSTANTS']:
            graph_surgery_transforms.append('FoldConstants')
        attr_dict['graph_surgery_transforms'] = ','.join(graph_surgery_transforms)

        vp_node = create_metablock(
            node_name,
            'VP',
            [input_meta_tensor],
            outputs,
            attr_dict
        )

        return vp_node


    def create_graph_desc(vp_name, input_config, primary_inputs, primary_outputs, input_preproc_mapping, gs_recs, output_folder):

        # working on one input

        # remap input_config keys to that in the relay -> onnx converted tensor names
        pr_inp = list(primary_inputs.keys())[0]
        cf_inp = list(input_config.keys())[0]

        inp_cfg = input_config[cf_inp]

        mangled_name = pr_inp
        orig_name = input_preproc_mapping[pr_inp]

        # create meta blocks for mean, scale and VP

        # boolean flag to identify first node
        # attrs like dra file, ... will be added only to the first node
        first_node = True

        # tensor name queue: tensor names have to be adjusted based on presence of mean and scale preproc ops
        # no preproc:
        #   score = VP(data)
        # mean:
        #   data_mean = Sub(data, mean_const)
        #   score = VP(data_mean)
        # mean, scale:
        #   data_mean = Sub(data, mean_const)
        #   data_scale = Div(data_mean, scale_const)
        #   score = VP(data_scale)
        tensor_names = Queue()

        tensor_names.put(orig_name)

        if tags.MEAN.value in input_config[cf_inp]:
            tensor_names.put(orig_name + '_mean_const')

            # out tensor name depends on scale
            if tags.SCALE.value in input_config[cf_inp]:
                tensor_names.put(orig_name + '_mean')

        if tags.SCALE.value in input_config[cf_inp]:
            tensor_names.put(orig_name + '_scale_const')

        if (tags.MEAN.value in input_config[cf_inp]) or (tags.SCALE.value in input_config[cf_inp]):
            tensor_names.put(mangled_name)

        for o in primary_outputs:
            tensor_names.put(o)

        graph_desc = []

        if tags.MEAN.value in input_config[cf_inp]:
            mean_node, first_node = create_preproc_node(
                'SGL::Sub',
                inp_cfg[tags.MEAN.value],
                tensor_names,
                inp_cfg,
                primary_inputs[pr_inp],
                output_folder,
                first_node
            )
            graph_desc.append(mean_node)

        if tags.SCALE.value in input_config[cf_inp]:
            scale_node, first_node = create_preproc_node(
                'SGL::Div',
                inp_cfg[tags.SCALE.value],
                tensor_names,
                inp_cfg,
                primary_inputs[pr_inp],
                output_folder,
                first_node
            )
            graph_desc.append(scale_node)

        vp_node = create_vp_node(
            vp_name,
            tensor_names,
            inp_cfg,
            primary_inputs[pr_inp],
            first_node,
            gs_recs
        )
        graph_desc.append(vp_node)

        return graph_desc


    def import_graph_surgery():

        gs_path = run_command('tv2 -basepath frameworklibs').strip()
        sys.path.append(join(gs_path, 'lib/python3.7/site-packages/frameworklibs/onnx/'))

    # flatten i/o using graph surgery
    def flatten_io(in_model_proto):

        from onnx_transform import OnnxGraphTransform
        gs = OnnxGraphTransform(model=in_model_proto)
        out_model_proto = gs.apply_transforms(transforms='FlattenIO')

        return out_model_proto


    def rename_input_tensors(model_proto, input_config):

        # get graph info
        model_summary = OnnxGraphUtils.determine_graph_info(model_proto, None)[0]
        primary_inputs = model_summary['PRIMARY_INPUTS']

        if len(primary_inputs) != 1:
            raise ValueError('[CvflowCompilation] unsupported case: only single input is supported')

        pr_inp = list(primary_inputs.keys())[0]
        cf_inp = list(input_config.keys())[0]

        # maps preproc name to input
        input_preproc_mapping = {}
        if (tags.MEAN.value in input_config[cf_inp]) or (tags.SCALE.value in input_config[cf_inp]):
            input_preproc_mapping[pr_inp+'_preproc'] = pr_inp

        if input_preproc_mapping:
            transform_str = 'RenameTensors('

            for i in input_preproc_mapping:
                transform_str += input_preproc_mapping[i]
                transform_str += '='
                transform_str += i
                transform_str += ','

            transform_str += ')'

            from onnx_transform import OnnxGraphTransform
            gs = OnnxGraphTransform(model=model_proto)
            model_proto = gs.apply_transforms(transforms=transform_str)

        # (TBD): this is done to avoid conditional code downstream
        # (TBD): implement a better logic as this won't work for multi inputs case
        else:
            input_preproc_mapping[pr_inp] = pr_inp

        return model_proto, input_preproc_mapping


    def run_graph_surgery(model_proto, input_config):

        import_graph_surgery()

        # flatten i/o using graph surgery
        model_proto = flatten_io(model_proto)

        # rename input tensors if there are any preproc operations
        # this is done to ensure that primary input name remains the same even with preproc
        model_proto, input_preproc_mapping = rename_input_tensors(model_proto, input_config)

        return model_proto, input_preproc_mapping


    def gen_cavalry_bin(vas_outf, cavalry_bin_fname):

        cavalry_cmd = 'cavalry_gen -d ' + vas_outf + ' -f ' + cavalry_bin_fname
        os.system(cavalry_cmd)

        # check if bin was generated
        if not isfile(cavalry_bin_fname):
            raise ValueError('Cavalry bin (%s) not generated' % cavalry_bin_fname)


    def get_flexi_bin(cnngen_outf, outf, out_name, inputs, outputs, dummy_input_file, fwbase, diag_dir):

        # SUPERDAG GEN

        ## import superdag_gen
        tv2_p = subprocess.check_output(['tv2', '-basepath', 'gentask'])
        tv2_p = tv2_p.decode().rstrip('\n')
        if isdir(tv2_p):
            if tv2_p not in sys.path:
                sys.path.append(tv2_p)
        else:
            raise Exception('%s not found' % tv2_p)

        from superdag_gen import main as sdag_gen_entry_pt

        ## superdag gen args
        args = {}
        args['--name'] = out_name
        args['--cvtask'] = out_name
        args['--inputs'] = [i+'='+dummy_input_file for i in inputs]
        args['--outputs'] = [o+'='+o+'.out' for o in outputs]
        args['--vasdir'] = cnngen_outf
        args['--fwbase'] = fwbase
        args['--handle_opdirs'] = 'overwrite'
        args['--rundir'] = join(outf, out_name + '_flexidag')
        args['--codeonly'] = True
        args['--board'] = True
        args['--inputmode'] = 5
        args['--os'] = 1
        args['--yield'] = 'on'

        cmd = []
        for k,v in args.items():
            if isinstance(v, list):
                for vv in v:
                    cmd.extend([k,vv])

            elif isinstance(v, bool):
                if v:
                    cmd.append(k)

            else: # str
                cmd.extend([k,str(v)])

        ## run superdag_gen.py
        print('\nSuperdag gen args: %s' % cmd)
        sdag_gen_entry_pt(cmd)

        # REMOTECONFIG

        ## remember cwd
        cwd = os.getcwd()

        ## change to diag dir
        os.chdir(diag_dir)

        ## call remoteconfig
        cmd_str = 'remoteconfig ' + join(fwbase, 'tests', out_name+'_ag')
        print('\nRemoteconfig cmd line: %s' % cmd_str)
        os.system(cmd_str)

        # BUILD
        os.system('make build')

        ## check if bin was generated
        flexi_bin_fname = join(diag_dir, 'flexidag0', 'flexibin0.bin')
        if not isfile(flexi_bin_fname):
            raise ValueError('Flexi bin (%s) not generated' % flexi_bin_fname)

        ## restore cwd
        os.chdir(cwd)

        return flexi_bin_fname


    ###############################################################################################


    # run graph surgery transforms like FlattenIO, RenameTensors
    model_proto, input_preproc_mapping = run_graph_surgery(model_proto, input_config)
    onnx_save(model_proto, join(output_folder, output_name+'_modified.onnx'))

    # get graph info
    model_summary, gs_recs, _ = OnnxGraphUtils.determine_graph_info(model_proto, None)
    primary_inputs  = model_summary['PRIMARY_INPUTS']
    primary_outputs = model_summary['PRIMARY_OUTPUTS']

    # update metadata (non service case) with the correct input and output info
    if metadata:
        in_dtype = metadata['Inputs'][0]['dtype']
        metadata['Inputs'] = []
        for i,sh in primary_inputs.items():
            metadata['Inputs'].append({'name':i, 'shape':sh, 'dtype':in_dtype})

        '''
        out_dtype = metadata['Outputs'][0]['dtype']
        metadata['Outputs'] = []
        for o,sh in primary_outputs.items():
            metadata['Outputs'].append({'name':o, 'shape':sh, 'dtype':out_dtype})
        '''

    # set outputs list as env variable
    # this will be used by codegen
    pr_outputs_list = list(primary_outputs.keys())
    set_env_variable('CV22_OUTPUTS_LIST', ','.join(pr_outputs_list))

    # create splits json
    graph_desc = create_graph_desc(
        output_name,
        input_config,
        primary_inputs,
        primary_outputs,
        input_preproc_mapping,
        gs_recs,
        output_folder
    )
    schema.dump_json(graph_desc, join(output_folder, output_name+'_splits.json'))
    print('Saved splits json: %s' % join(output_folder, output_name+'_splits.json'))

    prepare_outf = join(output_folder, 'prepare')
    ckpt_ambapb = cvflowbackend.prepare(
        model=model_proto.SerializeToString(),
        graph_desc=graph_desc,
        framework='onnx',
        metagraph_type='fast_checkpoint',
        output_name=output_name,
        output_folder=prepare_outf,
        log_dir=join(output_folder, 'logs')
    )

    # roundabout way - convert fast checkpoint to checkpoint
    # this generates vas artifacts which can be used to generate 
    # cavalry bin / flexidag
    cvflowb_outf = join(output_folder, 'convert')
    ckpt_ambapb = cvflowbackend.convert(
        ckpt_ambapb.SerializeToString(),
        metagraph_type='checkpoint',
        vas_flags='-auto',
        output_name=output_name,
        output_folder=cvflowb_outf
    )

    ambapb_fpath = ir_helper.save_model(ckpt_ambapb, output_name, output_folder)

    cnngen_outf = join(cvflowb_outf, 'cnngen_out', output_name)
    vas_outf = join(cnngen_outf, 'vas_output')
    if not isdir(vas_outf):
        raise ValueError('Vas output folder (%s) not found' % vas_outf)

    sdk_bin_fpath = join(output_folder, output_name+'.amba')

    # generate cavalry bin
    if sdk == 'linux':
        gen_cavalry_bin(vas_outf, sdk_bin_fpath)

    # generate flexibin
    else: # sdk == 'ambalink'
        dummy_input_file = ambalink_cfg['sdag_input_file']
        fwbase = ambalink_cfg['firmware_base']
        diag_dir = ambalink_cfg['diag_dir']

        flexibin = get_flexi_bin(
            cnngen_outf,
            output_folder,
            output_name,
            list(input_preproc_mapping.values()),
            list(primary_outputs.keys()),
            dummy_input_file,
            fwbase,
            diag_dir
        )

        # copy it right location (expected by test_cv22.py)
        copy(flexibin, sdk_bin_fpath)

    return ambapb_fpath, sdk_bin_fpath


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

        def relayvm_build(self, mod, params, opt_level=3):
            with tvm.transform.PassContext(opt_level=opt_level, disabled_pass=["FoldScaleAxis", "AlterOpLayout"]):
                vm_exec = relay.vm.compile(mod, target="llvm -device=arm_cpu -mtriple=aarch64-linux-gnu", params=params)
                #vm_exec = relay.vm.compile(mod, target="llvm", params=params)
                self._json, self._lib = vm_exec.save()

        def relay_build(self, mod, params, opt_level=3):

                with relay.build_config(opt_level=opt_level):#, disabled_pass=["AlterOpLayout"]):

                        if self._mode == 'EMULATOR':
                            self._json, self._lib, self._params =\
                                        relay.build(mod, target='llvm', params=params)
                            self._logger.info('[CVFlowTVMWrapper] info: relay build for emulator complete')

                        else:
                            self._json, self._lib, self._params = \
                                        relay.build(mod, target='llvm -device=arm_cpu -mtriple=aarch64-linux-gnu', params=params)
                            self._logger.info('[CVFlowTVMWrapper] info: relay build for target complete')

        def serialize(self, basename='compiled'):

                self._json_fname   = basename + '.json'
                self._lib_fname    = basename + '.so'

                with open(self._json_fname, 'w') as f_graph_json:
                        f_graph_json.write(self._json)

                self._params_fname = None
                if self._params is not None:
                        self._params_fname = basename + '.params'
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

                        rt_mod = tvm.contrib.graph_executor.create(json, lib, tvm.cpu())
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
            libpath = subprocess.check_output(['tv2', '-basepath', 'AmbaCnnUtils'])
            libpath = libpath.decode().rstrip('\n')
            tv2_p = join(libpath, 'parser/common/')
            if isdir(tv2_p):
                if tv2_p not in sys.path:
                    sys.path.append(tv2_p)
            else:
                raise Exception('%s not found' % tv2_p)

            from logger import ModifiedABSLLogger
            log = ModifiedABSLLogger(program_name="CVFlowTVMWrapper", amba_verbosity_level=debuglevel)

            return log
