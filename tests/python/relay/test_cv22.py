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

"""Entry script for cv22 compilation"""

import sys
from os import makedirs, listdir, environ, urandom
from os.path import exists, join, isdir, basename
from shutil import copy, rmtree
import binascii
import json
import logging
import tarfile
import time
import numpy as np
import subprocess
import argparse
from enum import Enum
logging.basicConfig(level=logging.DEBUG)

# tvm imports
import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name

import tvm.relay.op.contrib.cv22
from tvm.relay.backend.contrib.cv22 import tags as T
from tvm.relay.backend.contrib.cv22 import set_env_variable, PruneSubgraphs, PartitionsToModules, PartitionOneToModule, GetCvflowExecutionMode, CvflowCompilation, CVFlowTVMWrapper

# https://pypi.org/project/NeoCompilerModelLoaders/
import neo_loader
from neo_loader import load_model
from neo_loader import find_archive
from neo_loader import extract_model_artifacts

# onnx imports
from tvm.contrib.target.onnx import to_onnx

# cvflow imports
from frameworklibs.common import json_schema

class cvflow_cfg_keys(Enum):
    GENERAL   = 'general'
    WORKDIR   = 'work_dir'
    AMBADIR   = 'amba_files_folder_name'
    PRB_BINS  = 'prebuilt_bin_list'
    COMPILER  = 'ann_compiler_name'

    AMBALINK  = 'ambalink'
    FWBASE    = 'firmware_base'
    DIAG_BASE = 'diag_base_dir'
    DIAGDIR   = 'diag_dir'
    SDAG_IN   = 'sdag_input_file'

# short-hand for convenience
CFG = cvflow_cfg_keys

def run_command(cmd):
    """Run command, return output as string."""
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    return output.decode("ascii")

class CV22_TVM_Compilation():

    def __init__(self, model_directory, output_directory, config_json, prebuilt_bins_path, metadata_file, tar_output, debuglevel=2):
        """
        model_directory: Directory containing model file (usually /compiler/)
        output_directory: Directory to output finished artifacts
        config_json: JSON config file containing hard coded paths and names
        prebuilt_bins_path: Folder containing the following pre-built binaries: libtvm_runtime.so, libdlr.so, libamba_tvm.so
        metadata_file: Path to default metadata file
        tar_output: Boolen flag to indicate whether outputs needs to be saved as tar gz
        debuglevel: Debug level (default: 2))
        """
        self.logger = self._init_logger_(debuglevel)

        self.dir = model_directory
        self.output_dir = output_directory

        self.json_config = self._read_json_(config_json)

        self.amba_folder = self.json_config[CFG.GENERAL.value][CFG.AMBADIR.value]
        self.amba_files_dir = join(self.output_dir, self.amba_folder)
        if not isdir(self.amba_files_dir):
            makedirs(self.amba_files_dir)

        self.prebuilt_bins_path = prebuilt_bins_path
        self.prebuilt_bins = self.json_config[CFG.GENERAL.value][CFG.PRB_BINS.value].split(",")
        self.prebuilt_bins_fpath = []

        # to store ambapb artefacts etc
        self.workdir = self.json_config[CFG.GENERAL.value][CFG.WORKDIR.value]
        self._remove_dir_(self.workdir)

        self.rand_id = binascii.b2a_hex(urandom(4)).decode("utf-8")
        self.out_bname = 'compiled_' + self.rand_id

        self.ambapb_fpaths = []

        # both cavalry and flexibin
        self.sdk_bin_fpaths = []

        self.output_files = []
        self.amba_files = []

        # check if compilation is running on service or locally
        self.neo_service = self._running_on_service_()

        # get framework
        self.framework = self._get_framework_()

        # check if all required files exist
        self.model = self._validate_input_files_()

        # read metadata to dict
        self.metadata = self._read_json_(metadata_file)

        # save output to tar file?
        self.tar_output = tar_output

    def process(self):
        self._convert_to_relay_()
        self._cv22_compilation_()
        out_fname = self._save_output_to_dir_()

        return out_fname

    # private methods

    def _error_(self, err):
        self.logger.error(err)
        logging.exception(err)
        raise Exception(err)

    def _init_logger_(self, debuglevel):
        libpath = subprocess.check_output(['tv2', '-basepath', 'AmbaCnnUtils'])
        libpath = libpath.decode().rstrip('\n')
        tv2_p = join(libpath, 'parser/common/')
        if isdir(tv2_p):
            if tv2_p not in sys.path:
                sys.path.append(tv2_p)
            else:
                raise Exception('%s not found' % tv2_p)

        from frameworklibs.common.logger import ModifiedABSLLogger
        log = ModifiedABSLLogger(program_name="CV22_TVM", amba_verbosity_level=debuglevel)

        return log

    def _running_on_service_(self):
        return 'ECS_CONTAINER_METADATA_URI_V4' in environ or 'ECS_CONTAINER_METADATA_URI' in environ

    def _list_prebuilt_bins_(self):
        for f in self.prebuilt_bins:
            fpath = join(self.prebuilt_bins_path, f.strip())
            self._check_for_file_(fpath)
            self.prebuilt_bins_fpath.append(fpath)

    def _validate_input_files_(self):
        model_file = self._get_model_file_()
        return model_file

    def _read_json_(self, json_fname):
        self._check_for_file_(json_fname)
        with open(json_fname) as f:
            return json.load(f)

    def _get_model_file_(self):
        """
        Search the directory for model file (<model>.tar.gz)
        model.tar.gz is expected to contain the following:
        1. model file
        2. config.json: this contains dictionary mapping each input to shape and filepath
        3. <calib>: folder containing calibration images in binary format
           Each input is expected to have a separate folder
           The calib folder path is provided in config.json as filepath
        """
        model = []
        if self._running_on_service_():
            return find_archive()

        for f in listdir(self.dir):
            if f.endswith('.tar.gz') or f.endswith('.tar'):
                model.append(join(self.dir, f))

        if len(model) != 1:
            err = 'Expecting exactly one model file in %s. Found %s' % (self.dir, len(model))
            self._error_(err)

        self.logger.info('Found model file: %s' % model[0])

        return model[0]

    def _check_for_file_(self, fname):
        if not exists(fname):
            err = '%s not found' % fname
            self._error_(err)

    def _remove_dir_(self, fpath, ignore_errors=True):
        rmtree(fpath, ignore_errors)

    def _list_from_str_(self, in_str, dtype):
        out_list = []

        if isinstance(in_str, list):
            out_list = in_str
        elif isinstance(in_str, str):
            sh_list = in_str.split(',')
            out_list = [dtype(s) for s in sh_list]
        else:
            self._error_('Unsupport type %s, supported types: [str, list]' % type(in_str))

        return out_list

    def _check_for_nhwc_(self, shape):
        if shape[1] == 3: # nchw
            nhwc = False
        elif shape[3] == 3: # nhwc
            nhwc = True
        else:
            self._error_('Expecting 3 channel input, got %s' % str(shape))

        return nhwc

    def _transpose_dra_bin_(self, fname, dtype, shape):
        # change to numpy type
        if dtype == 'float32':
            dfmt = np.float32
        elif dtype == 'float16':
            dfmt = np.float16
        elif dtype == 'int8':
            dfmt = np.int8
        elif dtype == 'int16':
            dfmt = np.int16
        elif dtype == 'int32':
            dfmt = np.int32
        elif dtype == 'uint8':
            dfmt = np.uint8
        elif dtype == 'uint16':
            dfmt = np.uint16
        elif dtype == 'uint32':
            dfmt = np.unt32

        self.logger.info('_transpose_dra_bin_: working on file %s' % fname)
        self.logger.info('Shape before transpose: %s' % shape)
        in_arr = np.fromfile(fname, count=-1, dtype=dfmt)
        in_arr = in_arr.reshape(shape)
        in_arr = in_arr.transpose(0,2,3,1)
        self.logger.info('Shape after transpose: %s' % str(in_arr.shape))
        in_arr = in_arr.flatten()
        in_arr.tofile(fname)

    def _dra_files_(self, out_dir, dra_fname, calib_fpath, input_shape, colorfmt, dtype):

        transpose_needed = False
        shape = input_shape.copy() # make copy

        # if all files are binary, nothing more to be done
        # if some files are non binary (i.e. jpg, png etc), call imgtobin
        # NOTE::imgtobin skips binary files
        bin_files = []
        img_files = []
        for fl in listdir(calib_fpath):
            if fl.endswith('.bin'):
                bin_files.append(fl)
            else:
                img_files.append(fl)

        # all files are binary
        if len(img_files) == 0:
            dra_bin_folder = calib_fpath

        # dra files are not binary
        else:
            if len(bin_files) > 0:
                self.logger.warn('Skipping the following binary files found in dra folder (%s)' % calib_fpath)
                [self.logger.warn('%s' % b) for b in bin_files]
                self.logger.warn('Do not mix prepared (binary) and image (jpg / png / ..) files')

            ccutils_path = run_command('tv2 -basepath CommonCnnUtils').strip()
            sys.path.append(ccutils_path)
            import imgtobin

            # adjust args as needed for imgtobin
            if dtype == 'float32':
                dfmt = '1,2,0,7'
            elif dtype == 'float16':
                dfmt = '1,1,0,4'
            elif dtype == 'int8':
                dfmt = '1,0,0,0'
            elif dtype == 'int16':
                dfmt = '1,1,0,0'
            elif dtype == 'int32':
                dfmt = '1,2,0,0'
            elif dtype == 'uint8':
                dfmt = '0,0,0,0'
            elif dtype == 'uint16':
                dfmt = '0,1,0,0'
            elif dtype == 'uint32':
                dfmt = '0,2,0,0'
            else:
                self._error_('Unknown dtype (%s) in input config' % dtype)

            # convert to list
            if isinstance(shape, str):
                shape = shape.split(',')
                shape = [int(s.strip()) for s in shape]

            if len(shape) == 3:
                shape = [1] + shape
            if len(shape) != 4:
                self._error_('Expecting shape to be 3D or 4D, got %s' % str(ishape))

            transpose_needed = self._check_for_nhwc_(shape)
            if transpose_needed:
                orig_shape = shape.copy()
                shape[1] = orig_shape[3]
                shape[3] = orig_shape[1]

            shape_str = [str(s) for s in shape]
            shape_str = ','.join(shape_str)

            dra_bin_folder = join(out_dir, 'dra_imgs_bin')
            makedirs(dra_bin_folder)

            cfmt = 1 if colorfmt=='RGB' else 0

            args = []
            args.extend(['-i', calib_fpath])
            args.extend(['-o', dra_bin_folder])
            args.extend(['-c', str(cfmt)])
            args.extend(['-d', dfmt])
            args.extend(['-s', shape_str])

            status = imgtobin.main(args)
            if not status:
                self._error_('Error converting DRA images to binary')

        dra_count = 0
        with open(dra_fname, 'w') as f:
            for fl in listdir(dra_bin_folder):
                if not fl.endswith('bin'):
                    self.logger.warn('Possible bug: found non binary file (%s) in dra folder (%s). Skipping for now' % (fl, dra_bin_folder))
                    continue

                fname_with_path = join(dra_bin_folder,fl)
                if transpose_needed:
                    self.logger.debug('Transposing DRA file %s' % fname_with_path)
                    self._transpose_dra_bin_(fname_with_path, dtype, shape)
                f.write(fname_with_path + '\n')
                dra_count += 1

        return dra_count

    def _convert_to_relay_(self):
        """
        Extract model file
        Create DRA list text file
        Convert model to relay
        """
        model_path = join(self.workdir, 'model')
        rmtree(model_path, ignore_errors=True)
        makedirs(model_path)

        model_files = extract_model_artifacts(self.model, model_path)

        config_json = None
        for f in model_files:
            if f.endswith('config.json'):
                config_json = f
                break

        if not model_files:
            err = 'Model file not found in %s' % model_path 
            self._error_(err)
        if config_json is None:
            err = 'Config file (config.json) not in %s' % model_path
            self._error_(err)

        # read and parse config file
        input_shape = self._parse_config_(config_json, model_path)

        # parse model file and convert to relay
        self.module, self.params, self.aux_files, self.metadata = self._convert_model_to_relay_(model_files, input_shape)

    def _run_schema_validator_(self, config):

        CONFIG_SCHEMA = {
            "$schema" : "http://json-schema.org/draft-07/schema",
            "title" : "Ambarella Neo Compilation Spec",
            "description" : "Input config spec for cvflow compilation on neo sagemaker service",

            "definitions" : {
                "inputs" : {
                    "type" : "object",

                    "properties" : {
                        "type": "object",

                        "properties" : {
                            T.SHAPE.value : {
                                "anyOf" : [
                                    {
                                        "type" : "array",
                                        "items" : {"type" : "integer"},
                                        "minItems": 1,
                                    },
                                    {"type" : "integer"}
                                ]
                            },

                            T.FPATH.value : {"type" : "string"},

                            T.CFMT.value : {"enum" : ["RGB", "BGR"]},

                            T.MEAN.value : {
                                "anyOf" : [
                                    {
                                        "type" : "array",
                                        "items" : {"type" : "number"},
                                        "minItems": 1,
                                    },
                                    {"type" : "number"}
                                ]
                            },

                            T.SCALE.value : {
                                "anyOf" : [
                                    {
                                        "type" : "array",
                                        "items" : {"type" : "number"},
                                        "minItems" : 1,
                                    },
                                    {"type" : "number"}
                                ]
                            },
                        },

                        "required" : [T.SHAPE.value, T.FPATH.value],

                        "additionalProperties" : False,
                    },

                    "additionalProperties" : False
                },

                T.SDK.value : {"enum" : ["linux", "ambalink"]},
            }
        }

        validator = json_schema.SchemaValidator(schema=CONFIG_SCHEMA)
        validator.validate(definition=config)

    def _parse_config_(self, config_json, model_path):

        # read json config
        with open(config_json) as f:
            config_data = json.load(f)

        # validate
        self._run_schema_validator_(config_data)

        input_config = config_data['inputs']

        # check for sdk
        # default: linux
        sdk = config_data.get(T.SDK.value, 'linux')

        if sdk == 'ambalink':
            # check if firmware base exists
            self._check_for_file_(self.json_config[CFG.AMBALINK.value][CFG.FWBASE.value])

            # create diag dir
            diag_dir = self.json_config[CFG.AMBALINK.value][CFG.DIAG_BASE.value]
            rmtree(diag_dir, ignore_errors=True)
            makedirs(diag_dir)

            # create input file for superdag_gen
            if not exists(self.json_config[CFG.AMBALINK.value][CFG.SDAG_IN.value]):
                from pathlib import Path
                #open(self.json_config[CFG.AMBALINK.value][CFG.SDAG_IN.value], 'a').close()
                Path(self.json_config[CFG.AMBALINK.value][CFG.SDAG_IN.value]).touch()

            self.prebuilt_bins_path = join(self.prebuilt_bins_path, 'ambalink')

        else:
            self.json_config[CFG.AMBALINK.value] = {}
            self.prebuilt_bins_path = join(self.prebuilt_bins_path, 'linux')

        # collect prebuilt bins
        self._list_prebuilt_bins_()

        # create a txt file for DRA

        input_shape = {}

        for name,items in input_config.items():
            mangled_name = name.replace('/','__')

            calib_fpath = join(model_path, items[T.FPATH.value])

            # convert shape from str to list
            input_config[name][T.SHAPE.value] = self._list_from_str_(items[T.SHAPE.value], int)
            input_shape[name] = input_config[name][T.SHAPE.value]

            # check for colorformat
            # default: RGB
            input_config[name][T.CFMT.value] = items.get(T.CFMT.value, 'RGB')

            # check for dtype
            # default: float32
            input_config[name][T.DTYPE.value] = items.get(T.DTYPE.value, 'float32')

            # look for file with extn .bin in calib_fpath
            # if extn is not .bin, convert them to binary
            dra_fname = join(self.workdir, mangled_name+'_dra_list.txt')
            dra_count = self._dra_files_(self.workdir, dra_fname, calib_fpath, input_config[name][T.SHAPE.value], \
                                         input_config[name][T.CFMT.value], input_config[name][T.DTYPE.value])
            if dra_count == 0:
                self._error_('No dra files found in %s' % calib_fpath)

            input_config[name][T.EXTN.value] = '.bin'

            # overwrite fpath with txt file
            input_config[name][T.FPATH.value] = dra_fname

            # check for mean
            # default: None
            if T.MEAN.value in items:
                input_config[name][T.MEAN.value] = self._list_from_str_(items[T.MEAN.value], float)

            # check for scale
            # default: None
            if T.SCALE.value in items:
                input_config[name][T.SCALE.value] = self._list_from_str_(items[T.SCALE.value], float)

        self.sdk = sdk
        self.input_config = input_config

        return input_shape

    def _convert_model_to_ir_(self, model_files, input_shape):
        try:
            loader = load_model(model_files, input_shape)

        except Exception as e:
            self._error_("Loading %s model failed" % self.framework)

        else:
            return {
                'metadata': loader.metadata,
                'model_objects': loader.model_objects,
                'aux_files': loader.aux_files # should be packaged with final archive
            }

    def _get_framework_(self):
        if 'FRAMEWORK' not in environ:
            self._error_('Unknown framework, set enviroment variable FRAMEWORK before running')
        framework = environ['FRAMEWORK'].lower()

        return framework

    def _convert_model_to_relay_(self, model_artifacts, input_shape):
        conversion_dict = self._convert_model_to_ir_(model_artifacts, input_shape)

        mod = conversion_dict['model_objects'][0]
        params = conversion_dict['model_objects'][1]
        aux_files = conversion_dict['aux_files']

        if self.neo_service:
            # overwrite Model dict
            metadata = {
                'Requirements': {
                    'TargetDevice': 'AMBARELLA_CV22'
                },
                'Compilation': {
                    'CreatedTime': int(time.time())
                },
                'Model': conversion_dict['metadata']
            }
        else:
            metadata = self.metadata

        return mod, params, aux_files, metadata 

    def _cv22_compilation_(self):
        json_fname, lib_fname, params_fname = self._compile_model_(self.module, self.params, self.json_config[CFG.GENERAL.value][CFG.COMPILER.value], self.input_config, self.out_bname)

        self.output_files = [json_fname, lib_fname]
        if params_fname is not None:
            self.output_files.append(params_fname)

        self.output_files.extend(self.ambapb_fpaths)
        self.output_files.extend(self.sdk_bin_fpaths)

    def _compile_model_(self, mod, params, compiler, input_config, output_basename):
        """
        1) Convert NHWC layout (if any) to NCHW
        2) Annotate ops belonging to cv22 white list as "cv22"
        3) Partition graph to create multiple subgraphs
        4) Prune subgraphs to retain only one cv22 subgraph (Note: currently it is the first cv22 subgraph)
        5) Convert cv22 subgraphs to onnx
        6) Compile onnx models using cvflow compiler
        7) Call relay.build
        8) Serialize to files
        """
        try:
            self.logger.debug("---------- Infer relay expression type ----------")
            mod = transform.InferType()(mod)
            self.logger.debug(mod.astext(show_meta_data=False))

            self.logger.debug('---------- Original Graph ----------')
            mod = transform.RemoveUnusedFunctions()(mod)
            self.logger.debug(mod.astext(show_meta_data=False))

            self.logger.debug('---------- NHWC -> NCHW ----------')
            mod = transform.ConvertLayout({'nn.conv2d': ['NCHW', 'default']})(mod)
            self.logger.debug(mod.astext(show_meta_data=False))

            self.logger.debug('---------- Bound Graph ----------')
            if params:
                mod['main'] = bind_params_by_name(mod['main'], params)
            self.logger.debug(mod.astext(show_meta_data=False))

            mod = transform.FoldConstant()(mod)

            self.logger.debug("---------- Annotated Graph ----------")
            mod = transform.AnnotateTarget(compiler)(mod)
            self.logger.debug(mod.astext(show_meta_data=False))
            
            self.logger.debug("---------- Merge Compiler Regions ----------")
            mod = transform.MergeCompilerRegions()(mod)
            self.logger.debug(mod.astext(show_meta_data=False))
            
            self.logger.debug("---------- Partioned Graph ----------")
            mod = transform.PartitionGraph()(mod)
            self.logger.debug(mod.astext(show_meta_data=False))
            
            self.logger.debug("---------- Infer relay expression type ----------")
            mod = transform.InferType()(mod)
            self.logger.debug(mod.astext(show_meta_data=False))
            
            self.logger.debug("---------- Pruned Graph ----------")
            mod = PruneSubgraphs(mod, prune_first=True)
            self.logger.debug(mod.astext(show_meta_data=False))
            
            module_list = PartitionOneToModule(mod, compiler)

            if isinstance(mod['main'].ret_type, tvm.ir.type.TupleType):
                out_list = mod['main'].ret_type.fields
            elif isinstance(mod['main'].ret_type, tvm.ir.tensor_type.TensorType):
                out_list = [mod['main'].ret_type]
            else:
                self._error_('Unknown return type %s' % type(mod['main'].ret_type))
            self._update_metadata_outputs_(out_list)

            output_folder = join(self.workdir, 'cvflow')
            makedirs(output_folder)

            for name, module in module_list.items():
                self.logger.info("---------- Converting subgraph %s to onnx ----------" % name)
                mod_name = name + '_' + self.rand_id
                onnx_model = to_onnx(module, {}, mod_name, path=mod_name+'.onnx')

                # create diag dir for this subgraph
                if self.sdk == 'ambalink':
                    diag_dir = join(self.json_config[CFG.AMBALINK.value][CFG.DIAG_BASE.value], name)
                    makedirs(diag_dir)
                    self.json_config[CFG.AMBALINK.value][CFG.DIAGDIR.value] = diag_dir

                self.logger.info("---------- Invoking Cvflow Compilation ----------")
                ambapb_fpath, sdk_bin_fpath = CvflowCompilation(
                        model_proto=onnx_model,
                        output_name=mod_name,
                        output_folder=output_folder,
                        metadata=self.metadata['Model'],
                        input_config=input_config,
                        sdk=self.sdk,
                        ambalink_cfg=self.json_config[CFG.AMBALINK.value])
                self.logger.info('Saved ambapb to: %s\n' % ambapb_fpath)
                self.logger.info('Saved compiled model to: %s\n' % sdk_bin_fpath)

                self._check_for_file_(ambapb_fpath)
                self.ambapb_fpaths.append(ambapb_fpath)

                self._check_for_file_(sdk_bin_fpath)
                self.sdk_bin_fpaths.append(sdk_bin_fpath)
            
            # set env variable CV22_COMPILED_BNAMES - to be used by codegen and runtime
            set_env_variable('CV22_RAND_ID', self.rand_id)

            # mode: EMULATOR or TARGET
            exe_mode = GetCvflowExecutionMode()

            # initialize wrapper
            ct = CVFlowTVMWrapper(exe_mode, self.logger)

            # tvm compilation
            ct.relay_build(mod, params, opt_level=3)

            # TODO: Need to add logic to determine to use vm or graph compilation api
            #ct.relayvm_build(mod, params, opt_level=3)

            # serialize
            json_fname, lib_fname, params_fname = ct.serialize(basename=output_basename)

            return json_fname, lib_fname, params_fname

        except Exception as e:
            self._error_(str(e))

    def _update_metadata_outputs_(self, out_list):

        outputs = []
        cnt = 0
        for o in out_list:
            name = 'output_' + str(cnt)
            sh = [int(k) for k in o.shape]
            dt = o.dtype
            outputs.extend([{'name':name, 'shape':sh, 'dtype':dt}])

            cnt += 1

        self.metadata['Model']['Outputs'] = outputs.copy()

    def _save_output_to_dir_(self):
        metadata_file = join(self.workdir, self.out_bname + '.meta')
        self._save_dict_to_json_file_(metadata_file, self.metadata)

        self.output_files.extend([metadata_file])
        self.amba_files.extend(self.prebuilt_bins_fpath)
        self.amba_files.extend([self.aux_files])

        out_fname = self._get_output_fname_()
        self._save_output_(out_fname)

        return self.output_dir

    def _get_output_fname_(self):
        model_name = basename(self.model)
        if model_name.endswith('.tar.gz'):
            model_name = model_name[:len(model_name)-7]
        else:
            model_name = model_name[:len(model_name)-4]

        output_name = join(self.dir, model_name+'_compiled.tar.gz')

        return output_name

    def _save_dict_to_json_file_(self, json_fname, data):
        with open(json_fname, 'w') as fp:
            json.dump(data, fp, indent=1)

    def _save_output_(self, out_fname):
        flist = [f for f in self.output_files if f is not None]
        logging.info("{}".format(flist))
        flat_list = []
        for i in flist:
            if isinstance(i, list):
                flat_list.extend(i)
            else:
                flat_list.append(i)
        logging.info("{}".format(flat_list))

        alist = [f for f in self.amba_files if f is not None]
        logging.info("{}".format(alist))
        amba_list = []
        for i in alist:
            if isinstance(i, list):
                amba_list.extend(i)
            else:
                amba_list.append(i)
        logging.info("{}".format(amba_list))

        all_files = self._consolidate_files_(flat_list, amba_list)

        if self.tar_output:
            self._compress_(out_fname, all_files)

    def _consolidate_files_(self, flat_list, amba_list):
        all_files = []
        for item in flat_list:
            copy(item, self.output_dir)
            all_files.append(join(self.output_dir, basename(item)))

        for item in amba_list:
            copy(item, self.amba_files_dir)

        # copy amba_files dir
        # doing it this way to preserve hierarchy when tarring
        all_files.append(self.amba_files_dir)

        return all_files

    def _compress_(self, tar_fname, flist):
        with tarfile.open(tar_fname, 'w:gz') as tar:
            for item in flist:
                tar.add(item, arcname=basename(item))

def write_status(log, status):
    with open(log, 'w') as f:
        f.write(str(status)+'\n')

def makerun(args):

    # this is to catch the case when script is invoked without model or framework set
    # need /compiler/ folder to write out error message
    if not isdir(args.model_dir):
        makedirs(args.model_dir)
    if not isdir(args.output_dir):
        makedirs(args.output_dir)

    try:
        c = CV22_TVM_Compilation(args.model_dir, args.output_dir, args.config, args.prebuilt_bins_path, args.metadata_path, args.tar_output, args.verbosity)
        out_fname = c.process()

        print('CV22 compilation successful!')

    except Exception as e:
        # write appropriate error message
        logging.exception(e)
        err_str = 'AmbarellaError::' + str(e)
        err_file = environ.get('SM_NEO_COMPILATION_ERROR_FILE', join(args.output_dir, 'COMPILATION_FAILED'))

        # COMPILATION_FAILED
        write_status(err_file, err_str)

        print('CV22 compilation failed!')


def main(args):
    model_input_dir = environ.get('SM_NEO_INPUT_MODEL_DIR', '/compiler/')
    model_output_dir = environ.get('SM_NEO_COMPILED_MODEL_DIR', '/compiler/')

    parser = argparse.ArgumentParser(description='Script to run tvm compilation for cv22')

    parser.add_argument('-d', '--model_dir', type=str, required=False, default=model_input_dir,
                        metavar='Directory containing model file',
                        help='Directory containing input <model>.tar.gz')

    parser.add_argument('-o', '--output_dir', type=str, required=False, default=model_output_dir,
                        metavar='Directory containing output files',
                        help='Directory to contain output files')

    parser.add_argument('-p', '--prebuilt_bins_path', type=str, required=False, default='/home/dlr/prebuild/amba/lib/',
                        metavar='Folder containing pre-built binaries necessary for tvm / dlr compilation and runtime',
                        help='Folder containing the following pre-built binaries: libtvm_runtime.so, libdlr.so, libamba_tvm.so.*')

    parser.add_argument('-m', '--metadata_path', type=str, required=False, default='/home/amba_tvm_release/metadata/default_metadata.json',
                        metavar='Default metadata file path',
                        help='Path to default metadata file')

    parser.add_argument('-c', '--config', type=str, required=False, default='/home/tvm/tests/python/relay/cvflow_config.json',
                        metavar='Cvflow compilation specific config containing hard coded paths and names',
                        help='Cvflow compilation specific config containing hard coded paths and names')

    parser.add_argument('-v', '--verbosity', type=int, required=False, default=2,
                        metavar='Debug level 0 - 5',
                        help='Debug level 0 - 5')

    parser.add_argument('-t', '--tar_output', action='store_true', required=False,
                        help='Save output to tar.gz')

    args = parser.parse_args(args)

    return (makerun(args))

# Entry point
if __name__ == '__main__':
    main(sys.argv[1:])

