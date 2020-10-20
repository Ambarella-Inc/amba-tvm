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
from os import makedirs, listdir, environ
from os.path import exists, join, isdir, basename
from shutil import rmtree
import json
import logging
logging.basicConfig(level=logging.DEBUG)

# tvm imports
import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
from tvm.contrib.tar import tar

# onnx imports
from tvm.contrib.target.onnx import to_onnx

class CV22_TVM_Compilation():

    def __init__(self, model_directory, cc_libtvm_path, debuglevel=2):
        """
        model_directory: Directory containing model file (usually /compiler/). Output will be stored in the same directory
        cc_libtvm_path:  Path to cross-compiled libtvm_runtime.so file (usually /compiler/libtvm_runtime.so)
        """
        self.dir    = model_directory
        self.libtvm = cc_libtvm_path

        # to store ambapb artefacts etc
        self.tmpdir = '/tmp/test_amba/'
        self._remove_tmp_dir_()

        self.out_bname = 'compiled'

        self.logger = self._init_logger_(debuglevel)

        # import neo loader package either from service or locally
        self._import_loader_()

        # get framework
        self.framework = self._get_framework_()

        # check if all required files exist
        self.model = self._validate_input_files_()

    def process(self):
        self._convert_to_relay_()
        self._cv22_compilation_()
        out_fname = self._save_output_to_file_()

        return out_fname

    # private methods

    def _error_(self, err):
        self.logger.error(err)
        logging.exception(err)
        raise Exception(err)

    def _init_logger_(self, debuglevel):
        import subprocess
        libpath = subprocess.check_output(['tv2', '-basepath', 'AmbaCnnUtils'])
        libpath = libpath.decode().rstrip('\n')
        tv2_p = join(libpath, 'parser/common/')
        if isdir(tv2_p):
            if tv2_p not in sys.path:
                sys.path.append(tv2_p)
            else:
                raise Exception('%s not found' % tv2_p)

        from logger import ModifiedABSLLogger
        log = ModifiedABSLLogger(program_name="CV22_TVM", amba_verbosity_level=debuglevel)

        return log

    def _import_loader_(self):
        LOADER_SERVICE_PATH = '/compiler/neo_shared/modules/'
        LOADER_LOCAL_PATH = '/home/neo_loader/'

        if isdir(LOADER_SERVICE_PATH):
            self.logger.info('Loading loader from neo service')
            sys.path.append(LOADER_SERVICE_PATH)
        elif isdir(LOADER_LOCAL_PATH):
            self.logger.info('Loading local loader')
            sys.path.append(LOADER_LOCAL_PATH)
        else:
            err = 'Unable to load neo loader locally (%s) or from neo service (%s)' % (LOADER_LOCAL_PATH, LOADER_SERVICE_PATH) 
            self._error_(err)

        import neo_loader

    def _validate_input_files_(self):
        model_file = self._get_model_file_()
        self._check_for_file_(self.libtvm)

        return model_file

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

    def _remove_tmp_dir_(self):
        rmtree(self.tmpdir, ignore_errors=True)

    def _convert_to_relay_(self):
        """
        Extract model file
        Create DRA list text file
        Convert model to relay
        """
        model_path = join(self.tmpdir, 'model')
        rmtree(model_path, ignore_errors=True)
        makedirs(model_path)

        from neo_loader import extract_model_artifacts
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

        ## CONFIG STUFF
        # (TBD) move to a func

        # read config
        input_config = self._parse_config_(config_json)

        # create a txt file for DRA
        from tvm.relay.backend.contrib.cv22 import tags as T

        input_shape = {}

        for name,items in input_config.items():
            calib_fpath = join(model_path, items[T.FPATH.value])

            mangled_name = name.replace('/','__')

            dra_fname = join(self.tmpdir, mangled_name+'_dra_list.txt')
            with open(dra_fname, 'w') as f:
                for fl in listdir(calib_fpath):
                    if fl.endswith('.bin'):
                        f.write(join(calib_fpath,fl) + '\n')

            # overwrite fpath with txt file
            input_config[name][T.FPATH.value] = dra_fname

            # convert shape from str to list
            sh_list = items[T.SHAPE.value].split(',')
            sh_list = [int(s) for s in sh_list]
            input_shape[name] = sh_list
            input_config[name][T.SHAPE.value] = sh_list

        ## (END) CONFIG STUFF

        # parse model file and convert to relay
        self.module, self.params, self.aux_files = self._convert_model_to_relay_(model_files, input_shape)

        self.input_config = input_config

    def _parse_config_(self, config_json):
        with open(config_json) as f:
            config_data = json.load(f)

        if 'inputs' not in config_data:
            self._error_('"inputs" not found in config')

        input_config = config_data['inputs']

        return input_config

    def _convert_model_to_ir_(self, model_files, input_shape):
        try:
            from neo_loader import load_model
            loader = load_model(model_files, input_shape)

        except Exception as e:
            self._error_("Loading %s model failed" % self.framework)

        else:
            return {
                'metadata': loader.metadata, # shouldn't be needed
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

        return mod, params, aux_files 

    def _cv22_compilation_(self):
        json_fname, lib_fname, params_fname = self._compile_model_(self.module, self.params, 'cv22', self.input_config, self.out_bname)

        cv22_bin_fname = join(self.tmpdir, 'prepare/cv22_0.ambapb.fastckpt.onnx')
        self._check_for_file_(cv22_bin_fname)

        cavalry_bin_fname = join(self.tmpdir, 'prepare/cv22_0.amba')
        self._check_for_file_(cavalry_bin_fname)

        self.output_files = [json_fname, lib_fname, params_fname, cv22_bin_fname, cavalry_bin_fname]

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
            # cvflow imports
            import tvm.relay.op.contrib.cv22
            from tvm.relay.backend.contrib.cv22 import PruneSubgraphs, PartitionsToModules, GetCvflowExecutionMode, CvflowCompilation, CVFlowTVMWrapper

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

            self.logger.debug("---------- Annotated Graph ----------")
            mod = transform.AnnotateTarget(compiler)(mod)
            self.logger.debug(mod.astext(show_meta_data=False))

            self.logger.debug("---------- Merge Compiler Regions ----------")
            mod = transform.MergeCompilerRegions()(mod)
            self.logger.debug(mod.astext(show_meta_data=False))

            self.logger.debug("---------- Partioned Graph ----------")
            mod = transform.PartitionGraph()(mod)
            self.logger.debug(mod.astext(show_meta_data=False))

            self.logger.debug("---------- Pruned Graph ----------")
            mod = PruneSubgraphs(mod, compiler, 1, self.logger)
            self.logger.debug(mod.astext(show_meta_data=False))

            output_folder = join(self.tmpdir, 'prepare')
            makedirs(output_folder)

            module_list = PartitionsToModules(mod, compiler)
            for name, module in module_list.items():
                self.logger.info("---------- Converting subgraph %s to onnx ----------" % name)
                onnx_model = to_onnx(module, {}, name)

                self.logger.info("---------- Invoking Cvflow Compilation ----------")
                save_path = CvflowCompilation(model_proto=onnx_model, \
                                              output_name=name, \
                                              output_folder=output_folder, \
                                              input_config=input_config)
                self.logger.info('Saved compiled model to: %s\n' % save_path)

            # mode: EMULATOR or TARGET
            exe_mode = GetCvflowExecutionMode()

            # initialize wrapper
            ct = CVFlowTVMWrapper(exe_mode, self.logger)

            # tvm compilation
            ct.relay_build(mod)

            # serialize
            json_fname, lib_fname, params_fname = ct.serialize(basename=output_basename)

            return json_fname, lib_fname, params_fname

        except Exception as e:
            self._error_(str(e))

    def _save_output_to_file_(self):

        out_fname = self._get_output_fname_()

        self.output_files.extend([self.libtvm, self.aux_files])
        self._save_output_(out_fname)

        return out_fname

    def _get_output_fname_(self):
        model_name = basename(self.model)
        if model_name.endswith('.tar.gz'):
            model_name = model_name[:len(model_name)-7]
        else:
            model_name = model_name[:len(model_name)-4]

        output_name = join(self.dir, model_name+'_compiled.tar.gz')

        return output_name

    def _save_output_(self, tar_fname):
        flist = [f for f in self.output_files if f is not None] 

        logging.info("{}".format(flist))
        flat_list = []
        for i in flist:
            if isinstance(i, list):
                flat_list.extend(i)
            else:
                flat_list.append(i)
        logging.info("{}".format(flat_list))

        tar(tar_fname, flat_list)


def write_status(log, status):
    with open(log, 'w') as f:
        f.write(status+'\n')

def makerun(args):

    # this is catch the case when script is invoked without model or framework set
    # need /compiler/ folder to write out error message
    if not isdir(args.modeldir):
        makedirs(args.modeldir)

    try:
        c = CV22_TVM_Compilation(args.modeldir, args.libtvmpath)
        out_fname = c.process()

        # COMPILATION_COMPLETE
        write_status(join(args.modeldir,'COMPILATION_COMPLETE'), out_fname)

        print('CV22 compilation successful!')

    except Exception as e:
        # write appropriate error message
        logging.exception(e)
        err_str = 'AmbarellaError::' + str(e)

        # COMPILATION_FAILED
        write_status(join(args.modeldir,'COMPILATION_FAILED'), err_str)

        print('CV22 compilation failed!')

import argparse

def main(args):
    parser = argparse.ArgumentParser(description='Script to run tvm compilation for cv22')

    parser.add_argument('-d', '--modeldir', type=str, required=False, default='/compiler/',
                        metavar='Directory containing model file',
                        help='Directory containing input <model>.tar.gz')

    parser.add_argument('-l', '--libtvmpath', type=str, required=False, default='/home/amba_tvm_release/cross_compiled/libtvm_runtime.so',
                        metavar='libtvm_runtime.so path',
                        help='Path to cross-compiled libtvm_runtime.so')

    args = parser.parse_args(args)

    return (makerun(args))

# Entry point
if __name__ == '__main__':
    main(sys.argv[1:])


