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

"""Unit tests for cv22 ades runtime"""
import sys
from os import makedirs, listdir
from os.path import join, isdir
from shutil import rmtree
import numpy as np

# tvm imports
from tvm.contrib.tar import untar
from tvm.relay.backend.contrib.cv22 import CVFlowTVMWrapper

class CV22_TVM_Emu_Runtime():

    def __init__(self, compiled_model):

        # create logger
        self.logger = self._init_logger_(debuglevel=2)

        # create new tmpdir
        self.tmpdir = '/tmp/test_amba/eval/'
        self._create_dir_(self.tmpdir)

        # extract ambapb, .json, .lib and .params
        self._extract_compilation_artefacts_(compiled_model)

        # move amba to /tmp/test_amba/
        self._move_ambapb_to_rundir_()

    def run(self, inputs):
        # create inputs_map
        self._create_inputs_map_(inputs)

        # run model
        self._run_model_()

    def save_outputs(self):
        output_bname = 'output_'
        cnt = 0
        for o in self.outputs:
            output_fname = output_bname + str(cnt) + '.bin'
            out = o.flatten()
            out.tofile(output_fname)
            self.logger.info('Saved output file %s' % output_fname)
            cnt += 1

            '''
            # print top 5
            softmax = np.exp(out) / np.sum(np.exp(out))
            top5_idx = np.argsort(softmax)[::-1][:5]
            top5_val = [softmax[i] for i in top5_idx]

            print('Top 5 categories:', top5_idx)
            print('Top 5 scores:', top5_val)
            '''

    ## private methods

    def _create_dir_(self, directory):
        rmtree(directory, ignore_errors=True)
        makedirs(directory)

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

        from frameworklibs.common.logger import ModifiedABSLLogger
        log = ModifiedABSLLogger(program_name="CV22_TVM", amba_verbosity_level=debuglevel)

        return log

    def _extract_compilation_artefacts_(self, compiled_model):
        untar(compiled_model, self.tmpdir)

        for f in listdir(self.tmpdir):
            fpath = join(self.tmpdir,f)
            if f.endswith('.json') and ('amba' not in f):
                self.json_fname = fpath
            elif f.endswith('.so') and ('libtvm_runtime' not in f):
                self.lib_fname = fpath
            elif f.endswith('.params'):
                self.params_fname = fpath
            elif f.endswith('.onnx'):
                self.ambapb_fname = fpath

    def _move_ambapb_to_rundir_(self):
        pass

    def _create_inputs_map_(self, inputs_arg):
        self.inputs_map = {}
        for i in inputs_arg:
            n,f = i.split('=')

            if isinstance(f, str):
                self.inputs_map[n] = np.fromfile(f, count=-1, dtype=np.float32)
            else:
                self.inputs_map[n] = f

    def _run_model_(self):
        # initialize wrapper
        ct = CVFlowTVMWrapper(mode='EMULATOR', logger=self.logger)

        # deserialize
        ct.deserialize(self.json_fname, self.lib_fname, self.params_fname)

        # run
        self.outputs = ct.tvm_runtime(self.inputs_map)

def makerun(args):
    rt = CV22_TVM_Emu_Runtime(args.compiledmodel)
    rt.run(args.inputs)
    rt.save_outputs()
    print('Execution completed!')

import argparse

def main(args):
    parser = argparse.ArgumentParser(description='Script to run inference on tvm compiled model')

    parser.add_argument('-m', '--compiledmodel', type=str, required=True,
                        metavar='Path to compiled model artefacts file',
                        help='Path to <>_compiled.tar.gz')

    parser.add_argument('-i', '--inputs', type=str, action='append', required=True,
                        metavar='Input name and file',
                        help='Eg: -i data=input.bin')

    args = parser.parse_args(args)

    return (makerun(args))

# Entry point
if __name__ == '__main__':
    main(sys.argv[1:])
