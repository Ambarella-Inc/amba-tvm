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
from tvm import relay
from tvm.relay import transform
from tvm.relay.op.annotation import compiler_begin, compiler_end

# Imports for expr
from tvm.relay.expr_functor import ExprMutator, ExprVisitor
from tvm.relay.function import Function

# Onnx imports
from tvm.contrib.target.onnx import to_onnx

import tvm.relay.op.contrib.cv22
from tvm.relay.backend.contrib.cv22 import PruneSubgraphs, PartitionsToModules

def test_manual():
    #============= Constructing a simple graph ==============
    dtype = "float32"
    i_shape = (1, 3, 224, 224) # NCHW
    o_shape = (100, 428) # NCHW

    data = relay.var('data', shape=(i_shape), dtype=dtype)

    begin = relay.annotation.compiler_begin(data, "cv22")
    node0 = relay.image.resize(begin, size=o_shape, method='bicubic', coordinate_transformation_mode='asymmetric')
    out   = relay.annotation.compiler_end(node0, "cv22")
    f     = relay.Function([data], out)
    mod2  = tvm.IRModule.from_expr(f)

    print('---------- Annotated graph ----------')
    #print(mod2.astext(show_meta_data=False))

    # graph partitioning
    mod2_partition = transform.PartitionGraph()(mod2)
    print('---------- Partitioned graph ----------')
    print(mod2_partition.astext(show_meta_data=False))

    # convert to onnx
    module_list = PartitionsToModules(mod2_partition, 'cv22')
    for name, module in module_list.items():
        print("---------- Converting subgraph %s to onnx ----------" % name)
        onnx_model = to_onnx(module, {}, name, path='resize.onnx')

    print('DONE')

if __name__ == '__main__':
    test_manual()
