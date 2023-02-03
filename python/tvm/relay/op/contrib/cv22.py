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
# pylint: disable=invalid-name, unused-argument
"""CV22 library supported operators.
"""
import tvm.ir

target = "target.cv22"

def _register_external_op_helper(op_name, supported=True):
    """The helper function to indicate that a given operator can be supported
    by CV22.
    Paramters
    ---------
    op_name : Str
        The name of operator that will be registered.
    Returns
    -------
    f : callable
        A function that returns if the operator is supported by CV22.
    """
    @tvm.ir.register_op_attr(op_name, "target.cv22")
    def _func_wrapper(expr):
        return supported

    return _func_wrapper

_register_external_op_helper("log")
_register_external_op_helper("sqrt")
_register_external_op_helper("rsqrt")
#_register_external_op_helper("exp")
_register_external_op_helper("sigmoid")
_register_external_op_helper("add")
_register_external_op_helper("subtract")
_register_external_op_helper("multiply")
_register_external_op_helper("divide")
#_register_external_op_helper("mod") # TBD, Can be supported by using two or more SGL nodes
_register_external_op_helper("tanh")
_register_external_op_helper("concatenate")
_register_external_op_helper("expand_dims")
_register_external_op_helper("nn.softmax")
_register_external_op_helper("nn.log_softmax")
_register_external_op_helper("nn.relu")
_register_external_op_helper("nn.dropout")
_register_external_op_helper("nn.batch_norm")
_register_external_op_helper("nn.bias_add")

_register_external_op_helper("nn.conv2d")
_register_external_op_helper("nn.conv2d_transpose")
_register_external_op_helper("nn.dense")
_register_external_op_helper("nn.max_pool2d")
_register_external_op_helper("nn.avg_pool2d")
_register_external_op_helper("nn.global_max_pool2d")
_register_external_op_helper("nn.global_avg_pool2d")
_register_external_op_helper("nn.upsampling")
_register_external_op_helper("nn.batch_flatten")
_register_external_op_helper("nn.pad")
_register_external_op_helper("nn.lrn")
_register_external_op_helper("nn.l2_normalize")
_register_external_op_helper("nn.contrib_conv2d_winograd_without_weight_transform")

#_register_external_op_helper("nn.leaky_relu")
_register_external_op_helper("nn.prelu")
_register_external_op_helper("image.resize2d")
_register_external_op_helper("reshape")
_register_external_op_helper("reshape_like")
_register_external_op_helper("copy")
_register_external_op_helper("transpose")
_register_external_op_helper("layout_transform")
_register_external_op_helper("squeeze")
_register_external_op_helper("floor")
_register_external_op_helper("ceil")
_register_external_op_helper("sign")
_register_external_op_helper("trunc")
_register_external_op_helper("clip")
_register_external_op_helper("round")
_register_external_op_helper("abs")
_register_external_op_helper("negative")
_register_external_op_helper("take")
_register_external_op_helper("zeros")
_register_external_op_helper("zeros_like")
_register_external_op_helper("ones")
_register_external_op_helper("ones_like")
_register_external_op_helper("gather_nd")
_register_external_op_helper("full")
_register_external_op_helper("full_like")
_register_external_op_helper("cast")
_register_external_op_helper("reinterpret")
_register_external_op_helper("split")
_register_external_op_helper("arange")
_register_external_op_helper("stack")
_register_external_op_helper("repeat")
_register_external_op_helper("tile")
_register_external_op_helper("reverse")

_register_external_op_helper("maximum")
_register_external_op_helper("minimum")
_register_external_op_helper("power")
_register_external_op_helper("argmax")
_register_external_op_helper("argmin")
_register_external_op_helper("sum")
_register_external_op_helper("max")
_register_external_op_helper("min")
_register_external_op_helper("mean")
_register_external_op_helper("prod")
_register_external_op_helper("strided_slice")
_register_external_op_helper("broadcast_to")

