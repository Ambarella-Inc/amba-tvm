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
# pylint: disable=invalid-name,arguments-differ,no-else-return,unused-argument,missing-docstring
"""
Relay TensorRT codegen.
"""
import os
import numpy as np
import tvm
import tvm.ir
import tvm.relay.transform as transform
from tvm import relay
from tvm.relay.expr import Call, Constant, Tuple, GlobalVar, Var, TupleGetItem
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.transform import _ffi_api
from tvm.relay.expr_functor import ExprMutator, ExprVisitor


class LegalizeLayoutTranform(ExprMutator):
    """
    Legalize Relay layout transforms to transpose ops to simplify TensorRT conversion.
    """

    def visit_call(self, expr):
        visit = super().visit_call(expr)
        if expr.op == tvm.relay.op.get("layout_transform"):
            src_layout = expr.attrs["src_layout"]
            dst_layout = expr.attrs["dst_layout"]
            if src_layout == "NCHW" and dst_layout == "NHWC":
                return relay.transpose(visit.args[0], axes=[0, 2, 3, 1])
            elif src_layout == "NHWC" and dst_layout == "NCHW":
                return relay.transpose(visit.args[0], axes=[0, 3, 1, 2])
            elif src_layout == "NDHWC" and dst_layout == "NCDHW":
                return relay.transpose(visit.args[0], axes=[0, 4, 1, 2, 3])
            elif src_layout == "NCDHW" and dst_layout == "NDHWC":
                return relay.transpose(visit.args[0], axes=[0, 2, 3, 4, 1])
            elif src_layout == "HWIO" and dst_layout == "OIHW":
                return relay.transpose(visit.args[0], axes=[3, 2, 0, 1])
            elif src_layout == "HWOI" and dst_layout == "OIHW":
                return relay.transpose(visit.args[0], axes=[2, 3, 0, 1])
            elif src_layout == "HWIO" and dst_layout == "IOHW":
                return relay.transpose(visit.args[0], axes=[2, 3, 0, 1])
        return visit


class RemoveDropout(ExprMutator):
    """
    Removes all nn.dropout from an expr.
    """

    def visit_tuple_getitem(self, expr):
        visit = super().visit_tuple_getitem(expr)
        if visit.index != 0:
            return visit
        elif isinstance(visit.tuple_value, Call) and visit.tuple_value.op.name == "nn.dropout":
            return visit.tuple_value.args[0]
        return visit


@transform.function_pass(opt_level=0)
class LegalizeLayoutTranformPass:
    def transform_function(self, func, mod, _):
        return LegalizeLayoutTranform().visit(func)


@transform.function_pass(opt_level=0)
class RemoveDropoutPass:
    def transform_function(self, func, mod, _):
        return RemoveDropout().visit(func)


def GetTrtVersion():
    """Gets the version of TensorRT that TVM is built against.

    Returns
    -------
    ret: Tuple[int]
        TensorRT version as a tuple of major, minor, and patch number. If TVM
        is not built with TensorRT, an empty tuple is returned instead.
    """
    return tuple(map(int, _ffi_api.GetTrtVersion()))


def IsTrtRuntimeAvailable():
    if not tvm.get_global_func("relay._transform.GetTrtVersion", True):
        return False
    return GetTrtVersion() != ()


def check_dynamism(args, op_name):
    """
    This function checks for dynamism inside any of the args in the op.
    Can be used to offload dynamic ops that are not supported by TRT to
    be offloaded to relay VM.

    Raises a NotImplementedError if the type of the arg is not of types
    Call, Var, Constant, or TupleGetItem.

    Parameters
    ----------
    args: a TRT array of the arguments of the op
    op_name: name of the op for debugging purposes only

    Returns
    ----------
    True if dynamism is present, False otherwise
    """
    for arg in args:
        if isinstance(arg, (Call, Var, Constant, TupleGetItem)):
            for dim_shape in arg.checked_type.shape[1:]:
                if isinstance(dim_shape, tvm.tir.expr.Any):
                    print(
                        "Dynamic inputs are not supported for TensorRT for ",
                        op_name,
                        arg.checked_type.shape,
                    )
                    return True
        elif isinstance(arg, Tuple):
            return check_dynamism(arg.fields, op_name)
        else:
            print(
                "Arg not supported in TensorRT for ",
                op_name,
                type(arg),
            )
            return True
    return False


def _register_external_op_helper(op_name, supported=True):
    @tvm.ir.register_op_attr(op_name, "target.tensorrt")
    def _func_wrapper(attrs, args):
        if check_dynamism(args, op_name):
            return False
        if any([x.checked_type.dtype != "float32" for x in args]):
            print("Only float32 inputs are supported for TensorRT.")
            return False
        # TODO (codeislife99): Here we are excluding multiply calculations which get "batched" in
        # implicit batch mode. This leads to wrong or invalid multiply calculations.
        # Since the Neo-service uses implicit batch mode=True, this is a temporary workaround.
        # A more generalizable workaround is in the works for a future update.
        if op_name == "multiply":
            shapes = [
                [
                    int(x) if not isinstance(x, tvm.tir.expr.Any) else -1
                    for x in arg.checked_type.shape
                ]
                for arg in args
            ]
            if all(
                [list(map(int, shape)) in [[300, 64, 7, 7], [300, 1, 1, 1]] for shape in shapes]
            ):
                return False

        return supported

    return _func_wrapper


def _register_external_op_helper_func(op_name, func, trt_version):
    @tvm.ir.register_op_attr(op_name, "target.tensorrt")
    def _func_wrapper(attrs, args):
        if check_dynamism(args, op_name):
            return False
        if any([x.checked_type.dtype != "float32" for x in args]):
            print("Only float32 inputs are supported for TensorRT.")
            return False
        return func(attrs, args, op_name, trt_version)

    return _func_wrapper


def _register_external_dynamic_check_func(op_name, func):
    @tvm.ir.register_op_attr(op_name, "target.tensorrt")
    def _func_wrapper(attrs, args):
        if check_dynamism(args, op_name):
            return False
        return func(attrs, args)

    return _func_wrapper


def register_tensorrt_annotations(trt_version, use_implicit_batch=True):
    if hasattr(register_tensorrt_annotations, "registered"):
        # Can't register annotations more than once.
        return
    register_tensorrt_annotations.registered = True
    if not use_implicit_batch and trt_version < (6, 0, 1):
        print("Explicit batch mode only available for TRT 6+")
        use_implicit_batch = True
    # Ops which are always supported
    _register_external_op_helper("nn.relu")
    _register_external_op_helper("sigmoid")
    _register_external_op_helper("tanh")
    _register_external_op_helper("subtract")
    _register_external_op_helper("multiply")
    _register_external_op_helper("divide")
    _register_external_op_helper("power")
    _register_external_op_helper("maximum")
    _register_external_op_helper("minimum")
    _register_external_op_helper("exp")
    _register_external_op_helper("log")
    _register_external_op_helper("sqrt")
    _register_external_op_helper("abs")
    _register_external_op_helper("negative")
    _register_external_op_helper("nn.batch_flatten")
    _register_external_op_helper("clip")
    # TODO(trevmorr): Temporarily disable split due to TRT bug on xavier.
    # _register_external_op_helper("split")
    # _register_external_op_helper("slice_like")

    def add_whitelist_fn(attrs, args):  # pylint: disable=unused-variable
        shapes = [
            [int(x) if not isinstance(x, tvm.tir.expr.Any) else -1 for x in arg.checked_type.shape]
            for arg in args
        ]

        for shape in shapes:
            if len(shape) < 1:
                return False

        if any([x.checked_type.dtype != "float32" for x in args]):
            print("Only float32 inputs are supported for TensorRT.")
            return False

        if (
            (isinstance(args[0], Constant) or isinstance(args[1], Constant))
            and shapes[0][0] == shapes[1][0]
            and shapes[0][0] != 1
            and (len(shapes[0]) > 3 or len(shapes[1]) > 3)
        ):
            print("add: bug in TRT with adding batched constants.")
            return False

        # Skip this add op in TRT to avoid accuracy mismatch
        if all([list(map(int, shape)) == [1, 546, 1, 1] for shape in shapes]):
            print("add: bug in TRT with add of shape (1, 546, 1, 1).")
            return False

        return True

    def batch_norm_whitelist_fn(attrs, args):  # pylint: disable=unused-variable
        if any([x.checked_type.dtype != "float32" for x in args]):
            print("Only float32 inputs are supported for TensorRT.")
            return False
        if int(attrs.axis) != 1 and int(attrs.axis) != 3:
            print("nn.batch_norm: axis is {} but must be 1 or 3.".format(int(attrs.axis)))
            return False
        return True

    def softmax_whitelist_fn(attrs, args):  # pylint: disable=unused-variable
        if any([x.checked_type.dtype != "float32" for x in args]):
            print("Only float32 inputs are supported for TensorRT.")
            return False
        if use_implicit_batch and int(attrs.axis) == 0:
            print("nn.softmax: can't modify batch dimension.")
            return False
        return True

    def conv2d_whitelist_fn(attrs, args):  # pylint: disable=unused-variable
        if any([x.checked_type.dtype != "float32" for x in args]):
            print("Only float32 inputs are supported for TensorRT.")
            return False
        if attrs.data_layout != "NCHW":
            print("nn.conv2d: data_layout is {} but must be NCHW.".format(attrs.data_layout))
            return False
        if attrs.kernel_layout != "OIHW":
            print("nn.conv2d: kernel_layout is {} but must be OIHW.".format(attrs.kernel_layout))
            return False
        if attrs.out_layout and attrs.out_layout != "NCHW":
            print("nn.conv2d: out_layout is {} but must be NCHW.".format(attrs.out_layout))
            return False
        return True

    def dense_whitelist_fn(attrs, args):  # pylint: disable=unused-variable
        if any([x.checked_type.dtype != "float32" for x in args]):
            print("Only float32 inputs are supported for TensorRT.")
            return False
        input_rank = len(args[0].checked_type.shape)
        weight_rank = len(args[1].checked_type.shape)
        if input_rank < 2 or input_rank > 4:
            print("nn.dense: input has rank {} but must be 2, 3 or 4.".format(input_rank))
            return False
        if weight_rank != 2:
            print("nn.dense: weight has rank {} but must be 2.".format(weight_rank))
            return False
        return True

    def bias_add_whitelist_fn(attrs, args):  # pylint: disable=unused-variable
        # TODO(trevmorr): BiasAddSimplifier creates a pattern which cannot be
        # converted to TRT without binding params and constant folding.
        # if trt_version < (6, 0, 1):
        #     return False
        if any([x.checked_type.dtype != "float32" for x in args]):
            print("Only float32 inputs are supported for TensorRT.")
            return False
        input_rank = len(args[0].checked_type.shape)
        if input_rank < 2 or input_rank > 4:
            print("nn.bias_add: input rank is {} but must be 2, 3 or 4.".format(input_rank))
            return False
        return True

    def max_pool_2d_whitelist_fn(attrs, args):  # pylint: disable=unused-variable
        if any([x.checked_type.dtype != "float32" for x in args]):
            print("Only float32 inputs are supported for TensorRT.")
            return False
        if attrs.layout != "NCHW":
            print("nn.max_pool2d: layout is {} but must be NCHW.".format(attrs.layout))
            return False
        if attrs.ceil_mode and trt_version < (5, 1, 5):
            print("nn.avg_pool2d: ceil_mode=True requires TensorRT 5.1.5 or greater.")
            return False
        return True

    def avg_pool_2d_whitelist_fn(attrs, args):  # pylint: disable=unused-variable
        if any([x.checked_type.dtype != "float32" for x in args]):
            print("Only float32 inputs are supported for TensorRT.")
            return False
        if attrs.layout != "NCHW":
            print("nn.avg_pool2d: layout is {} but must be NCHW.".format(attrs.layout))
            return False
        if (
            attrs.count_include_pad
            and len(attrs.padding) == 4
            and (
                int(attrs.padding[0]) != int(attrs.padding[2])
                or int(attrs.padding[1]) != int(attrs.padding[3])
            )
        ):
            print(
                "nn.avg_pool2d: inclusive-counted blended or average "
                "pooling is not supported in combination with asymmetric padding"
            )
            return False
        if attrs.ceil_mode and trt_version < (5, 1, 5):
            print("nn.avg_pool2d: ceil_mode=True requires TensorRT 5.1.5 or greater.")
            return False
        return True

    def global_max_pool_2d_whitelist_fn(attrs, args):  # pylint: disable=unused-variable
        if any([x.checked_type.dtype != "float32" for x in args]):
            print("Only float32 inputs are supported for TensorRT.")
            return False
        if attrs.layout != "NCHW":
            print("nn.global_max_pool2d: layout is {} but must be NCHW.".format(attrs.layout))
            return False
        return True

    def global_avg_pool_2d_whitelist_fn(attrs, args):  # pylint: disable=unused-variable
        if any([x.checked_type.dtype != "float32" for x in args]):
            print("Only float32 inputs are supported for TensorRT.")
            return False
        if attrs.layout != "NCHW":
            print("nn.global_avg_pool2d: layout is {} but must be NCHW.".format(attrs.layout))
            return False
        return True

    def expand_dims_whitelist_fn(attrs, args):  # pylint: disable=unused-variable
        if any([x.checked_type.dtype != "float32" for x in args]):
            print("Only float32 inputs are supported for TensorRT.")
            return False
        if use_implicit_batch and int(attrs.axis) == 0:
            print("expand_dims: can't modify batch dimension.")
            return False
        return True

    def squeeze_whitelist_fn(attrs, args):  # pylint: disable=unused-variable
        if any([x.checked_type.dtype != "float32" for x in args]):
            print("Only float32 inputs are supported for TensorRT.")
            return False
        if not attrs.axis:
            print("squeeze: must explicitly set axis.")
            return False
        if use_implicit_batch and any([axis == 0 for axis in map(int, attrs.axis)]):
            print("squeeze: can't modify batch dimension.")
            return False
        return True

    def concatenate_whitelist_fn(attrs, args):  # pylint: disable=unused-variable
        if any([x.dtype != "float32" for x in args[0].checked_type.fields]):
            print("Only float32 inputs are supported for TensorRT.")
            return False
        if not use_implicit_batch:
            return True
        if int(attrs.axis) == 0:
            print("concatenate: can't modify batch dimension.")
            return False
        if isinstance(args[0], Tuple):
            for tuple_input in args[0].fields:
                if isinstance(tuple_input, Constant):
                    print("concatenate: can't concatenate tensors with constants.")
                    return False
        return True

    def conv2d_transpose_whitelist_fn(attrs, args):  # pylint: disable=unused-variable
        if any([x.checked_type.dtype != "float32" for x in args]):
            print("Only float32 inputs are supported for TensorRT.")
            return False
        if attrs.data_layout != "NCHW":
            print(
                "nn.conv2d_transpose: data_layout is {} but must be NCHW.".format(attrs.data_layout)
            )
            return False
        if attrs.kernel_layout != "OIHW":
            print(
                "nn.conv2d_transpose: kernel_layout is {} but must be OIHW.".format(
                    attrs.kernel_layout
                )
            )
            return False
        if attrs.out_layout and attrs.out_layout != "NCHW":
            print(
                "nn.conv2d_transpose: out_layout is {} but must be NCHW.".format(attrs.out_layout)
            )
            return False
        if attrs.dilation and any([rate != 1 for rate in map(int, attrs.dilation)]):
            print("nn.conv2d_transpose: dilation rate must be 1.")
            return False
        return True

    def transpose_whitelist_fn(attrs, args):  # pylint: disable=unused-variable
        if any([x.checked_type.dtype != "float32" for x in args]):
            print("Only float32 inputs are supported for TensorRT.")
            return False
        if use_implicit_batch and int(attrs.axes[0]) != 0:
            print("transpose: can't modify batch dimension.")
            return False
        return True

    def reshape_whitelist_fn(attrs, args):  # pylint: disable=unused-variable
        if args[0].checked_type.dtype != "float32":
            print("Only float32 inputs are supported for TensorRT.")
            return False
        if any([x < -1 for x in map(int, attrs.newshape)]):
            print("reshape: new shape dims must be explicit.")
            return False
        if use_implicit_batch:
            shape = args[0].checked_type.shape
            new_shape = attrs.newshape
            if len(new_shape) == 0 or len(shape) == 0:
                print("reshape: Can't reshape to or from scalar.")
                return False

            dynamic_reshape = any([isinstance(x, tvm.tir.expr.Any) for x in shape])

            if dynamic_reshape:
                # Make sure that the batch dim is unmodified.
                if int(new_shape[0]) < 0:
                    for shape_val, new_shape_val in enumerate(shape[1:], new_shape[1:]):
                        if not (
                            isinstance(shape_val, int)
                            and isinstance(new_shape_val, int)
                            and int(shape_val) == int(new_shape_val)
                        ):
                            return False
                elif int(new_shape[0]) > 0:
                    if not (
                        isinstance(shape[0], int)
                        and isinstance(new_shape[0], int)
                        and int(shape[0]) == int(new_shape[0])
                    ):
                        return False
                return True
            else:
                shape = list(map(int, shape))
                new_shape = list(map(int, new_shape))

                # TRT cannot modify batch dimension.
                original_volume = np.prod(shape)
                # First, resolve 0.
                for i, value in enumerate(new_shape):
                    if value == 0:
                        new_shape[i] = shape[i]
                # Resolve -1.
                for i, value in enumerate(new_shape):
                    if value == -1:
                        new_shape[i] = original_volume // np.prod([x for x in new_shape if x != -1])
                # Remove batch dimension and see if volumes match
                if shape[0] != new_shape[0]:
                    print("reshape: can't modify batch dimension.")
                    return False
        return True

    def pad_whitelist_fn(attrs, args):  # pylint: disable=unused-variable
        if any([x.checked_type.dtype != "float32" for x in args]):
            print("Only float32 inputs are supported for TensorRT.")
            return False
        if attrs.pad_mode != "constant":
            print("nn.pad: pad mode is {} but must be constant.".format(attrs.pad_mode))
            return False
        if float(attrs.pad_value) != 0.0:
            print("nn.pad: pad value is {} but must be 0.0.".format(float(attrs.pad_value)))
            return False
        return True

    def reduce_whitelist_fn(attrs, args, op_name, trt_version):
        if not attrs.axis or len(attrs.axis) == 0:
            print("{}: cannot reduce to scalar.".format(op_name))
            return False
        if attrs.exclude:
            print("{}: exclude not supported.".format(op_name))
            return False
        if use_implicit_batch and any([x == 0 for x in map(int, attrs.axis)]):
            print("{}: can't modify batch dimension.".format(op_name))
            return False
        return True

    def trt_5_1_5_whitelist_fn(attrs, args, op_name, trt_version):
        if trt_version < (5, 1, 5):
            print("{}: requires TensorRT version 5.1.5 or higher.".format(op_name))
            return False
        return True

    _register_external_op_helper_func("sum", reduce_whitelist_fn, trt_version)
    _register_external_op_helper_func("prod", reduce_whitelist_fn, trt_version)
    _register_external_op_helper_func("max", reduce_whitelist_fn, trt_version)
    _register_external_op_helper_func("min", reduce_whitelist_fn, trt_version)
    _register_external_op_helper_func("mean", reduce_whitelist_fn, trt_version)
    _register_external_op_helper_func("nn.leaky_relu", trt_5_1_5_whitelist_fn, trt_version)
    _register_external_op_helper_func("sin", trt_5_1_5_whitelist_fn, trt_version)
    _register_external_op_helper_func("cos", trt_5_1_5_whitelist_fn, trt_version)
    _register_external_op_helper_func("atan", trt_5_1_5_whitelist_fn, trt_version)
    _register_external_op_helper_func("ceil", trt_5_1_5_whitelist_fn, trt_version)

    def strided_slice_whitelist_fn(attrs, args):  # pylint: disable=unused-variable
        if any([x.checked_type.dtype != "float32" for x in args]):
            print("Only float32 inputs are supported for TensorRT.")
            return False
        if trt_version < (5, 1, 5):
            print("strided_slice: requires TensorRT version 5.1.5 or higher.")
            return False
        if args[0].checked_type.dtype != "float32":
            print("strided_slice: only fp32 inputs are supported.")
            return False
        if use_implicit_batch:
            batch_dim_begin_modified = attrs.begin[0] is not None and int(attrs.begin[0]) != 0
            batch_dim_end_modified = (
                attrs.end[0] is not None
                and int(attrs.end[0]) != -1
                and int(attrs.end[0]) != int(args[0].checked_type.shape[0])
            )
            if batch_dim_begin_modified or batch_dim_end_modified:
                print("strided_slice: can't modify batch dimension.")
                return False
        if any([x is not None and x <= 0 for x in attrs.strides]):
            print("strided_slice: stride must be positive")
            return False
        for i in range(0, len(args[0].checked_type.shape)):
            begin = int(attrs.begin[i])
            end = (
                int(attrs.end[i])
                if attrs.end[i] is not None and int(attrs.end[i]) != -1
                else args[0].checked_type.shape[i]
            )
            if int(end) - int(begin) < 1:
                print("strided_slice: size of slice must be at least 1")
                return False
        return True

    def resize_whitelist_fn(attrs, args):  # pylint: disable=unused-variable
        # TODO(trevmorr): Output does not match TVM. Disable.
        return False

    def adapative_max_pool2d_whitelist_fn(attrs, args):  # pylint: disable=unused-variable
        if any([x.checked_type.dtype != "float32" for x in args]):
            print("Only float32 inputs are supported for TensorRT.")
            return False
        if len(attrs.output_size) == 0 or any([size != 1 for size in map(int, attrs.output_size)]):
            print("nn.adaptive_max_pool2d: output size must be (1, 1).")
            return False
        return True

    def adapative_avg_pool2d_whitelist_fn(attrs, args):  # pylint: disable=unused-variable
        if any([x.checked_type.dtype != "float32" for x in args]):
            print("Only float32 inputs are supported for TensorRT.")
            return False
        if len(attrs.output_size) == 0 or any([size != 1 for size in map(int, attrs.output_size)]):
            print("nn.adaptive_avg_pool2d: output size must be (1, 1).")
            return False
        return True

    def upsampling_whitelist_fn(attrs, args):  # pylint: disable=unused-variable
        # TODO(trevmorr): Output does not match TVM. Disable.
        return False

    def conv3d_whitelist_fn(attrs, args):  # pylint: disable=unused-variable
        if any([x.checked_type.dtype != "float32" for x in args]):
            print("Only float32 inputs are supported for TensorRT.")
            return False
        if trt_version < (6, 0, 1):
            print("nn.conv3d: requires TensorRT version 6.0.1 or higher.")
            return False
        if attrs.data_layout != "NCDHW":
            print("nn.conv3d: data_layout is {} but must be NCDHW.".format(attrs.data_layout))
            return False
        if attrs.kernel_layout != "OIDHW":
            print("nn.conv3d: kernel_layout is {} but must be OIDHW.".format(attrs.kernel_layout))
            return False
        if attrs.out_layout and attrs.out_layout != "NCDHW":
            print("nn.conv3d: out_layout is {} but must be NCDHW.".format(attrs.out_layout))
            return False
        return True

    def max_pool_3d_whitelist_fn(attrs, args):  # pylint: disable=unused-variable
        if any([x.checked_type.dtype != "float32" for x in args]):
            print("Only float32 inputs are supported for TensorRT.")
            return False
        if trt_version < (6, 0, 1):
            print("nn.max_pool3d: requires TensorRT version 6.0.1 or higher.")
            return False
        if attrs.layout != "NCDHW":
            print("nn.max_pool3d: layout is {} but must be NCDHW.".format(attrs.layout))
            return False
        return True

    def avg_pool_3d_whitelist_fn(attrs, args):  # pylint: disable=unused-variable
        if any([x.checked_type.dtype != "float32" for x in args]):
            print("Only float32 inputs are supported for TensorRT.")
            return False
        if trt_version < (6, 0, 1):
            print("nn.avg_pool3d: requires TensorRT version 6.0.1 or higher.")
            return False
        if attrs.layout != "NCDHW":
            print("nn.avg_pool3d: layout is {} but must be NCDHW.".format(attrs.layout))
            return False
        return True

    def conv3d_transpose_whitelist_fn(attrs, args):  # pylint: disable=unused-variable
        if any([x.checked_type.dtype != "float32" for x in args]):
            print("Only float32 inputs are supported for TensorRT.")
            return False
        if trt_version < (6, 0, 1):
            print("nn.conv3d_transpose: requires TensorRT version 6.0.1 or higher.")
            return False
        if attrs.data_layout != "NCDHW":
            print(
                "nn.conv3d_transpose: data_layout is {} but must be NCDHW.".format(
                    attrs.data_layout
                )
            )
            return False
        if attrs.kernel_layout != "OIDHW":
            print(
                "nn.conv3d_transpose: kernel_layout is {} but must be OIDHW.".format(
                    attrs.kernel_layout
                )
            )
            return False
        if attrs.out_layout and attrs.out_layout != "NCDHW":
            print(
                "nn.conv3d_transpose: out_layout is {} but must be NCDHW.".format(attrs.out_layout)
            )
            return False
        if attrs.dilation and any([rate != 1 for rate in map(int, attrs.dilation)]):
            print("nn.conv3d_transpose: dilation rate must be 1.")
            return False
        if attrs.output_padding and any([x != 0 for x in map(int, attrs.output_padding)]):
            print("nn.conv3d_transpose: output padding is not supported.")
            return False
        return True

    _register_external_dynamic_check_func("add", add_whitelist_fn)
    _register_external_dynamic_check_func("nn.batch_norm", batch_norm_whitelist_fn)
    _register_external_dynamic_check_func("nn.softmax", softmax_whitelist_fn)
    _register_external_dynamic_check_func("nn.conv2d", conv2d_whitelist_fn)
    _register_external_dynamic_check_func("nn.dense", dense_whitelist_fn)
    _register_external_dynamic_check_func("nn.bias_add", bias_add_whitelist_fn)
    _register_external_dynamic_check_func("nn.max_pool2d", max_pool_2d_whitelist_fn)
    _register_external_dynamic_check_func("nn.avg_pool2d", avg_pool_2d_whitelist_fn)
    _register_external_dynamic_check_func("nn.global_max_pool2d", global_max_pool_2d_whitelist_fn)
    _register_external_dynamic_check_func("nn.global_avg_pool2d", global_avg_pool_2d_whitelist_fn)
    _register_external_dynamic_check_func("expand_dims", expand_dims_whitelist_fn)
    _register_external_dynamic_check_func("squeeze", squeeze_whitelist_fn)
    _register_external_dynamic_check_func("concatenate", concatenate_whitelist_fn)
    _register_external_dynamic_check_func("nn.conv2d_transpose", conv2d_transpose_whitelist_fn)
    _register_external_dynamic_check_func("transpose", transpose_whitelist_fn)
    _register_external_dynamic_check_func("reshape", reshape_whitelist_fn)
    _register_external_dynamic_check_func("nn.pad", pad_whitelist_fn)
    _register_external_dynamic_check_func("strided_slice", strided_slice_whitelist_fn)
    _register_external_dynamic_check_func("image.resize", resize_whitelist_fn)
    _register_external_dynamic_check_func(
        "nn.adaptive_max_pool2d", adapative_max_pool2d_whitelist_fn
    )
    _register_external_dynamic_check_func(
        "nn.adaptive_avg_pool2d", adapative_avg_pool2d_whitelist_fn
    )
    _register_external_dynamic_check_func("nn.upsampling", upsampling_whitelist_fn)
    _register_external_dynamic_check_func("nn.conv3d", conv3d_whitelist_fn)
    _register_external_dynamic_check_func("nn.max_pool3d", max_pool_3d_whitelist_fn)
    _register_external_dynamic_check_func("nn.avg_pool3d", avg_pool_3d_whitelist_fn)
    _register_external_dynamic_check_func("nn.conv3d_transpose", conv3d_transpose_whitelist_fn)


class VarReplacer(ExprMutator):
    """
    Visit an expression while replacing vars according to var_map. Used by
    SubgraphRemover/PruneSubgraphs to return a subgraph originally partitioned to TRT back to TVM.
    """

    def __init__(self, var_map):
        ExprMutator.__init__(self)
        self.var_map = var_map

    def visit_var(self, var):
        if var in self.var_map:
            return self.var_map[var]
        return super().visit_var(var)


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
                new_body = VarReplacer(var_map).visit(func.body)
                return new_body
            elif name != "main":
                # Copy the GlobalVar (subgraph function) to the new module and call.
                args = []
                for arg in call.args:
                    args.append(super().visit(arg))
                return call.op(*args)

        return super().visit_call(call)


class IsComputeIntensiveGraph(ExprVisitor):
    """
    Visits the Graph recursively and checks if it contains compute heavy ops like convolutions and
    its transpose, dense and batch mat-mul.
    """

    def __init__(self):
        ExprVisitor.__init__(self)
        self.is_compute_intensive = False

    def visit_call(self, call):
        heavy_ops = set(
            [
                "nn.conv2d",
                "nn.conv2d_transpose",
                "nn.conv3d",
                "nn.conv3d_transpose",
                "nn.dense",
                "nn.batch_matmul",
            ]
        )
        if isinstance(call.op, tvm.tir.op.Op):
            if str(call.op) in heavy_ops:
                self.is_compute_intensive = True

        return super().visit_call(call)

    def is_graph_compute_intensive(self, subgraph):
        self.visit(subgraph)
        return self.is_compute_intensive


def PruneSubgraphs(mod, compiler="tensorrt", use_implicit_batch=True, prune_no_macs=False):
    """
    If use_implicit_batch is True, removes subgraphs which were originally partitioned for TRT
    that are incompatible with implicit batch mode.
    If prune_no_macs is True, also remove subgraph if the number of multiply-accumulates is 0.
    This is a heuristic which can improve performance by around 5% because TVM provides better
    optimization for certain ops.

     Parameters
    ----------
    mod: Module
        The module which has been partitioned for tensorrt compiler.

    compiler : str
        Compiler string, should be "tensorrt".

    use_implicit_batch : bool
        Which mode we plan to use for TensorRT. Will be used to determine which subgraphs are
        valid. In implicit batch mode, all inputs to a subgraph must have the same batch size.

    prune_no_macs : bool
        Whether to also remove subgraphs which have no multiple-accumulate operations.

    Returns
    -------
    mod: Module
        The modified module which has pruned subgraphs reverted back to TVM.
    """
    subgraphs_to_remove = []

    def is_valid_subgraph(func):
        """Whether a subgraph is valid in TRT.

        Returns
        -------
        compatible : bool
            True if the subgraph is compatible with TRT.
        """
        if not use_implicit_batch:
            return True
        input_batch_sizes = []
        for var in func.params:
            # In implicit batch mode, all inputs must have same batch size
            if isinstance(var.checked_type, relay.TupleType):
                for tupe_type in var.checked_type.fields:
                    # Scalar inputs not allowed
                    if len(tupe_type.shape) == 0:
                        return False
                    input_batch_sizes.append(int(tupe_type.shape[0]))
            else:
                # Scalar inputs not allowed
                if len(var.checked_type.shape) == 0:
                    return False
                if not isinstance(var.checked_type.shape[0], tvm.tir.expr.Any):
                    input_batch_sizes.append(int(var.checked_type.shape[0]))
        if len(input_batch_sizes) > 1 and any(
            [x != input_batch_sizes[0] for x in input_batch_sizes[1:]]
        ):
            return False
        return True

    # Remove invalid subgraphs
    for subgraph in mod.get_global_vars():
        name = subgraph.name_hint
        if (
            mod[name].attrs
            and hasattr(mod[name].attrs, "SkipOptimization")
            and mod[name].attrs["SkipOptimization"] == 1
        ):
            continue
        if not mod[name].attrs or mod[name].attrs["Compiler"] != compiler:
            continue
        if not is_valid_subgraph(mod[name]):
            subgraphs_to_remove.append(name)

    # Remove subgraphs with no multiply-accumulates
    if prune_no_macs:
        subgraph_with_compute_intensive_filter = []
        for subgraph in mod.get_global_vars():
            name = subgraph.name_hint
            if (
                mod[name].attrs
                and hasattr(mod[name].attrs, "SkipOptimization")
                and mod[name].attrs["SkipOptimization"] == 1
            ):
                continue
            if not mod[name].attrs or mod[name].attrs["Compiler"] != compiler:
                continue
            is_compute_intensive = IsComputeIntensiveGraph().is_graph_compute_intensive(mod[name])
            subgraph_with_compute_intensive_filter.append([name, is_compute_intensive])
        print("Subgraphs with compute heavy filter", subgraph_with_compute_intensive_filter)
        subgraphs_to_remove.extend(
            [
                name
                for name, is_compute_intensive in subgraph_with_compute_intensive_filter
                if not is_compute_intensive
            ]
        )
    if len(subgraphs_to_remove) == 0:
        return mod
    print("Will remove these subgraphs:", subgraphs_to_remove)

    # Create new pruned module with functions and type defns from mod
    new_mod = tvm.IRModule(mod.functions, mod.type_definitions)
    new_mod["main"] = SubgraphRemover(subgraphs_to_remove, mod, new_mod).visit(mod["main"])
    return new_mod


def EnableTrt(
    mod,
    params=None,
    trt_version=None,
    use_implicit_batch=True,
    max_workspace_size=1 << 30,
    prune_subgraphs=False,
):
    """Converts the "main" function in the module into one that can be executed using
    TensorRT. If any of the operators are not supported by the TensorRT
    conversion, the unmodified program will be returned instead.

    Parameters
    ----------
    mod: Module
        The original module.

    params : dict of str to NDArray
        Input parameters to the graph that do not change
        during inference time. Used for constant folding.

    trt_version : Optional[Tuple[int]]
        Which version of TensorRT to target for partitioning as a tuple of
        (major, minor, patch). If not specified, will attempt to get using
        GetTrtVersion.

    use_implicit_batch : bool
        If false, will use explicit batch mode. Explicit batch mode is
        available in TRT 6+. It increases operator coverage but comes at a
        performance penalty.

    max_workspace_size : int
        Number of bytes for TensorRT workspace size.

    prune_subgraphs : bool
        If true, will prune subgraphs with 0 MACS and run them with TVM instead.

    Returns
    -------
    mod: Module
        The modified module which will use the TensorRT runtime if compatible.
    """
    if not trt_version:
        trt_version = GetTrtVersion()
        # If TVM wasn't built against TRT, default to target TRT 6. Since the
        # actual conversion to TRT is done at runtime, building against TRT is
        # not required for compilation.
        if not trt_version:
            trt_version = (6, 0, 1)
    assert isinstance(trt_version, (list, tuple))
    assert len(trt_version) == 3

    register_tensorrt_annotations(trt_version, use_implicit_batch=use_implicit_batch)

    def _set_optimization_attr(mod, skip_optimization=1):
        """
        Prepare the mod such that all functions except main are tagged for SkipOptimization
        :param mod: input TRT mod
        :param skip_optimization: flag to set SkipOptimization for all functions except main
        :return: updated mod
        """
        gvs = mod.get_global_vars()
        for gv in gvs:
            func = mod[gv]
            name = gv.name_hint
            if name != "main":
                new_func = func.with_attr(
                    "SkipOptimization", tvm.tir.IntImm("int32", skip_optimization)
                )
                mod.update_func(gv, new_func)
        return mod

    if params:
        # Bind params so that we can use FoldConstant.
        mod["main"] = bind_params_by_name(mod["main"], params)
    # Apply passes required for TRT
    mod = transform.InferType()(mod)
    seq = tvm.transform.Sequential(
        [
            transform.InferType(),
            RemoveDropoutPass(),
            transform.RemoveUnusedFunctions(),
            transform.ConvertLayout(
                {
                    "nn.conv2d": ["NCHW", "default"],
                    "nn.conv2d_transpose": ["NCHW", "default"],
                    "nn.conv3d": ["NCDHW", "default"],
                }
            ),
            transform.FoldConstant(),
            LegalizeLayoutTranformPass(),
            transform.InferType(),
        ]
    )
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)

    # Set SkipOptimization for all functions but main for the following passes
    mod = _set_optimization_attr(mod, skip_optimization=1)
    seq = tvm.transform.Sequential(
        [
            transform.AnnotateTarget("tensorrt"),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
            transform.InferType(),
        ]
    )
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    mod = PruneSubgraphs(mod, use_implicit_batch=use_implicit_batch, prune_no_macs=prune_subgraphs)

    # Set SkipOptimization back to 0
    mod = _set_optimization_attr(mod, skip_optimization=0)

    # Set environment variables used to communicate with TensorRT module.
    os.environ["TVM_TENSORRT_MAX_WORKSPACE_SIZE"] = str(max_workspace_size)
    os.environ["TVM_TENSORRT_USE_IMPLICIT_BATCH"] = str(int(use_implicit_batch))
    return mod
