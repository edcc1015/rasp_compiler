import logging
import math
import onnx
from onnx import numpy_helper
from typing import Dict


# ONNX data_type int → IR dtype
ONNX_DTYPE_MAP: Dict[int, str] = {
    1:  "float32",
    2:  "uint8",
    3:  "int8",
    5:  "int32",
    6:  "int64",
    7:  "string",
    10: "float16",
    11: "float64",
}

def onnx_dtype_to_ir(data_type: int) -> str:
    if data_type not in ONNX_DTYPE_MAP:
        raise TypeError(f"Unsupported ONNX data type id: {data_type}")
    return ONNX_DTYPE_MAP[data_type]

# Attributes parser
def parse_attributes(attrs) -> dict:
    """Convert ONNX AttributeProto list to Python dict."""
    result: dict = {}
    for attr in attrs:
        if attr.type == onnx.AttributeProto.INT:
            result[attr.name] = attr.i
        elif attr.type == onnx.AttributeProto.FLOAT:
            result[attr.name] = attr.f
        elif attr.type == onnx.AttributeProto.INTS:
            result[attr.name] = list(attr.ints)
        elif attr.type == onnx.AttributeProto.FLOATS:
            result[attr.name] = list(attr.floats)
        elif attr.type == onnx.AttributeProto.STRING:
            result[attr.name] = attr.s.decode("utf-8")
        elif attr.type == onnx.AttributeProto.TENSOR:
            result[attr.name] = numpy_helper.to_array(attr.t).tolist()
    return result


# Operator Handler
def _make_call(op_name: str, args: list, attrs: dict) -> dict:
    return {
        "kind": "Call",
        "op": {"kind": "Op", "name": op_name},
        "args": args,
        "attrs": attrs,
    }


# ONNX Operator Handler：(inputs, attrs) → Call dict
def _handle_conv(inputs: list, attrs: dict) -> dict:
    ir_attrs = {
        "strides":   attrs.get("strides",   [1, 1]),
        "padding":   attrs.get("pads",      [0, 0, 0, 0]),
        "dilation":  attrs.get("dilations", [1, 1]),
        "groups":    attrs.get("group",     1),
        "out_dtype": "float32",
    }
    return _make_call("nn.conv2d", inputs, ir_attrs)


def _handle_batch_norm(inputs: list, attrs: dict) -> dict:
    # inputs: [X, scale, B, mean, var]；推理时后四项均为 Constant
    ir_attrs = {"epsilon": attrs.get("epsilon", 1e-5)}
    return _make_call("nn.batch_norm", inputs, ir_attrs)


def _handle_relu(inputs: list, attrs: dict) -> dict:
    return _make_call("nn.relu", inputs, {})


def _handle_max_pool(inputs: list, attrs: dict) -> dict:
    ir_attrs = {
        "pool_size": attrs.get("kernel_shape", [1, 1]),
        "strides":   attrs.get("strides",      [1, 1]),
        "padding":   attrs.get("pads",         [0, 0, 0, 0]),
    }
    return _make_call("nn.max_pool2d", inputs, ir_attrs)


def _handle_avg_pool(inputs: list, attrs: dict) -> dict:
    ir_attrs = {
        "pool_size":         attrs.get("kernel_shape",      [1, 1]),
        "strides":           attrs.get("strides",           [1, 1]),
        "padding":           attrs.get("pads",              [0, 0, 0, 0]),
        "count_include_pad": bool(attrs.get("count_include_pad", 0)),
    }
    return _make_call("nn.avg_pool2d", inputs, ir_attrs)


def _handle_global_avg_pool(inputs: list, attrs: dict) -> dict:
    return _make_call("nn.global_avg_pool2d", inputs, {})


def _handle_gemm(inputs: list, attrs: dict) -> dict:
    ir_attrs = {
        "transA": attrs.get("transA", 0),
        "transB": attrs.get("transB", 0),
        "alpha":  attrs.get("alpha",  1.0),
        "beta":   attrs.get("beta",   1.0),
    }
    return _make_call("nn.dense", inputs, ir_attrs)


def _handle_matmul(inputs: list, attrs: dict) -> dict:
    return _make_call("nn.matmul", inputs, {})


def _handle_add(inputs: list, attrs: dict) -> dict:
    return _make_call("nn.add", inputs, {})


def _handle_concat(inputs: list, attrs: dict) -> dict:
    return _make_call("nn.concatenate", inputs, {"axis": attrs.get("axis", 1)})


def _handle_flatten(inputs: list, attrs: dict) -> dict:
    return _make_call("nn.flatten", inputs, {"axis": attrs.get("axis", 1)})


def _handle_reshape(inputs: list, attrs: dict) -> dict:
    # opset >= 5：new shape created by the second input (Constant)
    return _make_call("nn.reshape", inputs, {})


def _handle_clip(inputs: list, attrs: dict) -> dict:
    # opset < 11：min/max in attrs；
    # opset >= 11：min/max in inputs[1], inputs[2]
    a_min = attrs.get("min", float("-inf"))
    a_max = attrs.get("max", float("inf"))
    if math.isinf(a_min) or math.isinf(a_max):
        msg = (
            f"Clip op has infinite bound(s): a_min={a_min}, a_max={a_max}. "
            "For opset>=11, min/max must come from scalar Constant inputs "
            "(inputs[1]/inputs[2]), not from attrs. "
            "JSON cannot represent Infinity — serialization would produce invalid output."
        )
        logging.error(msg)
        raise ValueError(msg)
    ir_attrs = {"a_min": a_min, "a_max": a_max}
    return _make_call("nn.clip", inputs, ir_attrs)


def _handle_sigmoid(inputs: list, attrs: dict) -> dict:
    return _make_call("nn.sigmoid", inputs, {})


def _handle_softmax(inputs: list, attrs: dict) -> dict:
    return _make_call("nn.softmax", inputs, {"axis": attrs.get("axis", 1)})


# ONNX op_type → handler func
OP_REGISTRY: dict = {
    "Conv":               _handle_conv,
    "BatchNormalization": _handle_batch_norm,
    "Relu":               _handle_relu,
    "MaxPool":            _handle_max_pool,
    "AveragePool":        _handle_avg_pool,
    "GlobalAveragePool":  _handle_global_avg_pool,
    "Gemm":               _handle_gemm,
    "MatMul":             _handle_matmul,
    "Add":                _handle_add,
    "Concat":             _handle_concat,
    "Flatten":            _handle_flatten,
    "Reshape":            _handle_reshape,
    "Clip":               _handle_clip,
    "Sigmoid":            _handle_sigmoid,
    "Softmax":            _handle_softmax,
}