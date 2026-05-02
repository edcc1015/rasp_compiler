import base64
import numpy as np
import onnx
from onnx import numpy_helper

from .onnx_config import *


class ONNXFrontend:
    """ONNX Parser. ONNX ModelProto -> High-level IRModule JSON."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model: onnx.ModelProto = None
        self.value_map: dict[str, dict] = {}    # ONNX value name → IR Expr
        self.params: dict[str, np.ndarray] = {} # wgt name → numpy array

    # Public Interface
    def parse(self) -> dict:
        """Return IRModule json dict."""
        self.model = onnx.load(self.model_path)
        onnx.checker.check_model(self.model)
        self._infer_shapes()  # Update self.model shape/dtype info
        graph = self.model.graph

        # Step 1：parse weight (initializer → Constant)
        self._parse_initializers(graph)

        # Step 2：parse real model input (non-weight graph.input → Var)
        initializer_names = {init.name for init in graph.initializer}
        input_vars: list[dict] = []
        for inp in graph.input:
            if inp.name in initializer_names:
                continue
            var_node = {
                "kind": "Var",
                "name": inp.name,
                "type": self._parse_type(inp.type),
            }
            self.value_map[inp.name] = var_node
            input_vars.append(var_node)

        # Step 3：topo visit
        # The ONNX graph.nodes have been arranged in topological order.
        for node in graph.node:
            ir_node = self._parse_node(node)
            if len(node.output) == 1:
                self.value_map[node.output[0]] = ir_node
            else:
                # Multiple output nodes: split with TupleGetItem
                for idx, out_name in enumerate(node.output):
                    if out_name:
                        self.value_map[out_name] = {
                            "kind": "TupleGetItem",
                            "tuple": ir_node,
                            "index": idx,
                        }

        # Step 4：Create return and function
        outputs = [self.value_map[o.name] for o in graph.output]
        if len(outputs) == 1:
            return_expr = outputs[0]
            ret_type = self._parse_type(graph.output[0].type)
        else:
            return_expr = {"kind": "Tuple", "fields": outputs}
            ret_type = {
                "kind": "TupleType",
                "fields": [self._parse_type(o.type) for o in graph.output],
            }

        ir_func = {
            "kind": "Function",
            "params": input_vars,
            "body": return_expr,
            "ret_type": ret_type,
        }

        return {
            "kind": "IRModule",
            "functions": {"main": ir_func},
        }

    # private util func
    def _infer_shapes(self) -> None:
        # Call ONNX's built-in shape inference and write the results back to self.model.
        self.model = onnx.shape_inference.infer_shapes(self.model)

    def _parse_initializers(self, graph) -> None:
        # Convert graph.initializer (weight tensor) to Constant Expr nodes.
        for init in graph.initializer:
            arr = numpy_helper.to_array(init)
            dtype = onnx_dtype_to_ir(init.data_type)
            shape = list(init.dims)
            # The raw bytes are serialized to base64
            # consistent with the C++deserialization convention
            data_b64 = base64.b64encode(arr.tobytes()).decode("ascii")
            self.value_map[init.name] = {
                "kind": "Constant",
                "shape": shape,
                "dtype": dtype,
                "data": data_b64,
            }
            self.params[init.name] = arr

    def _parse_node(self, node: onnx.NodeProto) -> dict:
        # Convert a single NodeProto to an IR Call Expr.
        op_type = node.op_type
        handler = OP_REGISTRY.get(op_type)
        if handler is None:
            raise NotImplementedError(
                f"Unsupported op: {op_type!r} (model: {self.model_path!r}). "
                "Register a custom handler in OP_REGISTRY to add support."
            )

        # Empty string indicates missing optional input, skip
        inputs = [self.value_map[inp] for inp in node.input if inp]
        attrs = parse_attributes(node.attribute)
        return handler(inputs, attrs)

    def _parse_type(self, type_proto) -> dict:
        # Convert ONNX TypeProto to IR TensorType JSON
        tensor_type = type_proto.tensor_type
        dtype = (
            onnx_dtype_to_ir(tensor_type.elem_type)
            if tensor_type.elem_type != 0
            else "float32"
        )
        shape: list[int] = []
        if tensor_type.HasField("shape"):
            for dim in tensor_type.shape.dim:
                # When dim_ralue==0 and there is no dim_maram
                # it is considered a dynamic dimension
                if dim.HasField("dim_value") and dim.dim_value > 0:
                    shape.append(dim.dim_value)
                else:
                    shape.append(-1)
        return {"kind": "TensorType", "shape": shape, "dtype": dtype}
    
if __name__ == "__main__":
    frontend = ONNXFrontend("tests/frontend/tiny_smoke.onnx")
    ir_module_json = frontend.parse()

    import json
    with open("model_ir.json", "w") as f:
        json.dump(ir_module_json, f)
