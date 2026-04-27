# 02 前端解析模块

## 目录

- [1. 模块概述](#1-模块概述)
- [2. 整体流程](#2-整体流程)
- [3. ONNX 解析器设计](#3-onnx-解析器设计)
- [4. PyTorch 前端设计](#4-pytorch-前端设计)
- [5. 算子映射表](#5-算子映射表)
- [6. 前端 → High-level IR 转换规则](#6-前端--high-level-ir-转换规则)
- [7. 错误处理策略](#7-错误处理策略)
- [8. 数据结构与接口](#8-数据结构与接口)

---

## 1. 模块概述

前端模块负责将外部模型格式（ONNX / PyTorch）解析并转换为项目内部的 **High-level IRModule**，作为后续图优化与 Lowering 的统一输入。

**语言**：Python 3  
**依赖库**：`onnx`（≥1.13）、`torch`（PyTorch 前端可选）  
**输出格式**：序列化的 IRModule（JSON 中间格式，由 C++ 反序列化读入）

---

## 2. 整体流程

```
ONNX 文件（.onnx）
      │
      ▼
 onnx.load() → ModelProto
      │
      │  解析 GraphProto
      ▼
  遍历 initializer（权重 → Constant 节点）
      │
  遍历 NodeProto（算子 → Call 节点）
      │
  遍历 ValueInfoProto（中间张量 → Var 节点 + TensorType）
      │
      ▼
  构建 IRModule（Function + Expr 树）
      │
      ▼
  序列化输出（JSON）

PyTorch nn.Module
      │
      ▼  torch.onnx.export()
 ONNX 文件（临时）
      │
      └─────▶ 走 ONNX 前端流程
```

---

## 3. ONNX 解析器设计

### 3.1 设计原则

- **单向数据流**：严格按拓扑顺序遍历图，构建 SSA 形式的 IR
- **名称→节点映射**：使用 `value_map: dict[str, Expr]` 维护 ONNX value name 到 IR `Expr` 的映射
- **延迟类型推断**：前端只记录显式 shape/dtype，缺失部分交由 IR 的类型推断 Pass 处理

### 3.2 核心数据结构

```python
class ONNXFrontend:
    """ONNX 模型解析器，将 ONNX ModelProto 转换为 High-level IRModule JSON。"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model: onnx.ModelProto = None
        # value_map: ONNX value name → IR Expr（序列化 dict）
        self.value_map: dict[str, dict] = {}
        # params: 权重名 → numpy array
        self.params: dict[str, np.ndarray] = {}

    def parse(self) -> dict:
        """解析入口，返回 IRModule 的 JSON 字典。"""
        ...

    def _parse_initializers(self, graph: onnx.GraphProto) -> None:
        """将 initializer（权重）转为 Constant Expr 节点。"""
        ...

    def _parse_node(self, node: onnx.NodeProto) -> dict:
        """将单个 NodeProto 转为 IR Call Expr。"""
        ...

    def _parse_type(self, type_proto: onnx.TypeProto) -> dict:
        """将 ONNX TypeProto 转为 IR TensorType JSON。"""
        ...
```

### 3.3 解析流程详解

#### 步骤 1：加载模型与合法性检查

```python
def parse(self) -> dict:
    self.model = onnx.load(self.model_path)
    onnx.checker.check_model(self.model)       # 合法性校验
    graph = self.model.graph
    self._infer_shapes(graph)                   # 调用 ONNX shape inference
    ...
```

#### 步骤 2：解析权重（Initializers）

ONNX `initializer` 中存储所有静态权重（卷积核、偏置等）。将每个 TensorProto 转为 `Constant` Expr 节点：

```python
def _parse_initializers(self, graph):
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)       # TensorProto → numpy
        dtype = _onnx_dtype_to_ir(init.data_type)
        shape = list(init.dims)
        self.value_map[init.name] = {
            "type": "Constant",
            "data": arr.tolist(),
            "tensor_type": {"shape": shape, "dtype": dtype}
        }
        self.params[init.name] = arr
```

#### 步骤 3：拓扑排序遍历算子节点

ONNX Graph 中 NodeProto 已按拓扑顺序排列，直接顺序遍历即可：

```python
for node in graph.node:
    ir_node = self._parse_node(node)
    for out_name in node.output:
        self.value_map[out_name] = ir_node
```

#### 步骤 4：构建输出 Function

以图的 `output` 列表为返回值，封装为 `Function` Expr：

```python
outputs = [self.value_map[o.name] for o in graph.output]
return_expr = outputs[0] if len(outputs) == 1 else {"type": "Tuple", "fields": outputs}
ir_func = {
    "type": "Function",
    "params": [...],          # 图输入 Var 列表
    "body": return_expr,
    "ret_type": ...
}
```

### 3.4 算子解析通用流程

```python
def _parse_node(self, node: onnx.NodeProto) -> dict:
    op_name = node.op_type
    handler = OP_REGISTRY.get(op_name)
    if handler is None:
        raise NotImplementedError(f"Unsupported op: {op_name}")

    # 收集输入 Expr
    inputs = [self.value_map[inp] for inp in node.input if inp]

    # 解析属性（如 kernel_shape, strides, pads）
    attrs = _parse_attributes(node.attribute)

    return handler(inputs, attrs)
```

---

## 4. PyTorch 前端设计

### 4.1 设计策略

PyTorch 前端不直接解析 `nn.Module`，而是利用 `torch.onnx.export` 将其导出为 ONNX，再经由 ONNX 解析器处理。这样可复用 ONNX 解析逻辑，降低维护成本。

```
nn.Module
    │
    ▼ torch.onnx.export(model, dummy_input, tmp_path,
    │                   opset_version=13,
    │                   do_constant_folding=False)
    ▼
tmp.onnx
    │
    ▼ ONNXFrontend(tmp_path).parse()
    ▼
IRModule JSON
```

### 4.2 接口设计

```python
class PyTorchFrontend:
    """PyTorch 模型前端，先转 ONNX 再解析。"""

    def __init__(self, module: torch.nn.Module, input_shapes: list[tuple]):
        """
        Args:
            module: PyTorch nn.Module，需处于 eval 模式
            input_shapes: 输入张量 shape 列表，如 [(1, 3, 224, 224)]
        """
        self.module = module.eval()
        self.input_shapes = input_shapes

    def parse(self) -> dict:
        """转换为 IRModule JSON。"""
        dummy_inputs = tuple(
            torch.zeros(shape, dtype=torch.float32)
            for shape in self.input_shapes
        )
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            tmp_path = f.name

        torch.onnx.export(
            self.module,
            dummy_inputs,
            tmp_path,
            opset_version=13,
            do_constant_folding=False,   # 保留常量以供 IR Pass 处理
            input_names=["input"],
            output_names=["output"],
        )
        result = ONNXFrontend(tmp_path).parse()
        os.unlink(tmp_path)              # 清理临时文件
        return result
```

### 4.3 注意事项

- `do_constant_folding=False`：关闭 ONNX 自带常量折叠，让编译器 Pass 自行处理，保留更多优化空间。
- `opset_version=13`：覆盖主流 CNN 算子，兼容性好。
- 导出前需调用 `model.eval()`，避免 BN/Dropout 训练模式干扰。

---

## 5. 算子映射表

下表列出支持的 ONNX 算子及其映射到 IR Op 的规则：

| ONNX Op | IR Op 名称 | 主要属性映射 | 说明 |
|---------|-----------|------------|------|
| `Conv` | `nn.conv2d` | `kernel_shape`, `strides`, `pads`, `dilations`, `group` | 支持 depthwise（group=C） |
| `BatchNormalization` | `nn.batch_norm` | `epsilon`, `momentum`（训练时忽略） | 推理时 weight/bias/mean/var 均为 Constant |
| `Relu` | `nn.relu` | — | 逐元素激活 |
| `MaxPool` | `nn.max_pool2d` | `kernel_shape`, `strides`, `pads` | 支持 2D |
| `AveragePool` | `nn.avg_pool2d` | `kernel_shape`, `strides`, `pads`, `count_include_pad` | 支持 2D |
| `GlobalAveragePool` | `nn.global_avg_pool2d` | — | 等价于 kernel=HW |
| `Gemm` | `nn.dense` | `transA`, `transB`, `alpha`, `beta` | 全连接层 |
| `MatMul` | `nn.matmul` | — | 矩阵乘 |
| `Add` | `nn.add` | — | 逐元素加（残差连接） |
| `Concat` | `nn.concatenate` | `axis` | 张量拼接 |
| `Flatten` | `nn.flatten` | `axis` | 展平 |
| `Reshape` | `nn.reshape` | `shape`（second input） | 动态 shape 需特殊处理 |
| `Clip` | `nn.clip` | `min`, `max` | 用于 ReLU6 |
| `Sigmoid` | `nn.sigmoid` | — | 逐元素 |
| `Softmax` | `nn.softmax` | `axis` | 分类输出层 |

> **不支持的算子处理**：遇到未注册算子时，前端抛出 `NotImplementedError`，并打印算子名称与所在模型路径，提示用户添加自定义 handler。

---

## 6. 前端 → High-level IR 转换规则

### 6.1 Var 节点（图输入）

ONNX `graph.input` 中不在 `initializer` 中的条目为真正的模型输入，转为 IR `Var` 节点：

```json
{
  "type": "Var",
  "name": "data",
  "type_annotation": {
    "type": "TensorType",
    "shape": [1, 3, 224, 224],
    "dtype": "float32"
  }
}
```

### 6.2 Constant 节点（权重）

权重转为 `Constant` 节点，内嵌 ndarray 数据（序列化为 base64 或列表）：

```json
{
  "type": "Constant",
  "tensor_type": {
    "shape": [64, 3, 3, 3],
    "dtype": "float32"
  },
  "data": "<base64_encoded_raw_bytes>"
}
```

### 6.3 Call 节点（算子）

每个算子转为 `Call` 节点，引用 `Op` 并附带属性：

```json
{
  "type": "Call",
  "op": { "type": "Op", "name": "nn.conv2d" },
  "args": ["<var_data_ref>", "<const_weight_ref>", "<const_bias_ref>"],
  "attrs": {
    "strides": [1, 1],
    "padding": [1, 1, 1, 1],
    "dilation": [1, 1],
    "groups": 1,
    "out_dtype": "float32"
  }
}
```

### 6.4 整体 IRModule 结构

```json
{
  "type": "IRModule",
  "functions": {
    "main": {
      "type": "Function",
      "params": [
        { "type": "Var", "name": "data", ... }
      ],
      "body": { ... },
      "ret_type": { "type": "TensorType", ... }
    }
  }
}
```

---

## 7. 错误处理策略

| 错误类型 | 处理方式 |
|---------|---------|
| 不支持的算子 | 抛出 `NotImplementedError`，打印 op 名称和模型位置 |
| ONNX 合法性失败 | `onnx.checker.check_model` 抛出异常，透传给用户 |
| 动态 shape | 前端标记为 `-1`（未知维度），后续 Pass 按符号维度处理 |
| 权重缺失 | 抛出 `KeyError`，提示 initializer 中不存在对应名称 |
| 精度不支持 | 仅支持 float32（data_type=1），其他精度抛出 `TypeError` |

---

## 8. 数据结构与接口

### 8.1 dtype 映射

```python
ONNX_DTYPE_MAP = {
    1:  "float32",   # FLOAT
    2:  "uint8",
    3:  "int8",
    5:  "int32",
    6:  "int64",
    7:  "string",
    10: "float16",
    11: "float64",
}
```

### 8.2 属性解析辅助函数

```python
def _parse_attributes(attrs: list) -> dict:
    """将 ONNX AttributeProto 列表转为 Python dict。"""
    result = {}
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
        # TENSOR 类型属性（如 Reshape 的 shape）
        elif attr.type == onnx.AttributeProto.TENSOR:
            result[attr.name] = numpy_helper.to_array(attr.t).tolist()
    return result
```

### 8.3 前端调用示例

```python
# ONNX 模型解析
frontend = ONNXFrontend("resnet18.onnx")
ir_module_json = frontend.parse()

# PyTorch 模型解析
import torchvision
model = torchvision.models.resnet18(pretrained=True)
frontend = PyTorchFrontend(model, input_shapes=[(1, 3, 224, 224)])
ir_module_json = frontend.parse()

# 序列化到文件，供 C++ 读取
import json
with open("model_ir.json", "w") as f:
    json.dump(ir_module_json, f)
```
