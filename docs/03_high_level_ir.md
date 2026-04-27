# 03 高层 IR 设计（High-level IR）

## 目录

- [1. 设计目标与定位](#1-设计目标与定位)
- [2. 与 TVM Relay 的对比](#2-与-tvm-relay-的对比)
- [3. 核心数据结构总览](#3-核心数据结构总览)
- [4. IRNode 基类](#4-irnode-基类)
- [5. 类型系统](#5-类型系统)
- [6. 表达式节点（Expr）](#6-表达式节点expr)
- [7. Op 注册系统](#7-op-注册系统)
- [8. IRModule](#8-irmodule)
- [9. C++ 类继承关系图](#9-c-类继承关系图)
- [10. 内存管理策略](#10-内存管理策略)
- [11. 访问者模式（Visitor Pattern）](#11-访问者模式visitor-pattern)
- [12. 序列化与反序列化](#12-序列化与反序列化)

---

## 1. 设计目标与定位

High-level IR（后文简称 **HLIR**）是编译器的核心数据结构，承担以下职责：

1. **统一表示**：将不同前端（ONNX/PyTorch）解析得到的计算图统一表示为一套代数数据结构。
2. **图级优化**：作为常量折叠、算子融合、死代码消除、布局变换等 Pass 的操作对象。
3. **Lowering 输入**：为 Lowering 模块提供算子语义信息，驱动循环级 IR 的生成。

HLIR 采用**纯函数式**（functional）风格：表达式树由不可变节点构成，变换通过重写（rewrite）而非原地修改实现。

---

## 2. 与 TVM Relay 的对比

| 特性 | TVM Relay | RASP HLIR | 说明 |
|------|-----------|-----------|------|
| IR 风格 | 函数式 | 函数式 | 一致 |
| 类型系统 | 完整（包含 TypeVar） | 精简（TensorType + TupleType + FuncType） | 去除了泛型类型变量 |
| Let 绑定 | 支持 | 支持 | 用于名称绑定和 CSE |
| 算子定义 | TOPI + 自定义 | OpRegistry（精简） | 只注册 CNN 算子 |
| ADT / 递归 | 支持 | **不支持** | 精简设计，仅推理图 |
| 自动微分 | 支持 | **不支持** | 仅推理 |
| 动态 Shape | 部分支持 | 静态 Shape 为主 | 未知维度用 -1 表示 |
| 序列化 | protobuf / JSON | JSON | 实现简单 |

---

## 3. 核心数据结构总览

```
IRModule
    └── map<string, Function>       ← 具名函数（通常只有 "main"）

Function : Expr
    ├── params: vector<Var>         ← 函数参数
    ├── body: Expr                  ← 函数体（表达式树）
    └── ret_type: Type

Expr（表达式基类）
    ├── Var             ← 变量引用（函数参数 / Let 绑定变量）
    ├── Constant        ← 常量张量
    ├── Call            ← 算子调用，op(arg0, arg1, ..., attrs)
    ├── Let             ← Let 绑定：let var = value in body
    ├── Tuple           ← 元组构造
    ├── TupleGetItem    ← 元组索引
    └── Function        ← 匿名函数（用于 FusedOp）

Type（类型基类）
    ├── TensorType      ← 形状 + 数据类型
    ├── TupleType       ← 多类型组合
    └── FuncType        ← 函数签名类型

Op                      ← 算子描述符（名称 + 属性 Schema）
```

---

## 4. IRNode 基类

所有 IR 节点均继承自 `IRNode`，提供统一的节点类型标识与引用计数内存管理。

```cpp
// include/ir/ir_node.h

enum class IRNodeType {
    // Type 节点
    kTensorType,
    kTupleType,
    kFuncType,
    // Expr 节点
    kVar,
    kConstant,
    kCall,
    kLet,
    kTuple,
    kTupleGetItem,
    kFunction,
    // Module
    kIRModule,
    // Op
    kOp,
};

class IRNode {
public:
    virtual ~IRNode() = default;
    virtual IRNodeType node_type() const = 0;

    // 禁止拷贝，通过 shared_ptr 管理
    IRNode(const IRNode&) = delete;
    IRNode& operator=(const IRNode&) = delete;

protected:
    IRNode() = default;
};

// 所有 IR 对象通过 shared_ptr 传递
template <typename T>
using Ref = std::shared_ptr<T>;
```

---

## 5. 类型系统

类型系统描述每个 `Expr` 节点的输出数据特征，为类型检查、形状推断和 Lowering 提供基础。

### 5.1 DataType

```cpp
// include/ir/type.h

enum class DataType {
    kFloat32,
    kFloat16,
    kInt32,
    kInt64,
    kInt8,
    kUInt8,
    kBool,
};

std::string dtype_to_string(DataType dt);   // "float32", "int32", ...
DataType dtype_from_string(const std::string& s);
int dtype_bytes(DataType dt);               // sizeof
```

### 5.2 TensorType

描述静态形状张量：

```cpp
class TensorType : public IRNode {
public:
    // shape[i] == -1 表示该维度动态未知
    std::vector<int64_t> shape;
    DataType dtype;

    IRNodeType node_type() const override { return IRNodeType::kTensorType; }

    // 工厂函数
    static Ref<TensorType> make(std::vector<int64_t> shape, DataType dtype);

    // 辅助接口
    int64_t ndim() const { return shape.size(); }
    int64_t numel() const;  // 元素总数（有动态维度时返回 -1）
};
```

**示例**：

```cpp
auto t = TensorType::make({1, 64, 56, 56}, DataType::kFloat32);
// shape = [1, 64, 56, 56], dtype = float32
```

### 5.3 TupleType

多输出算子（如某些 ONNX 节点）的返回类型：

```cpp
class TupleType : public IRNode {
public:
    std::vector<Ref<IRNode>> fields;  // 每个 field 是 TensorType 或 TupleType

    IRNodeType node_type() const override { return IRNodeType::kTupleType; }
    static Ref<TupleType> make(std::vector<Ref<IRNode>> fields);
};
```

### 5.4 FuncType

函数（`Function` Expr）的类型签名：

```cpp
class FuncType : public IRNode {
public:
    std::vector<Ref<IRNode>> arg_types;   // 参数类型列表
    Ref<IRNode>              ret_type;    // 返回类型

    IRNodeType node_type() const override { return IRNodeType::kFuncType; }
    static Ref<FuncType> make(std::vector<Ref<IRNode>> arg_types, Ref<IRNode> ret_type);
};
```

---

## 6. 表达式节点（Expr）

所有 Expr 节点携带一个 `checked_type` 字段，在类型推断 Pass 后被填充。

```cpp
// include/ir/expr.h

class Expr : public IRNode {
public:
    // 类型推断后填充；初始为 nullptr
    mutable Ref<IRNode> checked_type;
};
```

### 6.1 Var —— 变量

```cpp
class Var : public Expr {
public:
    std::string        name;         // 变量名（调试用）
    Ref<IRNode>        type_annotation;  // 可选；前端直接设置 TensorType

    IRNodeType node_type() const override { return IRNodeType::kVar; }
    static Ref<Var> make(std::string name, Ref<IRNode> type_annotation = nullptr);
};
```

**用途**：
- `Function::params` 中的参数变量
- `Let` 绑定中的绑定变量

### 6.2 Constant —— 常量张量

```cpp
class Constant : public Expr {
public:
    // 原始数据（行主序，float32）
    std::vector<float>   data;
    Ref<TensorType>      tensor_type;   // shape + dtype

    IRNodeType node_type() const override { return IRNodeType::kConstant; }
    static Ref<Constant> make(std::vector<float> data, Ref<TensorType> ttype);
};
```

### 6.3 Call —— 算子调用

Call 节点是 HLIR 中最核心的节点，表示"对某个 Op 的一次调用"：

```cpp
class Attrs {
public:
    // 通用属性存储；各 Op 在解析时填充对应字段
    std::unordered_map<std::string, std::variant<
        int64_t,
        double,
        std::string,
        std::vector<int64_t>,
        std::vector<double>
    >> values;

    template <typename T>
    T get(const std::string& key) const;

    template <typename T>
    T get_or(const std::string& key, T default_val) const;

    bool has(const std::string& key) const;
};

class Call : public Expr {
public:
    Ref<IRNode>          op;    // 指向 Op 或 Function（用于内联融合函数）
    std::vector<Ref<Expr>> args; // 输入参数列表
    Attrs                attrs; // 算子属性

    IRNodeType node_type() const override { return IRNodeType::kCall; }
    static Ref<Call> make(Ref<IRNode> op,
                          std::vector<Ref<Expr>> args,
                          Attrs attrs = {});
};
```

**示例**（Conv2D 调用）：

```cpp
// conv2d(data, weight, bias, strides=[1,1], padding=[1,1,1,1])
Attrs conv_attrs;
conv_attrs.values["strides"]  = std::vector<int64_t>{1, 1};
conv_attrs.values["padding"]  = std::vector<int64_t>{1, 1, 1, 1};
conv_attrs.values["dilation"] = std::vector<int64_t>{1, 1};
conv_attrs.values["groups"]   = int64_t{1};

auto call = Call::make(
    OpRegistry::get("nn.conv2d"),
    {data_var, weight_const, bias_const},
    conv_attrs
);
```

### 6.4 Let —— Let 绑定

```cpp
class Let : public Expr {
public:
    Ref<Var>  var;    // 绑定的变量名
    Ref<Expr> value;  // 绑定的值
    Ref<Expr> body;   // 使用该变量的后续表达式

    IRNodeType node_type() const override { return IRNodeType::kLet; }
    static Ref<Let> make(Ref<Var> var, Ref<Expr> value, Ref<Expr> body);
};
```

**用途**：CSE（公共子表达式消除）将重复子表达式绑定为 Let，避免重复计算。

### 6.5 Tuple / TupleGetItem

```cpp
class Tuple : public Expr {
public:
    std::vector<Ref<Expr>> fields;

    IRNodeType node_type() const override { return IRNodeType::kTuple; }
    static Ref<Tuple> make(std::vector<Ref<Expr>> fields);
};

class TupleGetItem : public Expr {
public:
    Ref<Expr> tuple;   // 被索引的元组
    int       index;   // 索引位置

    IRNodeType node_type() const override { return IRNodeType::kTupleGetItem; }
    static Ref<TupleGetItem> make(Ref<Expr> tuple, int index);
};
```

### 6.6 Function —— 函数（含融合函数）

```cpp
class Function : public Expr {
public:
    std::vector<Ref<Var>> params;
    Ref<Expr>             body;
    Ref<IRNode>           ret_type;   // FuncType 或具体 Type
    // 可选属性标记，如 { "Primitive": 1 } 表示这是融合后的原始函数
    std::unordered_map<std::string, std::string> attrs;

    IRNodeType node_type() const override { return IRNodeType::kFunction; }
    static Ref<Function> make(std::vector<Ref<Var>> params,
                               Ref<Expr> body,
                               Ref<IRNode> ret_type = nullptr,
                               std::unordered_map<std::string, std::string> attrs = {});

    bool is_primitive() const {
        auto it = attrs.find("Primitive");
        return it != attrs.end() && it->second == "1";
    }
};
```

---

## 7. Op 注册系统

`Op` 是一个轻量级的算子描述符，存储于全局 `OpRegistry`。

### 7.1 Op 描述符

```cpp
// include/ir/op.h

struct AttrSchema {
    std::string name;
    std::string type;      // "int", "float", "ints", "floats", "string"
    bool        required;
    std::string default_value;
};

class Op : public IRNode {
public:
    std::string              name;         // 如 "nn.conv2d"
    std::string              description;
    std::vector<AttrSchema>  attr_schema;  // 属性定义列表

    IRNodeType node_type() const override { return IRNodeType::kOp; }
};
```

### 7.2 OpRegistry

```cpp
class OpRegistry {
public:
    static void register_op(const std::string& name,
                             std::vector<AttrSchema> schema,
                             const std::string& desc = "");

    static Ref<Op> get(const std::string& name);  // 不存在则 throw

    static bool has(const std::string& name);

    static std::vector<std::string> list_all();

private:
    static std::unordered_map<std::string, Ref<Op>>& registry();
};
```

**注册示例**：

```cpp
// 在 src/ir/op.cpp 中完成所有算子的注册
OpRegistry::register_op("nn.conv2d", {
    {"strides",  "ints",   false, "[1,1]"},
    {"padding",  "ints",   false, "[0,0,0,0]"},
    {"dilation", "ints",   false, "[1,1]"},
    {"groups",   "int",    false, "1"},
    {"out_dtype","string", false, "float32"},
}, "2D Convolution");

OpRegistry::register_op("nn.relu",   {}, "ReLU activation");
OpRegistry::register_op("nn.batch_norm", {
    {"epsilon",  "float", false, "1e-5"},
}, "Batch Normalization");
// ...
```

---

## 8. IRModule

`IRModule` 是整个模型的顶层容器，管理所有具名函数：

```cpp
// include/ir/ir_module.h

class IRModule : public IRNode {
public:
    // 函数名 → Function Expr
    std::unordered_map<std::string, Ref<Function>> functions;

    IRNodeType node_type() const override { return IRNodeType::kIRModule; }

    static Ref<IRModule> make();

    void add_function(const std::string& name, Ref<Function> func);
    Ref<Function> get_function(const std::string& name) const;
    bool has_function(const std::string& name) const;

    // 序列化
    std::string to_json() const;
    static Ref<IRModule> from_json(const std::string& json_str);
};
```

---

## 9. C++ 类继承关系图

```
IRNode
├── Type
│   ├── TensorType
│   ├── TupleType
│   └── FuncType
├── Expr
│   ├── Var
│   ├── Constant
│   ├── Call
│   ├── Let
│   ├── Tuple
│   ├── TupleGetItem
│   └── Function
├── Op
└── IRModule
```

所有节点均通过 `Ref<T>` （即 `std::shared_ptr<T>`）持有，提供自动引用计数内存管理。

---

## 10. 内存管理策略

### 10.1 共享所有权（shared_ptr）

所有 IR 节点使用 `std::shared_ptr` 管理，允许多处引用同一节点（如被多个 Call 共享的 Constant），无需手动管理生命周期。

```cpp
// 节点共享示例
auto weight = Constant::make(data, ttype);  // 只分配一次

// conv1 和 conv2 可以共享同一个 weight（如 tied weights）
auto call1 = Call::make(op, {data1, weight}, attrs);
auto call2 = Call::make(op, {data2, weight}, attrs);
```

### 10.2 节点不可变（Immutable）

IR 节点构造后字段不可修改（公共字段可读，通过工厂函数创建）。Pass 通过 **Mutator**（见 [11 访问者模式](#11-访问者模式visitor-pattern)）构造新节点实现变换，旧节点的引用计数自然归零后被释放。

### 10.3 弱引用避免循环（weak_ptr）

HLIR 是 DAG（有向无环图），理论上不存在引用环。但若工具层（如调试器）需要向上追踪父节点，使用 `std::weak_ptr` 存储反向边，避免循环引用。

---

## 11. 访问者模式（Visitor Pattern）

所有对 IR 的遍历与变换通过 **访问者** 实现，避免在节点类中堆积逻辑。

### 11.1 ExprVisitor（只读遍历）

```cpp
class ExprVisitor {
public:
    virtual void visit(Ref<Expr> expr);

    virtual void visit_var(Ref<Var> node)             {}
    virtual void visit_constant(Ref<Constant> node)   {}
    virtual void visit_call(Ref<Call> node);           // 默认递归访问 args
    virtual void visit_let(Ref<Let> node);             // 默认递归访问 value + body
    virtual void visit_tuple(Ref<Tuple> node);
    virtual void visit_tuple_get_item(Ref<TupleGetItem> node);
    virtual void visit_function(Ref<Function> node);
};
```

### 11.2 ExprMutator（带变换的遍历）

Pass 通过继承 `ExprMutator` 并重写特定 `mutate_xxx` 方法实现节点替换：

```cpp
class ExprMutator {
public:
    // 入口：对整个 Expr 树应用变换，返回新树
    virtual Ref<Expr> mutate(Ref<Expr> expr);

    virtual Ref<Expr> mutate_var(Ref<Var> node)            { return node; }
    virtual Ref<Expr> mutate_constant(Ref<Constant> node)  { return node; }
    virtual Ref<Expr> mutate_call(Ref<Call> node);         // 默认：递归 mutate args
    virtual Ref<Expr> mutate_let(Ref<Let> node);
    virtual Ref<Expr> mutate_tuple(Ref<Tuple> node);
    virtual Ref<Expr> mutate_tuple_get_item(Ref<TupleGetItem> node);
    virtual Ref<Expr> mutate_function(Ref<Function> node);
};
```

**Pass 示例（折叠 ReLU(Constant) → Constant）**：

```cpp
class ConstantFoldingMutator : public ExprMutator {
    Ref<Expr> mutate_call(Ref<Call> node) override {
        // 先递归处理子节点
        auto new_node = ExprMutator::mutate_call(node);
        auto call = std::static_pointer_cast<Call>(new_node);

        // 若所有参数均为 Constant，则在编译期求值
        if (all_constant(call->args)) {
            return eval_constant_call(call);
        }
        return call;
    }
};
```

---

## 12. 序列化与反序列化

HLIR 提供 JSON 序列化，用于前端（Python）→ C++ 之间的模型传递。

### 12.1 序列化格式（节选）

节点类型通过 `"kind"` 字段区分：

```json
{
  "kind": "IRModule",
  "functions": {
    "main": {
      "kind": "Function",
      "params": [
        { "kind": "Var", "name": "data",
          "type": { "kind": "TensorType", "shape": [1,3,224,224], "dtype": "float32" } }
      ],
      "body": {
        "kind": "Call",
        "op": { "kind": "Op", "name": "nn.conv2d" },
        "args": [
          { "kind": "Var", "name": "data" },
          { "kind": "Constant", "shape": [64,3,3,3], "dtype": "float32", "data": "..." }
        ],
        "attrs": {
          "strides": [1, 1],
          "padding": [1, 1, 1, 1],
          "groups": 1
        }
      },
      "ret_type": { "kind": "TensorType", "shape": [1,64,224,224], "dtype": "float32" }
    }
  }
}
```

### 12.2 反序列化入口

```cpp
// 读取前端生成的 JSON，构建 C++ IRModule
Ref<IRModule> mod = IRModule::from_json(json_str);
```

反序列化按 `"kind"` 字段分发到对应构造逻辑，递归重建完整节点树。
