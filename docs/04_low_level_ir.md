# 04 低层 IR 设计（Low-level IR）

## 目录

- [1. 设计目标与定位](#1-设计目标与定位)
- [2. 与 TVM TIR 的对比](#2-与-tvm-tir-的对比)
- [3. 核心数据结构总览](#3-核心数据结构总览)
- [4. DataType 与 Var](#4-datatype-与-var)
- [5. Buffer —— 多维内存描述符](#5-buffer--多维内存描述符)
- [6. PrimExpr —— 基本表达式](#6-primexpr--基本表达式)
- [7. Stmt —— 语句节点](#7-stmt--语句节点)
- [8. PrimFunc —— 低层函数](#8-primfunc--低层函数)
- [9. LLIRModule](#9-llirmodule)
- [10. C++ 类继承关系图](#10-c-类继承关系图)
- [11. 访问者模式](#11-访问者模式)
- [12. 设计示例：Conv2D PrimFunc 结构](#12-设计示例conv2d-primfunc-结构)

---

## 1. 设计目标与定位

Low-level IR（后文简称 **LLIR**）是编译器的**循环级**中间表示，描述算子的具体执行方式：

- 以**多重嵌套 For 循环**、**Buffer 读写**为基本单元
- 直接映射到 C++ 循环与数组访问，易于 Codegen
- 承载 **Loop Tiling** 等循环变换 Pass
- 支持向量化循环（Vectorize）注解，指导 NEON Intrinsics 生成

与 HLIR 的关键区别：

| 维度 | High-level IR | Low-level IR |
|------|--------------|-------------|
| 抽象层次 | 算子语义（做什么） | 执行细节（怎么做） |
| 控制流 | 无（纯数据流图） | 显式 For / IfThenElse |
| 数据访问 | 隐式（参数传递） | 显式 Buffer 读写（BufferLoad/Store） |
| 内存分配 | 无 | Allocate 语句 |
| 向量化 | 无 | ForKind::Vectorized |

---

## 2. 与 TVM TIR 的对比

| 特性 | TVM TIR | RASP LLIR | 说明 |
|------|---------|-----------|------|
| 语句节点 | 全套（Block、BufferRealize 等） | 精简（For, SeqStmt, IfThenElse, Allocate, BufferStore） | 去掉 Block 抽象，结构更直接 |
| For 类型 | Serial/Vectorize/Unroll/Parallel/ThreadBinding | Serial/Vectorize/Unroll（仅 CPU） | 不含 GPU 线程 |
| 表达式 | 全套 PrimExpr | 精简（基本算术 + Load + Ramp + Broadcast） | 去掉复杂类型 |
| Schedule | TE Schedule / Block Schedule | **不使用 Schedule 对象**，Pass 直接变换 Stmt 树 | 降低复杂度 |
| 内省 / 分析 | 完整 IR Analysis | 按需实现（如 LoopVarRange） | 按需扩展 |

---

## 3. 核心数据结构总览

```
LLIRModule
    └── map<string, PrimFunc>

PrimFunc
    ├── params: vector<Buffer>        ← 函数参数（Buffer 描述符）
    └── body:   Stmt                  ← 函数体（语句树）

Stmt（语句基类）
    ├── SeqStmt            ← 顺序执行语句列表
    ├── For                ← for 循环（含 ForKind：Serial/Vectorized/Unrolled）
    ├── IfThenElse         ← 条件分支
    ├── Allocate           ← 局部 Buffer 分配
    ├── BufferStore        ← Buffer 写入
    └── Evaluate           ← 表达式作为语句（含 extern call）

PrimExpr（基本表达式基类，标量/向量）
    ├── IntImm             ← 整数常量
    ├── FloatImm           ← 浮点常量
    ├── Var                ← 循环变量 / 临时标量
    ├── BufferLoad         ← Buffer 读取
    ├── Add / Sub / Mul / Div / Mod
    ├── FloorDiv / FloorMod
    ├── Min / Max
    ├── Select             ← 三目表达式
    ├── Cast               ← 类型转换
    ├── Ramp               ← 向量化索引（base + stride * lanes）
    ├── Broadcast          ← 标量广播为向量
    └── Call               ← 内置函数 / NEON Intrinsic 调用

Buffer
    ├── name: string
    ├── dtype: DataType
    ├── shape: vector<PrimExpr>
    ├── strides: vector<PrimExpr>    ← 可选；nullptr 表示 row-major
    ├── data_alignment: int
    └── scope: string               ← "global" / "local"
```

---

## 4. DataType 与 Var

### 4.1 DataType（复用 HLIR 中的定义）

LLIR 与 HLIR 共享同一套 `DataType` 枚举，不重复定义。

### 4.2 Var —— 标量变量

LLIR 中的 `Var` 与 HLIR 中的 `Var` **不同**：LLIR `Var` 专指标量整数/浮点变量（循环变量、临时累加器等），持有 `DataType`。

```cpp
// include/ir/prim_expr.h

class PrimExpr : public IRNode {
public:
    // 表达式的数据类型（标量或向量）
    DataType dtype;
};

class Var : public PrimExpr {
public:
    std::string name;

    IRNodeType node_type() const override { return IRNodeType::kLLVar; }
    static Ref<Var> make(std::string name, DataType dtype = DataType::kInt64);
};
```

---

## 5. Buffer —— 多维内存描述符

`Buffer` 描述一块内存区域的访问方式，不持有实际数据指针（运行时分配）。

```cpp
// include/ir/buffer.h

enum class BufferScope {
    kGlobal,   // 模型输入/输出/权重（外部传入）
    kLocal,    // 函数内局部临时 Buffer（由 Allocate 分配）
};

class Buffer : public IRNode {
public:
    std::string              name;
    DataType                 dtype;
    std::vector<Ref<PrimExpr>> shape;     // 各维度大小（PrimExpr，支持符号维）
    std::vector<Ref<PrimExpr>> strides;   // 为空则表示 row-major 连续存储
    int                      data_alignment;  // 字节对齐（默认 16，NEON 要求）
    BufferScope              scope;

    IRNodeType node_type() const override { return IRNodeType::kBuffer; }

    static Ref<Buffer> make(
        std::string name,
        DataType dtype,
        std::vector<Ref<PrimExpr>> shape,
        std::vector<Ref<PrimExpr>> strides = {},
        int data_alignment = 16,
        BufferScope scope = BufferScope::kGlobal
    );

    // 辅助：计算线性偏移（row-major）
    // offset = indices[0]*strides[0] + ... + indices[N-1]*strides[N-1]
    Ref<PrimExpr> linearize(const std::vector<Ref<PrimExpr>>& indices) const;
};
```

**内存布局说明**：

- `strides` 为空时，视为标准行主序（C-order）连续存储。
- Layout Transformation Pass 将 NCHW layout 的 Buffer 改写为 NCHWc，并相应更新 strides。
- NEON 要求 16 字节对齐（128-bit 向量）；`data_alignment = 16`。

---

## 6. PrimExpr —— 基本表达式

### 6.1 常量

```cpp
class IntImm : public PrimExpr {
public:
    int64_t value;
    static Ref<IntImm> make(int64_t v, DataType dt = DataType::kInt64);
};

class FloatImm : public PrimExpr {
public:
    double value;
    static Ref<FloatImm> make(double v, DataType dt = DataType::kFloat32);
};
```

### 6.2 BufferLoad —— Buffer 读取

```cpp
class BufferLoad : public PrimExpr {
public:
    Ref<Buffer>                  buffer;
    std::vector<Ref<PrimExpr>>   indices;  // 多维索引，与 buffer.shape 维度一致

    IRNodeType node_type() const override { return IRNodeType::kBufferLoad; }
    static Ref<BufferLoad> make(Ref<Buffer> buf, std::vector<Ref<PrimExpr>> indices);
};
```

### 6.3 算术二元运算

```cpp
// 二元运算节点统一结构（以 Add 为例）
class BinaryOp : public PrimExpr {
public:
    Ref<PrimExpr> a;
    Ref<PrimExpr> b;
};

class Add  : public BinaryOp { ... };
class Sub  : public BinaryOp { ... };
class Mul  : public BinaryOp { ... };
class Div  : public BinaryOp { ... };    // 截断除法
class Mod  : public BinaryOp { ... };
class FloorDiv : public BinaryOp { ... };
class FloorMod : public BinaryOp { ... };
class Min  : public BinaryOp { ... };
class Max  : public BinaryOp { ... };
```

### 6.4 Select —— 三目运算

```cpp
class Select : public PrimExpr {
public:
    Ref<PrimExpr> condition;  // bool 型
    Ref<PrimExpr> true_value;
    Ref<PrimExpr> false_value;

    static Ref<Select> make(Ref<PrimExpr> cond,
                             Ref<PrimExpr> true_val,
                             Ref<PrimExpr> false_val);
};
```

### 6.5 Cast —— 类型转换

```cpp
class Cast : public PrimExpr {
public:
    Ref<PrimExpr> value;
    // target dtype 存储在基类 PrimExpr::dtype

    static Ref<Cast> make(DataType target_dtype, Ref<PrimExpr> value);
};
```

### 6.6 Ramp —— 向量化索引

用于生成连续的向量化索引，如 `[base, base+stride, base+2*stride, ..., base+(lanes-1)*stride]`：

```cpp
class Ramp : public PrimExpr {
public:
    Ref<PrimExpr> base;    // 起始索引（标量）
    Ref<PrimExpr> stride;  // 步长（标量，通常为 1）
    int           lanes;   // 向量宽度（FP32 NEON = 4）

    // dtype 为向量类型，如 float32x4
    static Ref<Ramp> make(Ref<PrimExpr> base, Ref<PrimExpr> stride, int lanes);
};
```

### 6.7 Broadcast —— 标量广播

将标量广播为 lanes 宽度的向量，用于向量乘法中的权重加载：

```cpp
class Broadcast : public PrimExpr {
public:
    Ref<PrimExpr> value;   // 标量
    int           lanes;   // 向量宽度

    static Ref<Broadcast> make(Ref<PrimExpr> value, int lanes);
};
```

### 6.8 Call —— 内置函数调用

```cpp
enum class CallType {
    kIntrinsic,   // NEON intrinsic（如 vfmaq_f32）
    kExtern,      // 外部 C 函数
    kBuiltin,     // 内置运算（如 exp、log）
};

class Call : public PrimExpr {
public:
    std::string              func_name;
    CallType                 call_type;
    std::vector<Ref<PrimExpr>> args;

    static Ref<Call> make(DataType dtype,
                           std::string func_name,
                           CallType call_type,
                           std::vector<Ref<PrimExpr>> args);
};
```

---

## 7. Stmt —— 语句节点

### 7.1 SeqStmt —— 顺序语句

```cpp
class SeqStmt : public Stmt {
public:
    std::vector<Ref<Stmt>> seq;

    IRNodeType node_type() const override { return IRNodeType::kSeqStmt; }
    static Ref<SeqStmt> make(std::vector<Ref<Stmt>> seq);

    // 工厂：自动压平嵌套 SeqStmt
    static Ref<Stmt> flatten(std::vector<Ref<Stmt>> stmts);
};
```

### 7.2 For —— 循环

```cpp
enum class ForKind {
    kSerial,      // 普通顺序循环：for (i = min; i < min+extent; i++)
    kVectorized,  // 向量化：编译为 NEON 向量指令
    kUnrolled,    // 展开：codegen 时完全展开循环体
};

class For : public Stmt {
public:
    Ref<Var>      loop_var;   // 循环变量（整数）
    Ref<PrimExpr> min;        // 循环起始值（通常为 IntImm(0)）
    Ref<PrimExpr> extent;     // 循环次数（即循环上界 = min + extent）
    ForKind       kind;
    Ref<Stmt>     body;

    IRNodeType node_type() const override { return IRNodeType::kFor; }
    static Ref<For> make(Ref<Var> loop_var,
                          Ref<PrimExpr> min,
                          Ref<PrimExpr> extent,
                          ForKind kind,
                          Ref<Stmt> body);
};
```

**示例**：7 重循环的 Conv2D（展开 OC 内层为向量化循环）：

```
For(n, 0, N, Serial,
  For(oc_outer, 0, OC/4, Serial,
    For(oh, 0, OH, Serial,
      For(ow, 0, OW, Serial,
        For(ic, 0, IC, Serial,
          For(kh, 0, KH, Serial,
            For(kw, 0, KW, Serial,
              For(oc_inner, 0, 4, Vectorized,
                BufferStore(output, [n, oc_outer*4+oc_inner, oh, ow],
                  Add(BufferLoad(output, [...]),
                      Mul(BufferLoad(input, [...]),
                          BufferLoad(weight, [...])))))))))))))
```

### 7.3 IfThenElse —— 条件分支

```cpp
class IfThenElse : public Stmt {
public:
    Ref<PrimExpr> condition;
    Ref<Stmt>     then_case;
    Ref<Stmt>     else_case;  // 可为 nullptr

    IRNodeType node_type() const override { return IRNodeType::kIfThenElse; }
    static Ref<IfThenElse> make(Ref<PrimExpr> condition,
                                 Ref<Stmt> then_case,
                                 Ref<Stmt> else_case = nullptr);
};
```

### 7.4 Allocate —— 局部内存分配

用于在 PrimFunc 内部声明临时中间 Buffer（如 im2col 临时矩阵、局部累加器）：

```cpp
class Allocate : public Stmt {
public:
    Ref<Var>                   buffer_var;  // 对应 Buffer 的数据指针变量
    DataType                   dtype;
    std::vector<Ref<PrimExpr>> extents;     // 各维度大小
    Ref<Stmt>                  body;        // 分配的作用域

    IRNodeType node_type() const override { return IRNodeType::kAllocate; }
    static Ref<Allocate> make(Ref<Var> buffer_var,
                               DataType dtype,
                               std::vector<Ref<PrimExpr>> extents,
                               Ref<Stmt> body);
};
```

### 7.5 BufferStore —— Buffer 写入

```cpp
class BufferStore : public Stmt {
public:
    Ref<Buffer>                  buffer;
    std::vector<Ref<PrimExpr>>   indices;
    Ref<PrimExpr>                value;   // 写入的值（标量或向量）

    IRNodeType node_type() const override { return IRNodeType::kBufferStore; }
    static Ref<BufferStore> make(Ref<Buffer> buffer,
                                  std::vector<Ref<PrimExpr>> indices,
                                  Ref<PrimExpr> value);
};
```

### 7.6 Evaluate —— 表达式语句

将一个表达式作为语句执行（通常用于有副作用的 intrinsic 调用，或 void 函数调用）：

```cpp
class Evaluate : public Stmt {
public:
    Ref<PrimExpr> value;

    IRNodeType node_type() const override { return IRNodeType::kEvaluate; }
    static Ref<Evaluate> make(Ref<PrimExpr> value);
};
```

---

## 8. PrimFunc —— 低层函数

`PrimFunc` 是 LLIR 的函数单元，对应 HLIR 中一个算子（或融合算子）的具体执行实现：

```cpp
// include/ir/prim_func.h

class PrimFunc : public IRNode {
public:
    std::string               name;
    std::vector<Ref<Buffer>>  params;    // 函数参数（输入/输出 Buffer）
    Ref<Stmt>                 body;      // 函数体（Stmt 树）

    // 元信息（供 Codegen 使用）
    std::unordered_map<std::string, std::string> attrs;
    // 示例：attrs["target"] = "arm_neon"

    IRNodeType node_type() const override { return IRNodeType::kPrimFunc; }

    static Ref<PrimFunc> make(std::string name,
                               std::vector<Ref<Buffer>> params,
                               Ref<Stmt> body,
                               std::unordered_map<std::string, std::string> attrs = {});
};
```

**PrimFunc 与 HLIR Function 的对应关系**：

```
HLIR: Call(nn.conv2d, [data, weight, bias], attrs)
          │
          │ Lowering
          ▼
LLIR: PrimFunc "conv2d_0"
      params: [Buffer data, Buffer weight, Buffer bias, Buffer output]
      body:   SeqStmt [
                Allocate(...),   // 可选临时 Buffer
                For(...),        // 7 重嵌套循环
              ]
```

---

## 9. LLIRModule

```cpp
// include/ir/ir_module.h（扩展）

class LLIRModule : public IRNode {
public:
    // 函数名 → PrimFunc
    std::unordered_map<std::string, Ref<PrimFunc>> functions;
    // 执行顺序（拓扑序）
    std::vector<std::string>  exec_order;

    IRNodeType node_type() const override { return IRNodeType::kLLIRModule; }

    static Ref<LLIRModule> make();
    void add_func(const std::string& name, Ref<PrimFunc> func);
    Ref<PrimFunc> get_func(const std::string& name) const;
};
```

---

## 10. C++ 类继承关系图

```
IRNode
├── PrimExpr
│   ├── Var (LLIR)
│   ├── IntImm
│   ├── FloatImm
│   ├── BufferLoad
│   ├── BinaryOp
│   │   ├── Add / Sub / Mul / Div / Mod
│   │   ├── FloorDiv / FloorMod
│   │   └── Min / Max
│   ├── Select
│   ├── Cast
│   ├── Ramp
│   ├── Broadcast
│   └── Call (PrimExpr)
├── Stmt
│   ├── SeqStmt
│   ├── For
│   ├── IfThenElse
│   ├── Allocate
│   ├── BufferStore
│   └── Evaluate
├── Buffer
├── PrimFunc
└── LLIRModule
```

---

## 11. 访问者模式

LLIR 同样使用 Visitor/Mutator 模式进行遍历与变换：

```cpp
// StmtVisitor：只读遍历
class StmtVisitor {
public:
    virtual void visit_stmt(Ref<Stmt> stmt);
    virtual void visit_for(Ref<For> node);
    virtual void visit_seq_stmt(Ref<SeqStmt> node);
    virtual void visit_buffer_store(Ref<BufferStore> node);
    virtual void visit_allocate(Ref<Allocate> node);
    virtual void visit_if_then_else(Ref<IfThenElse> node);
    virtual void visit_evaluate(Ref<Evaluate> node);
};

// StmtMutator：带变换的遍历（Loop Tiling Pass 使用此类）
class StmtMutator {
public:
    virtual Ref<Stmt> mutate_stmt(Ref<Stmt> stmt);
    virtual Ref<Stmt> mutate_for(Ref<For> node);
    virtual Ref<Stmt> mutate_seq_stmt(Ref<SeqStmt> node);
    virtual Ref<Stmt> mutate_buffer_store(Ref<BufferStore> node);
    // ...
};
```

---

## 12. 设计示例：Conv2D PrimFunc 结构

以 Conv2D（N=1, IC=3, OC=64, IH=IW=224, KH=KW=3, Stride=1, Pad=1）为例，展示 LLIR 结构（伪代码形式）：

```
PrimFunc "conv2d_0" (
    input:  Buffer[float32, shape=[1,3,224,224]],
    weight: Buffer[float32, shape=[64,3,3,3]],
    bias:   Buffer[float32, shape=[64]],
    output: Buffer[float32, shape=[1,64,224,224]]
) {
  SeqStmt [
    // 初始化：output = bias（广播）
    For n=0..1 [Serial]
      For oc=0..64 [Serial]
        For oh=0..224 [Serial]
          For ow=0..224 [Serial]
            BufferStore(output, [n,oc,oh,ow], BufferLoad(bias, [oc]))

    // 主计算：累加卷积
    For n=0..1 [Serial]
      For oc_o=0..16 [Serial]          // OC/4 = 16（Tiling 后）
        For oh=0..224 [Serial]
          For ow=0..224 [Serial]
            For ic=0..3 [Serial]
              For kh=0..3 [Serial]
                For kw=0..3 [Serial]
                  For oc_i=0..4 [Vectorized]   // NEON 4×float32
                    BufferStore(
                      output, [n, oc_o*4+oc_i, oh, ow],
                      Add(
                        BufferLoad(output, [n, oc_o*4+oc_i, oh, ow]),
                        Mul(
                          BufferLoad(input,  [n, ic, oh+kh-1, ow+kw-1]),
                          BufferLoad(weight, [oc_o*4+oc_i, ic, kh, kw])
                        )
                      )
                    )
  ]
}
```

此结构清晰映射到最终生成的 C++ 嵌套循环，向量化 For 将被 Codegen 翻译为 NEON Intrinsics。
