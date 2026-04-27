# 07 Lowering 模块

## 目录

- [1. 设计目标与职责](#1-设计目标与职责)
- [2. Lowering 总体流程](#2-lowering-总体流程)
- [3. 接口设计](#3-接口设计)
- [4. 算子 Lowering 规则框架](#4-算子-lowering-规则框架)
- [5. Buffer 分配策略](#5-buffer-分配策略)
- [6. 各算子 Lowering 规则详解](#6-各算子-lowering-规则详解)
  - [6.1 Conv2D](#61-conv2d)
  - [6.2 BatchNorm（推理期）](#62-batchnorm推理期)
  - [6.3 ReLU / Clip](#63-relu--clip)
  - [6.4 MaxPool2D / AvgPool2D](#64-maxpool2d--avgpool2d)
  - [6.5 Dense（全连接）](#65-dense全连接)
  - [6.6 Add（逐元素加）](#66-add逐元素加)
  - [6.7 Concat](#67-concat)
  - [6.8 融合算子（CompositeFunction）](#68-融合算子compositefunction)
- [7. Conv2D Lowering 完整示例](#7-conv2d-lowering-完整示例)
- [8. 拓扑排序与执行序列](#8-拓扑排序与执行序列)

---

## 1. 设计目标与职责

Lowering 模块是**连接 High-level IR 与 Low-level IR 的唯一通道**，承担以下职责：

1. 将 HLIR 中的每个 `Call` 节点（算子调用）展开为对应的 `PrimFunc`（多重循环结构）
2. 为每个 `PrimFunc` 的输入/输出分配 `Buffer` 描述符
3. 确定 `PrimFunc` 的执行顺序（拓扑序）
4. 生成 `LLIRModule`，供后续 Loop Tiling Pass 和 Codegen 消费

Lowering 模块**不做优化**，只做语义等价的结构转换。优化由 Pass 负责。

---

## 2. Lowering 总体流程

```
输入：优化后的 HLIRModule（main Function）
       │
       ▼
Step 1：图拓扑排序
  - 从 Function.body 出发，DFS 收集所有 Call 节点
  - 按计算依赖关系排成 topological order
       │
       ▼
Step 2：Buffer 分配
  - 为图的输入 Var → 分配 global Buffer（输入 Buffer）
  - 为每个 Call 的输出 → 分配 local Buffer（中间结果 Buffer）
  - 为 Constant → 分配 global Buffer（权重 Buffer，只读）
  - 为图的最终输出 → 分配 global Buffer（输出 Buffer）
       │
       ▼
Step 3：逐算子 Lowering
  - 对拓扑序中的每个 Call 节点，查找对应的 Lowering 规则（OpLoweringRule）
  - 根据规则 + 输入/输出 Buffer + 算子属性，生成 PrimFunc
       │
       ▼
Step 4：构建 LLIRModule
  - 将所有 PrimFunc 按拓扑序放入 LLIRModule
  - 记录执行顺序列表（exec_order）
       │
       ▼
输出：LLIRModule
```

---

## 3. 接口设计

### 3.1 Lowering 主接口

```cpp
// include/lowering/lowering.h

class Lowering {
public:
    // 将 HLIRModule 转换为 LLIRModule
    static Ref<LLIRModule> lower(Ref<IRModule> hlir_mod);

private:
    Lowering() = default;

    // 内部状态：Buffer 映射表（Expr指针 → Buffer）
    std::unordered_map<IRNode*, Ref<Buffer>> buffer_map_;

    // 已生成的 PrimFunc 列表
    std::vector<std::pair<std::string, Ref<PrimFunc>>> prim_funcs_;

    // 函数计数器（用于生成唯一函数名）
    int func_counter_ = 0;

    // 执行主流程
    Ref<LLIRModule> lower_function(Ref<Function> func);

    // 拓扑排序
    std::vector<Ref<Call>> topological_sort(Ref<Expr> body);

    // Buffer 分配
    Ref<Buffer> alloc_buffer(const std::string& name,
                              Ref<TensorType> ttype,
                              BufferScope scope);

    // 单算子 Lowering
    Ref<PrimFunc> lower_call(Ref<Call> call,
                              const std::vector<Ref<Buffer>>& in_bufs,
                              Ref<Buffer> out_buf);

    // 生成唯一函数名
    std::string gen_func_name(const std::string& op_name);
};
```

### 3.2 OpLoweringRule 抽象接口

每个算子对应一个 `OpLoweringRule`，封装其 Lowering 逻辑：

```cpp
// include/lowering/op_lowering_rules.h

class OpLoweringRule {
public:
    virtual ~OpLoweringRule() = default;

    // 该 Rule 处理哪个 Op 名称
    virtual std::string op_name() const = 0;

    // 生成 PrimFunc body（Stmt 树）
    // 参数：
    //   in_bufs  - 输入 Buffer 列表（与 Call.args 一一对应）
    //   out_buf  - 输出 Buffer
    //   attrs    - 算子属性
    virtual Ref<Stmt> lower(
        const std::vector<Ref<Buffer>>& in_bufs,
        Ref<Buffer> out_buf,
        const Attrs& attrs
    ) = 0;
};

// 全局注册表
class OpLoweringRegistry {
public:
    static void register_rule(std::shared_ptr<OpLoweringRule> rule);
    static OpLoweringRule* get(const std::string& op_name);
    static bool has(const std::string& op_name);
};

#define REGISTER_LOWERING_RULE(cls) \
    static bool _##cls##_rule_registered = []() { \
        OpLoweringRegistry::register_rule(std::make_shared<cls>()); \
        return true; \
    }()
```

---

## 4. 算子 Lowering 规则框架

### 4.1 辅助函数

Lowering 规则实现时频繁使用以下辅助函数，统一封装：

```cpp
namespace lower_utils {

// 创建常量整数表达式
Ref<IntImm> ci(int64_t v);

// 创建常量浮点表达式
Ref<FloatImm> cf(double v);

// 创建 For 循环（Serial）
Ref<For> serial_for(const std::string& var_name,
                     int64_t extent,
                     Ref<Stmt> body);

// 创建 For 循环（Vectorized）
Ref<For> vec_for(const std::string& var_name,
                  int64_t extent,
                  Ref<Stmt> body);

// 多维 BufferLoad
Ref<BufferLoad> load(Ref<Buffer> buf,
                      std::initializer_list<Ref<PrimExpr>> indices);

// 多维 BufferStore
Ref<BufferStore> store(Ref<Buffer> buf,
                        std::initializer_list<Ref<PrimExpr>> indices,
                        Ref<PrimExpr> value);

// 构建 SeqStmt（自动压平）
Ref<Stmt> seq(std::vector<Ref<Stmt>> stmts);

// padding 越界检查条件
Ref<PrimExpr> in_bounds(Ref<PrimExpr> idx, int64_t size);

} // namespace lower_utils
```

---

## 5. Buffer 分配策略

### 5.1 Buffer 命名规则

| 场景 | Buffer 名称 | Scope |
|------|------------|-------|
| 图输入 Var | `"input_<var_name>"` | Global |
| 常量权重 | `"const_<op_name>_<idx>"` | Global |
| 中间结果（Call 输出） | `"buf_<op_name>_<counter>"` | Local（后续可提升为 Global 参数） |
| 最终输出 | `"output"` | Global |

### 5.2 shape 与 dtype 来源

从 HLIR 中每个节点的 `checked_type`（`TensorType`）获取 shape 和 dtype，确保与前端解析结果一致。因此 Lowering 前**必须**完成 HLIR 的类型推断（Type Inference Pass）。

### 5.3 内存对齐

所有 Buffer 统一设置 `data_alignment = 16`（128-bit，适配 NEON `vld1q_f32`）。

---

## 6. 各算子 Lowering 规则详解

### 6.1 Conv2D

**输入 Buffer**：`[data(NCHW/NCHWc), weight(OIHW), bias(O)]`  
**输出 Buffer**：`output(NCHW/NCHWc)`

**循环结构**（NCHW 格式，7 重）：

```
for n in [0, N):
  for oc in [0, OC):
    for oh in [0, OH):
      for ow in [0, OW):
        // 初始化：output[n,oc,oh,ow] = bias[oc]
        output[n,oc,oh,ow] = bias[oc]
        for ic in [0, IC):
          for kh in [0, KH):
            for kw in [0, KW):
              ih = oh * stride_h + kh - pad_h
              iw = ow * stride_w + kw - pad_w
              if 0 <= ih < IH and 0 <= iw < IW:
                output[n,oc,oh,ow] += input[n,ic,ih,iw] * weight[oc,ic,kh,kw]
```

**LLIR 对应结构**（简化）：

```cpp
// 生成 7 重嵌套 For Stmt
auto body = lower_utils::seq({
    // 偏置初始化
    init_loop,
    // 卷积主循环
    conv_loop
});
return PrimFunc::make(name, {data_buf, weight_buf, bias_buf, out_buf}, body);
```

**Padding 处理**：通过 `IfThenElse` 判断越界，越界位置用 0 填充（等价于 zero-padding）。

### 6.2 BatchNorm（推理期）

推理期 BN 参数（gamma, beta, mean, var）均为 `Constant`，可融合为一个仿射变换：

```
y = gamma / sqrt(var + eps) * x + (beta - gamma * mean / sqrt(var + eps))
  = scale * x + bias_bn
```

Lowering 时**在编译期预计算** `scale` 和 `bias_bn`，生成简单的逐元素仿射变换循环：

```
for n, c, h, w:
  output[n,c,h,w] = scale[c] * input[n,c,h,w] + bias_bn[c]
```

### 6.3 ReLU / Clip

```
// ReLU
for n, c, h, w:
  output[n,c,h,w] = max(0.0, input[n,c,h,w])

// Clip(min=0, max=6) → ReLU6
for n, c, h, w:
  output[n,c,h,w] = max(min_val, min(max_val, input[n,c,h,w]))
```

LLIR 中使用 `Max(FloatImm(0.0), BufferLoad(input, ...))` 节点，Codegen 映射为 `vmaxq_f32`。

### 6.4 MaxPool2D / AvgPool2D

**MaxPool 循环结构**：

```
for n in [0, N):
  for c in [0, C):
    for oh in [0, OH):
      for ow in [0, OW):
        output[n,c,oh,ow] = -INF
        for kh in [0, KH):
          for kw in [0, KW):
            ih = oh * stride_h + kh
            iw = ow * stride_w + kw
            if in_bounds(ih, IH) and in_bounds(iw, IW):
              output[n,c,oh,ow] = max(output[n,c,oh,ow], input[n,c,ih,iw])
```

**AvgPool 循环结构**（额外计算计数）：

```
for ...:
  sum = 0.0; count = 0
  for kh, kw:
    if in_bounds:
      sum += input[...]; count++
  output[...] = sum / count
```

### 6.5 Dense（全连接）

**输入 Buffer**：`[input(N×IC), weight(OC×IC), bias(OC)]`  
**输出 Buffer**：`output(N×OC)`

```
for n in [0, N):
  for oc in [0, OC):
    output[n,oc] = bias[oc]
    for ic in [0, IC):
      output[n,oc] += input[n,ic] * weight[oc,ic]
```

### 6.6 Add（逐元素加）

```
for n, c, h, w:
  output[n,c,h,w] = lhs[n,c,h,w] + rhs[n,c,h,w]
```

广播加（如 bias add）通过自动扩展维度索引实现：rhs 在需要广播的维度使用固定索引 0。

### 6.7 Concat

沿 `axis` 维度拼接，通过分段 `BufferStore` 实现：

```
// Concat 沿 axis=1（通道维度），输入 A(N,C1,H,W) 和 B(N,C2,H,W)
// 输出 output(N, C1+C2, H, W)

// 第一段：复制 A
for n, c, h, w (c in [0, C1)):
  output[n, c, h, w] = A[n, c, h, w]

// 第二段：复制 B，偏移 C1
for n, c, h, w (c in [0, C2)):
  output[n, C1+c, h, w] = B[n, c, h, w]
```

### 6.8 融合算子（CompositeFunction）

对 OperatorFusion Pass 产生的 `Call(CompositeFunction, ...)` 节点，Lowering 策略为：

1. 识别 `CompositeFunction` 的模式（通过 `attrs["Composite"]`）
2. 查找对应的**融合 Lowering 规则**（`FusedOpLoweringRule`）
3. 生成融合后的 PrimFunc，将多个算子的循环体合并

**Conv+BN+ReLU 融合 Lowering**：

```
// 在一次 7 重循环内完成：卷积 + BN 仿射 + ReLU
for n, oc, oh, ow, ic, kh, kw:
  val  = conv2d_accumulate(...)
  val  = bn_scale[oc] * val + bn_bias[oc]   // BN
  val  = max(0.0, val)                        // ReLU
  output[n,oc,oh,ow] = val
```

消除了 BN 和 ReLU 的独立 Buffer 读写。

---

## 7. Conv2D Lowering 完整示例

以下展示 Conv2D（N=1, IC=64, OC=128, IH=IW=56, KH=KW=3, stride=1, pad=1）的完整 LLIR 生成代码（C++ 伪代码）：

```cpp
Ref<PrimFunc> Conv2DLoweringRule::lower(
    const std::vector<Ref<Buffer>>& in_bufs,
    Ref<Buffer> out_buf,
    const Attrs& attrs
) {
    using namespace lower_utils;

    auto data   = in_bufs[0];  // [1, 64, 56, 56]
    auto weight = in_bufs[1];  // [128, 64, 3, 3]
    auto bias   = in_bufs[2];  // [128]

    // 从 Buffer shape 和 attrs 提取维度信息
    int64_t N  = 1, IC = 64, OC = 128;
    int64_t IH = 56, IW = 56, OH = 56, OW = 56;
    int64_t KH = 3, KW = 3;
    int64_t SH = attrs.get_or("strides_h", 1LL);
    int64_t SW = attrs.get_or("strides_w", 1LL);
    int64_t PH = attrs.get_or("pad_h", 1LL);
    int64_t PW = attrs.get_or("pad_w", 1LL);

    // 循环变量
    auto vn  = Var::make("n",  kInt64);
    auto voc = Var::make("oc", kInt64);
    auto voh = Var::make("oh", kInt64);
    auto vow = Var::make("ow", kInt64);
    auto vic = Var::make("ic", kInt64);
    auto vkh = Var::make("kh", kInt64);
    auto vkw = Var::make("kw", kInt64);

    // 计算输入坐标
    auto ih = Add::make(Mul::make(voh, ci(SH)), Sub::make(vkh, ci(PH)));
    auto iw = Add::make(Mul::make(vow, ci(SW)), Sub::make(vkw, ci(PW)));

    // 越界条件
    auto ih_valid = And::make(GE::make(ih, ci(0)), LT::make(ih, ci(IH)));
    auto iw_valid = And::make(GE::make(iw, ci(0)), LT::make(iw, ci(IW)));
    auto in_valid = And::make(ih_valid, iw_valid);

    // 卷积计算（带 padding 条件）
    auto acc_val = IfThenElse::make(
        in_valid,
        Add::make(
            load(out_buf, {vn, voc, voh, vow}),
            Mul::make(
                load(data, {vn, vic, ih, iw}),
                load(weight, {voc, vic, vkh, vkw})
            )
        ),
        load(out_buf, {vn, voc, voh, vow})  // padding 区域不更新
    );
    auto inner_store = store(out_buf, {vn, voc, voh, vow}, acc_val);

    // 初始化循环（output = bias）
    auto init_store = store(out_buf, {vn, voc, voh, vow},
                             load(bias, {voc}));
    auto init_loop = serial_for("n",  N,
                     serial_for("oc", OC,
                     serial_for("oh", OH,
                     serial_for("ow", OW, init_store))));

    // 主计算循环（7 重嵌套）
    auto main_loop = serial_for("n",  N,
                     serial_for("oc", OC,
                     serial_for("oh", OH,
                     serial_for("ow", OW,
                     serial_for("ic", IC,
                     serial_for("kh", KH,
                     serial_for("kw", KW, inner_store)))))));

    auto body = seq({init_loop, main_loop});

    return PrimFunc::make("conv2d_0",
                           {data, weight, bias, out_buf},
                           body,
                           {{"op", "nn.conv2d"}});
}
```

---

## 8. 拓扑排序与执行序列

HLIR Function.body 是一棵 DAG（有向无环图），Lowering 需确定各 PrimFunc 的执行顺序。

**算法**：后序 DFS（Post-order DFS）

```
topological_sort(expr):
  visited = set()
  result  = []

  dfs(expr):
    if expr in visited: return
    visited.add(expr)
    if isinstance(expr, Call):
      for arg in expr.args:
        dfs(arg)           // 先访问依赖
      result.append(expr)  // 后序：依赖先于当前
    elif isinstance(expr, Let):
      dfs(expr.value)
      dfs(expr.body)
    elif isinstance(expr, Tuple):
      for f in expr.fields: dfs(f)

  dfs(expr)
  return result  // 即为拓扑序
```

`LLIRModule::exec_order` 存储按此顺序生成的 PrimFunc 名称列表，Codegen 按此顺序生成函数调用链。
