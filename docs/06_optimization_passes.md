# 06 优化 Pass 设计

## 目录

- [1. 概述与 Pass 选择依据](#1-概述与-pass-选择依据)
- [2. Constant Folding（常量折叠）](#2-constant-folding常量折叠)
- [3. Operator Fusion（算子融合）](#3-operator-fusion算子融合)
- [4. Dead Code Elimination（死代码消除）](#4-dead-code-elimination死代码消除)
- [5. Layout Transformation（布局变换）](#5-layout-transformation布局变换)
- [6. Loop Tiling（循环分块）](#6-loop-tiling循环分块)

---

## 1. 概述与 Pass 选择依据

| Pass | 作用层 | 优化目标 | 预期收益 |
|------|--------|---------|---------|
| Constant Folding | HLIR（FunctionPass） | 编译期计算常量子图，减少运行时计算 | 减少算子数量，消除冗余计算 |
| Operator Fusion | HLIR（FunctionPass） | 合并 Conv+BN+ReLU 等链，减少内存往返 | 显著降低访存带宽，减少临时张量分配 |
| Dead Code Elimination | HLIR（ModulePass） | 删除输出不可达节点 | 减少编译产物大小，节省运行时内存 |
| Layout Transformation | HLIR（FunctionPass） | NCHW → NCHWc，改善 NEON 向量访存模式 | 提升 NEON 利用率，减少 gather 访存 |
| Loop Tiling | LLIR（ModulePass） | 分块循环以适配 Cache，改善数据局部性 | 提升 L1/L2 Cache 命中率，降低内存延迟 |

---

## 2. Constant Folding（常量折叠）

### 2.1 设计目标

在编译期对**所有输入均为 `Constant` 的 `Call` 节点**进行求值，将其替换为 `Constant` 节点，减少运行时计算量。

**典型场景**：
- `Flatten(Constant)` → 直接折叠
- BN 的 mean/variance/gamma/beta 均为常量 → `BatchNorm(data, Constant, ...)` 中非数据输入已折叠
- `Reshape(Constant, Constant)` → 直接替换为重形状后的 Constant

### 2.2 算法设计

**遍历策略**：后序遍历（Post-order）ExprMutator，确保子节点先处理，处理当前节点时子节点已被折叠。

```
mutate_call(Call node):
  1. 递归调用 ExprMutator::mutate_call(node) → 处理所有 args
  2. 检查处理后的 args 是否全部为 Constant：
       all_const = all(arg.node_type == kConstant for arg in new_args)
  3. if all_const:
       result = eval_op(op_name, new_args, attrs)  // 编译期求值
       return Constant(result)
  4. else:
       return Call(op, new_args, attrs)
```

### 2.3 编译期求值（eval_op）

```cpp
Ref<Constant> eval_op(const std::string& op_name,
                       const std::vector<Ref<Constant>>& args,
                       const Attrs& attrs) {
    // 将 Constant 数据转为 numpy-style ndarray（C++ 实现简单张量运算）
    // 根据 op_name 分发执行
    if (op_name == "nn.relu") {
        return eval_relu(args[0]);
    } else if (op_name == "nn.batch_norm") {
        return eval_batch_norm(args, attrs);
    } else if (op_name == "nn.flatten") {
        return eval_flatten(args[0], attrs);
    } else if (op_name == "nn.reshape") {
        return eval_reshape(args[0], args[1]);
    }
    // 不支持编译期求值的 op（如 Conv2D，输入太大）
    // 返回 nullptr，交由调用方保留原始 Call 节点
    return nullptr;
}
```

**注意**：对于计算量较大的算子（如 Conv2D），即使输入全是常量也**不应**在编译期展开（权重是常量，但数据不是）。实际上，Conv2D 的数据输入（graph input）永远不会是 `Constant`，因此自然不满足全常量条件。

### 2.4 核心类设计

```cpp
class ConstantFolding : public FunctionPass {
public:
    std::string name() const override { return "ConstantFolding"; }
    int min_opt_level() const override { return 1; }

    Ref<Function> transform_function(
        Ref<Function> func,
        Ref<IRModule> mod,
        const PassContext& ctx
    ) override;

private:
    class Folder : public ExprMutator {
        Ref<Expr> mutate_call(Ref<Call> node) override;
    };
};
```

---

## 3. Operator Fusion（算子融合）

### 3.1 设计目标

将满足特定**融合模式**的算子序列合并为一个 **Composite Function**（`Function` with `attrs["Composite"]`），在 Codegen 阶段作为一个整体生成优化后的代码，消除中间结果的内存读写。

**融合收益**：以 Conv+BN+ReLU 为例，融合前需 3 次内存读写（3 个中间张量），融合后只需 1 次写出最终结果，节省约 60% 的访存带宽。

### 3.2 支持的融合模式

| 模式名称 | 算子链 | 说明 |
|---------|--------|------|
| `conv2d_bias_relu` | Conv2D → BiasAdd → ReLU | 最常见 CNN 基础块 |
| `conv2d_bias_bn_relu` | Conv2D → BatchNorm → ReLU | 含 BN（推理期 BN 参数为常量） |
| `conv2d_bias` | Conv2D → BiasAdd | 无激活的卷积 |
| `dense_bias_relu` | Dense → BiasAdd → ReLU | 全连接块 |
| `add_relu` | Add → ReLU | 残差加激活 |

> 后续可扩展：通过注册新模式不需修改核心逻辑。

### 3.3 算法设计

融合分为**两个阶段**：

#### 阶段一：模式标注（Annotate）

对计算图进行数据流分析，对每个节点标记"可参与哪个融合模式"：

```
标注规则（以 conv2d_bias_relu 为例）：
  - 若 Call(relu, [Call(bias_add, [Call(conv2d, ...), bias])]) 则：
      conv2d 节点 → group_id = X, role = "head"
      bias_add 节点 → group_id = X, role = "inner"
      relu 节点 → group_id = X, role = "tail"
```

实现方式：**自底向上模式匹配**，从叶节点出发，用图遍历识别满足条件的算子链。

#### 阶段二：融合重写（Rewrite）

将标注同一 group_id 的节点替换为一个 `Call(CompositeFunction, ...)` 节点：

```cpp
// 融合后的 IR 结构
// 原来：relu(bias_add(conv2d(data, w, b), b))
// 变为：
//   CompositeFunc = Function([x, w, b], relu(bias_add(conv2d(x, w, b), b)))
//     attrs["Composite"] = "conv2d_bias_relu"
//   Call(CompositeFunc, [data, w, b])

auto fused_func = Function::make(
    {data_param, weight_param, bias_param},
    relu_expr,   // 函数体是融合后的子图
    nullptr,
    {{"Composite", "conv2d_bias_relu"}}
);
auto fused_call = Call::make(fused_func, {data, weight, bias});
```

### 3.4 融合合法性检查

融合需满足以下条件：

1. **单消费者**：中间节点只被一个算子消费（否则融合会重复计算）
2. **线性链**：当前只融合线性链（无 fork/join）
3. **算子支持**：参与融合的算子均在支持列表中

```cpp
bool is_fusable_chain(
    Ref<Call> tail,
    const FusionPattern& pattern,
    const UsageCountMap& usage_count
) {
    // 检查 tail 是否匹配 pattern 的末尾算子
    // 递归向上检查是否匹配完整链
    // 检查中间节点 usage_count == 1
    ...
}
```

### 3.5 核心类设计

```cpp
struct FusionPattern {
    std::string          name;         // 融合模式名
    std::vector<std::string> op_chain; // 从 head 到 tail 的算子名列表
};

class OperatorFusion : public FunctionPass {
public:
    std::string name() const override { return "OperatorFusion"; }
    int min_opt_level() const override { return 2; }
    std::vector<std::string> dependencies() const override {
        return {"ConstantFolding"};
    }

    Ref<Function> transform_function(
        Ref<Function> func,
        Ref<IRModule> mod,
        const PassContext& ctx
    ) override;

    // 注册新融合模式（可在外部扩展）
    static void register_pattern(FusionPattern pattern);
    static const std::vector<FusionPattern>& all_patterns();

private:
    static std::vector<FusionPattern> patterns_;
};
```

---

## 4. Dead Code Elimination（死代码消除）

### 4.1 设计目标

删除**不影响最终输出**的节点（变量、Let 绑定、Constant 等），减少图的复杂度和运行时内存消耗。

**触发场景**：
- OperatorFusion 后，中间节点（如 bias_add 的直接 Call）已被融合进 CompositeFunction，原始 Call 节点成为死代码
- 常量折叠后，原始参数 Constant 可能不再被引用
- 前端导入了一些最终未用的中间变量

### 4.2 算法设计

**引用计数分析（Reference Count Analysis）**：

1. 以 Function 的 `body`（返回值）为根节点
2. **前序 DFS** 遍历，统计每个 `Expr` 节点被引用的次数（`ref_count: map<Expr*, int>`）
3. 将 `ref_count == 0` 的节点视为死代码

**Let 绑定的特殊处理**：

```
let x = dead_expr in
let y = live_expr(x) in
y

→ 若 x 在 live_expr 中无引用，且 x 本身无副作用，则删除
```

```
mutate_let(Let node):
  1. 递归处理 body
  2. 检查 node.var 的 ref_count 是否为 0
  3. 若为 0 且 value 无副作用（非 extern call）：
       return mutate(node.body)   // 直接跳过该 Let 绑定
  4. 否则保留
```

### 4.3 副作用判断

对于 DCE 来说，只要 `Call` 的 `op` 不是 extern（无副作用的纯函数），均可安全删除。在 RASP 中所有 CNN 算子均为纯函数，因此 `Call` 节点均可被 DCE 删除。

### 4.4 核心类设计

```cpp
class DeadCodeElimination : public FunctionPass {
public:
    std::string name() const override { return "DeadCodeElimination"; }
    int min_opt_level() const override { return 1; }

    Ref<Function> transform_function(
        Ref<Function> func,
        Ref<IRModule> mod,
        const PassContext& ctx
    ) override;

private:
    // 第一遍：统计引用计数
    class RefCounter : public ExprVisitor {
    public:
        std::unordered_map<IRNode*, int> ref_count;
        void visit_var(Ref<Var> node) override;
    };

    // 第二遍：删除引用计数为 0 的 Let 绑定
    class DCEMutator : public ExprMutator {
        const std::unordered_map<IRNode*, int>& ref_count_;
    public:
        Ref<Expr> mutate_let(Ref<Let> node) override;
    };
};
```

---

## 5. Layout Transformation（布局变换）

### 5.1 设计目标

将计算图中张量的内存布局从 **NCHW**（PyTorch/ONNX 默认）转换为 **NCHWc**（packed layout），使 NEON 向量化时的数据访问连续，减少 cache miss。

**NCHWc 解释**：  
将 C（通道）维度按块大小 `c`（FP32+NEON → `c=4`）进行分块：  
`shape [N, C, H, W] → shape [N, C/c, H, W, c]`

**访存对比**：

| Layout | Conv2D 输出访问模式 | NEON 效果 |
|--------|-------------------|-----------|
| NCHW | output[n, oc, oh, ow]：oc 相邻元素不连续 | 需 gather 加载，效率低 |
| NCHWc | output[n, oc//4, oh, ow, oc%4]：oc_inner 连续 | vld1q_f32 一次加载 4 元素，高效 |

### 5.2 算法设计

**变换规则**（以 Conv2D 为例）：

```
原始：
  Call(nn.conv2d, [data_NCHW, weight_OIHW, bias_O], attrs)
  data:   shape [N, IC, IH, IW]
  weight: shape [OC, IC, KH, KW]
  output: shape [N, OC, OH, OW]

变换后：
  数据：插入 layout_transform(data_NCHW → data_NCHWc)
       data_packed: shape [N, IC/4, IH, IW, 4]
  权重：weight 为 Constant，直接在编译期重排
       weight_packed: shape [OC/4, IC/4, KH, KW, 4, 4]
  输出：改为 NCHWc layout，在需要 NCHW 输出的位置插入反变换
```

**变换步骤**：

1. 在图的输入端插入 `layout_transform` 节点（NCHW → NCHWc）
2. 对 Constant 权重节点，在编译期直接重排（调用 `eval_layout_transform`）
3. 将 Conv2D、Pooling 等受影响算子的 `attrs["data_layout"]` 更新为 `"NCHWc"`
4. 在图的输出端（或下一个不支持 NCHWc 的算子前）插入反变换节点

**块大小选择**：

```cpp
int get_channel_block_size(DataType dtype) {
    // ARM NEON 128-bit 寄存器：float32 → 4 元素
    if (dtype == DataType::kFloat32) return 4;
    return 4;  // 默认 4
}
```

### 5.3 layout_transform 算子

需在 OpRegistry 中注册专用的布局变换算子：

```cpp
OpRegistry::register_op("nn.layout_transform", {
    {"src_layout", "string", true,  "NCHW"},
    {"dst_layout", "string", true,  "NCHWc"},
    {"block_size",  "int",    false, "4"},
}, "Tensor layout transformation");
```

### 5.4 核心类设计

```cpp
class LayoutTransformation : public FunctionPass {
public:
    std::string name() const override { return "LayoutTransformation"; }
    int min_opt_level() const override { return 2; }
    std::vector<std::string> dependencies() const override {
        return {"OperatorFusion"};
    }

    Ref<Function> transform_function(
        Ref<Function> func,
        Ref<IRModule> mod,
        const PassContext& ctx
    ) override;

private:
    // 对 Constant weight 进行编译期重排
    Ref<Constant> repack_weight(Ref<Constant> weight, const std::string& op_name);

    // 插入 layout_transform 节点
    Ref<Expr> insert_layout_transform(Ref<Expr> expr,
                                       const std::string& src,
                                       const std::string& dst);

    class LayoutMutator : public ExprMutator {
        LayoutTransformation& pass_;
        bool in_nchwc_region_ = false;
    public:
        Ref<Expr> mutate_call(Ref<Call> node) override;
    };
};
```

---

## 6. Loop Tiling（循环分块）

### 6.1 设计目标

对 LLIR 中的 `PrimFunc` 进行循环分块变换，使内层循环的数据访问范围适配 Cortex-A72 的 Cache 大小，减少 Cache miss。

**Cortex-A72 Cache 参数**（树莓派 4B）：

| 级别 | 容量 | Cache Line | 特性 |
|------|------|-----------|------|
| L1 Data | 32 KB（每核） | 64 Bytes | 4-way set associative |
| L2 | 1 MB（共享，每簇 512KB） | 64 Bytes | 16-way set associative |

**目标**：卷积内层 tile 的工作集 ≤ L1 Data Cache（32 KB）。

### 6.2 分块策略（以 Conv2D 为例）

原始 7 重循环：`N × OC × OH × OW × IC × KH × KW`

**分块维度选择**：

| 维度 | 原因 | Tile 大小 | 说明 |
|------|------|----------|------|
| OC | NEON 向量化 | `tc = 4` | 与 NCHWc block 大小一致，最内层向量化 |
| OW | 输出行分块，改善 output 局部性 | `tw = 4` | 4 个输出元素对应 4×NEON |
| IC | 减少 weight 的 working set | `tic = 8` | 平衡 L1 容量 |

**Tile 工作集估算**：

```
单次内层 tile 工作集（OC-tile × OW-tile × IC-tile × KH × KW）：
  output tile: tc × tw × sizeof(float32) = 4 × 4 × 4 = 64 Bytes
  input  tile: tic × (tw*SH+KH-1) × sizeof(float32) ≈ 8 × 6 × 4 = 192 Bytes
  weight tile: tc × tic × KH × KW × sizeof(float32) = 4 × 8 × 3 × 3 × 4 = 3456 Bytes
  Total ≈ 3.7 KB  ← 远小于 L1 32 KB，Cache 友好
```

### 6.3 Tile 变换规则

对 `For(oc, 0, OC, Serial, body)` 应用 tile_size=4：

```
原始：
  For(oc, 0, OC, Serial, body)

变换后：
  For(oc_outer, 0, OC/4, Serial,       ← 外层循环，步长 4
    For(oc_inner, 0, 4, Vectorized,     ← 内层循环，向量化
      body[oc := oc_outer * 4 + oc_inner]))
```

### 6.4 算法设计

```
StmtMutator::mutate_for(For node):
  if node.loop_var.name matches tiling_config:
    tile_size = tiling_config[node.loop_var.name]
    // 生成外层变量
    outer_var = Var(loop_var.name + "_outer", kInt64)
    inner_var = Var(loop_var.name + "_inner", kInt64)
    // 替换 body 中 loop_var 的所有引用
    //   loop_var → outer_var * tile_size + inner_var
    new_body = VarReplacer(loop_var,
                 Add(Mul(outer_var, tile_size), inner_var))
               .mutate(body)
    // 构造新的 For 结构
    inner_for = For(inner_var, 0, tile_size, Vectorized, new_body)
    outer_for = For(outer_var, 0, extent/tile_size, Serial, inner_for)
    return outer_for
  else:
    return ExprMutator::mutate_for(node)
```

**边界处理**：若 `OC % tile_size != 0`，在外层循环后添加剩余部分的 serial 循环（epilogue）。对于典型 CNN（OC 为 2 的幂次），此情况通常不发生。

### 6.5 Tiling 配置

```cpp
struct TilingConfig {
    // 每个循环变量名模式 → tile 大小
    // 支持通配：如 "oc" 匹配所有名为 "oc" 的循环变量
    std::unordered_map<std::string, int> tile_sizes;

    // 默认配置（针对 Cortex-A72）
    static TilingConfig default_arm_a72() {
        TilingConfig cfg;
        cfg.tile_sizes["oc"]  = 4;   // NEON float32x4
        cfg.tile_sizes["oc_"] = 4;   // 融合后的 oc 前缀
        cfg.tile_sizes["ow"]  = 4;   // 输出宽度
        cfg.tile_sizes["ic"]  = 8;   // 输入通道
        return cfg;
    }
};
```

### 6.6 核心类设计

```cpp
class LoopTiling : public Pass {
public:
    std::string name() const override { return "LoopTiling"; }
    int min_opt_level() const override { return 2; }

    // 作用于 LLIR
    Ref<LLIRModule> transform_llir(
        Ref<LLIRModule> mod,
        const PassContext& ctx
    ) override;

    // 从 PassContext config 中读取 tile 大小（可覆盖默认值）
    // 如：ctx.config["loop_tiling.oc_tile"] = "4"
    TilingConfig parse_tiling_config(const PassContext& ctx) const;

private:
    class TilingMutator : public StmtMutator {
        const TilingConfig& config_;
    public:
        Ref<Stmt> mutate_for(Ref<For> node) override;
    };

    // 将 loop_var 替换为 outer * tile + inner
    class VarReplacer : public StmtMutator {
        Ref<Var> target_;
        Ref<PrimExpr> replacement_;
    public:
        Ref<PrimExpr> mutate_var(Ref<Var> node) override;
    };
};
```

### 6.7 向量化与 Tiling 的协同

Loop Tiling 生成的内层 `For(kind=Vectorized)` 在 Codegen 阶段会被翻译为 NEON Intrinsics：

```
Tiling 生成：
  For(oc_inner, 0, 4, ForKind::Vectorized, ...)

Codegen 翻译：
  // oc_inner 循环展开，用 NEON 4×float32 向量操作替代
  float32x4_t acc = vld1q_f32(&output[n][oc_outer*4][oh][ow]);
  float32x4_t w   = vld1q_f32(&weight[oc_outer*4][ic][kh][kw]);
  float32x4_t in  = vdupq_n_f32(input[n][ic][oh*sh+kh][ow*sw+kw]);
  acc = vfmaq_f32(acc, in, w);
  vst1q_f32(&output[n][oc_outer*4][oh][ow], acc);
```
