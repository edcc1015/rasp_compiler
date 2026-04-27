# 08 Codegen 与 Runtime

## 目录

- [1. 设计目标](#1-设计目标)
- [2. 目标架构参数（Cortex-A72）](#2-目标架构参数cortex-a72)
- [3. Codegen 总体框架](#3-codegen-总体框架)
- [4. Stmt/PrimExpr → C++ 代码映射规则](#4-stmtprimexpr--c-代码映射规则)
- [5. NEON Intrinsics 映射表](#5-neon-intrinsics-映射表)
- [6. 向量化 For 循环的代码生成](#6-向量化-for-循环的代码生成)
- [7. 内存对齐策略](#7-内存对齐策略)
- [8. 生成代码结构示例](#8-生成代码结构示例)
- [9. Runtime 设计](#9-runtime-设计)
- [10. Codegen 完整接口设计](#10-codegen-完整接口设计)

---

## 1. 设计目标

Codegen 模块将 LLIR 的 `LLIRModule` 转换为可在树莓派 4B 上运行的 **C++ 源文件**，该文件包含：

- 每个 `PrimFunc` 对应的 C++ 函数
- 使用 ARM NEON Intrinsics 实现向量化循环
- 正确的内存对齐声明
- 一个顶层推理函数（按拓扑序调用各 PrimFunc）
- 与 Runtime 对接的公共接口（`extern "C"` 函数）

生成的 C++ 文件使用 `arm-linux-gnueabihf-g++` 或 `clang++ --target=aarch64-linux-gnu` 编译，链接后得到共享库 `.so` 或可执行文件。

---

## 2. 目标架构参数（Cortex-A72）

### 2.1 处理器规格

| 参数 | 值 |
|------|-----|
| 处理器型号 | ARM Cortex-A72（树莓派 4B） |
| ISA | ARMv8-A（64-bit，AArch64） |
| SIMD 扩展 | ARM NEON（Advanced SIMD） |
| 核心数 | 4 核 |
| L1 I-Cache | 48 KB（每核） |
| L1 D-Cache | 32 KB（每核） |
| L2 Cache | 1 MB（每簇，2核共享） |
| Cache Line | 64 Bytes |
| NEON 寄存器 | 32 × 128-bit V 寄存器 |
| FP32 NEON 宽度 | 4 × float32 / 向量 |
| 峰值 FP32 吞吐 | 4 FLOPs/cycle/core（FMA 算 2 FLOPs） |

### 2.2 NEON 数据类型

| C 类型 | 含义 | 用途 |
|--------|------|------|
| `float32x4_t` | 4 × float32 向量 | 主要推理数据类型 |
| `int32x4_t` | 4 × int32 向量 | 索引计算（辅助） |
| `uint8x16_t` | 16 × uint8 向量 | 量化（本版本不涉及） |

---

## 3. Codegen 总体框架

### 3.1 Codegen 类设计

```cpp
// include/codegen/codegen.h

class CodegenContext {
public:
    // 输出代码流
    std::ostringstream code;

    // 当前缩进级别
    int indent_level = 0;

    // 循环变量名栈（用于向量化索引生成）
    std::vector<std::string> vec_loop_vars;

    // Buffer 名 → C++ 变量名映射
    std::unordered_map<std::string, std::string> buf_name_map;

    // 辅助：发射一行代码（自动缩进）
    void emit(const std::string& line);

    // 辅助：增加/减少缩进
    void indent()   { indent_level++; }
    void dedent()   { indent_level--; }

    // 辅助：获取当前缩进字符串
    std::string get_indent() const;
};

class Codegen {
public:
    // 将 LLIRModule 生成为 C++ 源文件
    static void generate(Ref<LLIRModule> mod, const std::string& output_path);

private:
    CodegenContext ctx_;

    // 生成文件头（include、namespace、前置声明）
    void emit_header();

    // 生成所有 PrimFunc
    void emit_all_prim_funcs(Ref<LLIRModule> mod);

    // 生成单个 PrimFunc
    void emit_prim_func(Ref<PrimFunc> func);

    // 生成 Stmt
    void emit_stmt(Ref<Stmt> stmt);

    // 生成 PrimExpr（返回字符串形式的 C++ 表达式）
    std::string emit_expr(Ref<PrimExpr> expr);

    // 生成顶层推理函数
    void emit_inference_func(Ref<LLIRModule> mod);

    // 生成 Buffer 函数参数声明
    std::string emit_buffer_param(Ref<Buffer> buf, bool is_const = false);
};
```

### 3.2 生成流程

```
generate(mod, output_path):
  1. emit_header()
     → #include <arm_neon.h>
     → #include <cmath>, <cstring>, <algorithm>
     → 宏定义（ALIGN16 等）

  2. for each PrimFunc in mod.exec_order:
       emit_prim_func(func)

  3. emit_inference_func(mod)
     → 按 exec_order 调用各 PrimFunc
     → extern "C" run(inputs, outputs) 接口

  4. 写入 output_path
```

---

## 4. Stmt/PrimExpr → C++ 代码映射规则

### 4.1 Stmt 节点映射

| LLIR Stmt | 生成 C++ 代码 |
|-----------|-------------|
| `SeqStmt([s1, s2, ...])` | 顺序发射每个语句 |
| `For(v, 0, N, Serial, body)` | `for (int64_t v = 0; v < N; ++v) { body }` |
| `For(v, 0, N, Vectorized, body)` | 展开为 NEON 向量操作（见第 6 节） |
| `For(v, 0, N, Unrolled, body)` | `#pragma GCC unroll N` + serial for |
| `IfThenElse(cond, then, else)` | `if (cond) { then } else { else }` |
| `Allocate(var, dtype, extents, body)` | `float var[extent0*extent1*...] __attribute__((aligned(16)));` |
| `BufferStore(buf, indices, val)` | `buf_ptr[linearize(indices)] = val;` |
| `Evaluate(expr)` | `expr;` |

### 4.2 PrimExpr 节点映射

| LLIR PrimExpr | 生成 C++ 表达式 |
|--------------|---------------|
| `IntImm(v)` | `int64_t(v)` 或直接 `v`（视上下文） |
| `FloatImm(v)` | `float(v)` 或直接 `vf` |
| `Var(name)` | `name`（直接使用变量名） |
| `BufferLoad(buf, indices)` | `buf_ptr[linearize(indices)]` |
| `Add(a, b)` | `(a + b)` |
| `Sub(a, b)` | `(a - b)` |
| `Mul(a, b)` | `(a * b)` |
| `Div(a, b)` | `(a / b)` |
| `FloorDiv(a, b)` | `((a) / (b))` （整数除法向下取整） |
| `FloorMod(a, b)` | `((a) % (b))` |
| `Min(a, b)` | `std::min(a, b)` |
| `Max(a, b)` | `std::max(a, b)` |
| `Select(c, t, f)` | `(c) ? (t) : (f)` |
| `Cast(dtype, val)` | `(float)(val)` 或 `(int64_t)(val)` |
| `Call(kIntrinsic, name, args)` | NEON Intrinsic 调用（见第 5 节） |

### 4.3 多维索引线性化

Buffer 以一维数组存储，多维索引线性化为偏移量：

```cpp
// Row-major（C-order）线性化
// output[n][oc][oh][ow] → output + n*OC*OH*OW + oc*OH*OW + oh*OW + ow
std::string linearize_indices(Ref<Buffer> buf,
                               const std::vector<Ref<PrimExpr>>& indices) {
    // shape = [N, OC, OH, OW]
    // strides = [OC*OH*OW, OH*OW, OW, 1]
    std::string offset = "0";
    int64_t stride = 1;
    for (int i = indices.size() - 1; i >= 0; --i) {
        offset = "(" + emit_expr(indices[i]) + " * " +
                 std::to_string(stride) + " + " + offset + ")";
        stride *= buf->shape[i]; // 编译期已知 shape
    }
    return offset;
}
```

---

## 5. NEON Intrinsics 映射表

以下列出 FP32 推理中使用的核心 NEON Intrinsics：

### 5.1 加载/存储

| 操作 | NEON Intrinsic | 说明 |
|------|---------------|------|
| 加载 4×FP32 | `vld1q_f32(ptr)` | 从连续内存加载 128-bit |
| 存储 4×FP32 | `vst1q_f32(ptr, v)` | 写回连续内存 |
| 加载（非对齐） | `vld1q_f32(ptr)` | AArch64 不区分对齐/非对齐 |
| 广播标量 | `vdupq_n_f32(scalar)` | 将 float 广播为 `[s,s,s,s]` |
| 设置零向量 | `vdupq_n_f32(0.0f)` | 初始化累加器 |

### 5.2 算术运算

| 操作 | NEON Intrinsic | 等效运算 |
|------|---------------|---------|
| 向量加法 | `vaddq_f32(a, b)` | `a[i] + b[i]` |
| 向量减法 | `vsubq_f32(a, b)` | `a[i] - b[i]` |
| 向量乘法 | `vmulq_f32(a, b)` | `a[i] * b[i]` |
| 乘累加（FMA） | `vfmaq_f32(acc, a, b)` | `acc[i] + a[i]*b[i]` |
| 标量乘向量 | `vmulq_n_f32(v, s)` | `v[i] * s` |
| 标量 FMA | `vfmaq_n_f32(acc, v, s)` | `acc[i] + v[i]*s` |
| 向量除法 | `vdivq_f32(a, b)` | `a[i] / b[i]`（AArch64 支持） |

### 5.3 比较与激活

| 操作 | NEON Intrinsic | 用途 |
|------|---------------|------|
| 逐元素最大值 | `vmaxq_f32(a, b)` | ReLU：`max(0, x)` |
| 逐元素最小值 | `vminq_f32(a, b)` | Clip 上界 |
| 绝对值 | `vabsq_f32(a)` | — |
| 取负 | `vnegq_f32(a)` | — |

### 5.4 规约运算

| 操作 | NEON Intrinsic | 用途 |
|------|---------------|------|
| 水平加（4→1） | `vaddvq_f32(v)` | 向量归约求和（AArch64） |
| 水平最大 | `vmaxvq_f32(v)` | — |

### 5.5 类型转换

| 操作 | NEON Intrinsic | 说明 |
|------|---------------|------|
| int32→float32 | `vcvtq_f32_s32(v)` | 整数转浮点 |
| float32→int32 | `vcvtq_s32_f32(v)` | 浮点转整数（截断） |

---

## 6. 向量化 For 循环的代码生成

当 Codegen 遇到 `For(kind=Vectorized)` 节点时，采用以下策略：

### 6.1 一般向量化策略

对于 `For(oc_inner, 0, 4, Vectorized, body)` 中的 `body`：

1. 将循环体内的 `BufferStore(buf, [n, oc_outer*4+oc_inner, oh, ow], val)` 转化为向量存储：
   - 索引基址 = `oc_outer*4`，宽度 = 4
   - 生成 `vst1q_f32(buf_ptr + base_offset, vec_val)`

2. 将 `BufferLoad(buf, [n, oc_outer*4+oc_inner, oh, ow])` 转化为向量加载：
   - 同样使用 `vld1q_f32(buf_ptr + base_offset)`

3. 将算术运算提升为向量运算（`Add → vaddq_f32`，`Mul → vmulq_f32`，`Max → vmaxq_f32`）

4. `Broadcast(scalar, 4)` → `vdupq_n_f32(scalar)`

### 6.2 生成代码示例

**原 LLIR**（向量化循环）：

```
For(oc_inner, 0, 4, Vectorized,
  BufferStore(output, [n, oc_outer*4+oc_inner, oh, ow],
    vfma(
      BufferLoad(output, [n, oc_outer*4+oc_inner, oh, ow]),
      BufferLoad(input, [n, ic, oh*sh+kh, ow*sw+kw]),
      BufferLoad(weight, [oc_outer*4+oc_inner, ic, kh, kw])
    )
  )
)
```

**生成 C++ 代码**：

```cpp
// For(oc_inner, 0, 4, Vectorized) → NEON 向量化展开
{
    float* __restrict__ out_ptr = output + n*OC*OH*OW + oc_outer*4*OH*OW + oh*OW + ow;
    float* __restrict__ w_ptr   = weight + (oc_outer*4)*IC*KH*KW + ic*KH*KW + kh*KW + kw;
    float  in_val = input[n*IC*IH*IW + ic*IH*IW + (oh*SH+kh)*IW + (ow*SW+kw)];

    float32x4_t acc  = vld1q_f32(out_ptr);    // 加载 4 个输出累加值
    float32x4_t w    = vld1q_f32(w_ptr);      // 加载 4 个权重
    float32x4_t inp  = vdupq_n_f32(in_val);   // 广播输入标量

    acc = vfmaq_f32(acc, inp, w);              // FMA: acc += inp * w
    vst1q_f32(out_ptr, acc);                   // 存回
}
```

### 6.3 激活融合（ReLU 向量化）

融合算子中的 ReLU 直接在向量寄存器上操作：

```cpp
// ReLU 向量化：max(0, val)
float32x4_t zero = vdupq_n_f32(0.0f);
acc = vmaxq_f32(acc, zero);
```

---

## 7. 内存对齐策略

### 7.1 Buffer 对齐声明

所有动态分配的 Buffer（输入/输出/权重）需 16 字节对齐以满足 NEON `vld1q_f32` 要求：

```cpp
// 静态局部 Buffer（Allocate 节点）
float local_buf[256] __attribute__((aligned(16)));

// 动态分配（Runtime 层）
float* buf = static_cast<float*>(
    std::aligned_alloc(16, size * sizeof(float))
);
```

### 7.2 函数参数对齐注解

生成的 PrimFunc 使用 `__restrict__` 和对齐注解提示编译器：

```cpp
void conv2d_0(
    const float* __restrict__ data,    // 假设 16-byte aligned
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float*       __restrict__ output
);
```

### 7.3 权重预对齐

权重在编译阶段（Codegen）直接嵌入为对齐的 C++ 数组：

```cpp
// 生成的权重常量（静态数组）
static const float __attribute__((aligned(16)))
    conv2d_0_weight[128 * 64 * 3 * 3] = { ... };
```

---

## 8. 生成代码结构示例

以 Conv2D → ReLU 融合后的典型生成代码为例（精简版）：

```cpp
// === 自动生成 by RASP Compiler ===
// 目标：ARM Cortex-A72, ARMv8-A, NEON FP32

#include <arm_neon.h>
#include <cstring>
#include <algorithm>
#include <cstdint>

// --- 权重常量 ---
static const float __attribute__((aligned(16)))
    conv2d_0_weight[128*64*3*3] = { /* ... 编译期嵌入 ... */ };
static const float __attribute__((aligned(16)))
    conv2d_0_bias[128] = { /* ... */ };

// --- Conv2D + ReLU 融合算子 ---
void conv2d_relu_0(
    const float* __restrict__ data,   // [1, 64, 56, 56]
    float*       __restrict__ output  // [1, 128, 56, 56]
) {
    const int N=1, IC=64, OC=128, IH=56, IW=56, OH=56, OW=56;
    const int KH=3, KW=3, SH=1, SW=1, PH=1, PW=1;

    for (int64_t n = 0; n < N; ++n) {
    for (int64_t oc_o = 0; oc_o < OC/4; ++oc_o) {
    for (int64_t oh = 0; oh < OH; ++oh) {
    for (int64_t ow = 0; ow < OW; ++ow) {
        // 初始化累加器（4×float32）
        float32x4_t acc = vld1q_f32(conv2d_0_bias + oc_o*4);

        for (int64_t ic = 0; ic < IC; ++ic) {
        for (int64_t kh = 0; kh < KH; ++kh) {
        for (int64_t kw = 0; kw < KW; ++kw) {
            int64_t ih = oh*SH + kh - PH;
            int64_t iw = ow*SW + kw - PW;
            if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                float in_val = data[n*IC*IH*IW + ic*IH*IW + ih*IW + iw];
                float32x4_t inp = vdupq_n_f32(in_val);
                float32x4_t w   = vld1q_f32(
                    conv2d_0_weight + (oc_o*4)*IC*KH*KW + ic*KH*KW + kh*KW + kw);
                acc = vfmaq_f32(acc, inp, w);
            }
        }}}

        // ReLU 融合
        float32x4_t zero = vdupq_n_f32(0.0f);
        acc = vmaxq_f32(acc, zero);

        // 写回输出
        float* out_ptr = output + n*OC*OH*OW + oc_o*4*OH*OW + oh*OW + ow;
        vst1q_f32(out_ptr, acc);
    }}}}
}

// --- 顶层推理函数 ---
extern "C" void model_run(
    const float* input,   // [1, 64, 56, 56]
    float*       output   // [1, 128, 56, 56]
) {
    conv2d_relu_0(input, output);
    // 更多算子...
}
```

---

## 9. Runtime 设计

Runtime 负责加载编译产物、管理张量内存、提供推理接口。

### 9.1 Tensor

```cpp
// runtime/tensor.h

enum class DType {
    kFloat32,
    kInt32,
    kInt64,
};

struct Tensor {
    void*                    data;      // 数据指针（16-byte 对齐）
    std::vector<int64_t>     shape;     // 形状
    DType                    dtype;     // 数据类型
    bool                     owned;     // 是否由 Tensor 管理内存

    Tensor() : data(nullptr), owned(false) {}

    // 分配新张量（管理内存）
    static Tensor allocate(const std::vector<int64_t>& shape, DType dtype);

    // 包装外部内存（不管理）
    static Tensor from_data(void* data,
                             const std::vector<int64_t>& shape,
                             DType dtype);

    ~Tensor();

    // 禁止拷贝，允许移动
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&&) noexcept;
    Tensor& operator=(Tensor&&) noexcept;

    // 辅助
    int64_t numel() const;
    size_t  nbytes() const;
    float*  data_f32() const { return static_cast<float*>(data); }
};
```

### 9.2 Runtime 接口

```cpp
// runtime/runtime.h

class Runtime {
public:
    Runtime() = default;
    ~Runtime();

    // 加载共享库（由 Codegen 编译后生成）
    // lib_path: path to .so file
    bool load(const std::string& lib_path);

    // 执行推理
    // inputs:  输入张量列表（顺序与模型输入一致）
    // outputs: 输出张量列表（由 Runtime 分配）
    bool run(const std::vector<Tensor>& inputs,
              std::vector<Tensor>& outputs);

    // 获取错误信息
    std::string last_error() const { return last_error_; }

    // 查询模型元信息（由编译器嵌入共享库）
    struct ModelMeta {
        std::vector<std::vector<int64_t>> input_shapes;
        std::vector<std::vector<int64_t>> output_shapes;
        std::string model_name;
    };
    ModelMeta get_meta() const;

private:
    void*  lib_handle_ = nullptr;   // dlopen 句柄

    // 函数指针（从共享库加载）
    using RunFunc = void(*)(const float** inputs, float** outputs);
    RunFunc run_func_ = nullptr;

    // 元信息加载函数
    using MetaFunc = void(*)(ModelMeta*);
    MetaFunc meta_func_ = nullptr;

    std::string last_error_;
};
```

### 9.3 Runtime 使用示例

```cpp
// 1. 编译模型（编译期，通常在 x86 开发机上交叉编译）
// rasp_compiler resnet18.onnx -o resnet18.so

// 2. 在树莓派上加载运行
Runtime rt;
if (!rt.load("resnet18.so")) {
    std::cerr << rt.last_error() << std::endl;
    return -1;
}

// 3. 准备输入 Tensor
Tensor input = Tensor::from_data(
    image_data,
    {1, 3, 224, 224},
    DType::kFloat32
);

// 4. 执行推理
std::vector<Tensor> inputs = {std::move(input)};
std::vector<Tensor> outputs;
rt.run(inputs, outputs);

// 5. 读取输出
float* logits = outputs[0].data_f32();
// ... 后处理（argmax 等）
```

### 9.4 内存管理

- 输入 Tensor：由用户提供，Runtime 不负责释放
- 输出 Tensor：由 Runtime 分配（`std::aligned_alloc`），用户使用后 `Tensor` 析构自动释放
- 权重：嵌入共享库的静态数组（`.rodata` 段），由 OS 管理

---

## 10. Codegen 完整接口设计

### 10.1 编译驱动入口

```cpp
// driver/compiler_driver.h

struct CompilerOptions {
    int         opt_level     = 2;
    std::string target        = "aarch64-linux-gnu";   // 交叉编译目标
    bool        embed_weights = true;     // 将权重嵌入生成文件
    bool        dump_ir       = false;
    std::string dump_dir      = "/tmp/rasp_ir_dump";
};

class CompilerDriver {
public:
    // 从 ONNX 文件编译到共享库
    static bool compile(
        const std::string& model_path,
        const std::string& output_so,
        const CompilerOptions& opts = {}
    );

    // 只生成 C++ 源文件（不调用编译器）
    static bool generate_cpp(
        const std::string& model_path,
        const std::string& output_cpp,
        const CompilerOptions& opts = {}
    );
};
```

### 10.2 构建命令

生成 `.cpp` 后，使用以下命令交叉编译（在 x86 开发机上）：

```bash
# 安装交叉编译工具链
# sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

aarch64-linux-gnu-g++ \
    -O3 \
    -march=armv8-a+simd \
    -mfpu=neon-fp-armv8 \
    -ftree-vectorize \
    -shared -fPIC \
    -o resnet18.so \
    model_gen.cpp
```

或使用 `clang++` 交叉编译：

```bash
clang++ \
    --target=aarch64-linux-gnu \
    -O3 \
    -march=armv8-a+simd \
    -shared -fPIC \
    -o resnet18.so \
    model_gen.cpp
```

### 10.3 生成文件完整结构

```cpp
// ============================================================
// 自动生成文件：resnet18_gen.cpp
// 编译器：RASP Compiler v0.1
// 目标：AArch64 ARMv8-A + NEON
// ============================================================
#include <arm_neon.h>
#include <cstring>
#include <cstdint>
#include <algorithm>

// --- 权重（只读静态数组）---
static const float __attribute__((aligned(16))) conv2d_0_weight[...] = {...};
static const float __attribute__((aligned(16))) conv2d_0_bias[...]   = {...};
// ... 其他层权重

// --- 算子函数（PrimFunc 生成）---
static void conv2d_relu_0(const float* data, float* output) { ... }
static void maxpool_0(const float* data, float* output)     { ... }
// ...
static void dense_relu_0(const float* data, float* output)  { ... }
static void softmax_0(const float* data, float* output)     { ... }

// --- 中间 Buffer 分配 ---
static float __attribute__((aligned(16))) buf_0[1*64*112*112];
static float __attribute__((aligned(16))) buf_1[1*64*56*56];
// ...

// --- 顶层推理函数 ---
extern "C" void model_run(
    const float* input,   // [1, 3, 224, 224]
    float*       output   // [1, 1000]
) {
    conv2d_relu_0(input, buf_0);
    maxpool_0(buf_0, buf_1);
    // ... 按拓扑序调用
    softmax_0(buf_N, output);
}

// --- 元信息接口 ---
extern "C" void model_get_meta(ModelMeta* meta) {
    meta->model_name = "resnet18";
    meta->input_shapes  = {{1, 3, 224, 224}};
    meta->output_shapes = {{1, 1000}};
}
```
