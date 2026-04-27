# RASP Compiler 设计文档

> 一个面向树莓派 4B（ARMv8-A + NEON SIMD）的精简深度学习编译器，支持 ONNX/PyTorch CNN 模型的解析、优化与代码生成。

---

## 项目背景

深度学习推理部署在边缘设备上面临性能瓶颈。TVM 等工业级编译器虽然功能全面，但体量庞大、二次开发门槛高。RASP Compiler 以 TVM 为参考原型，实现一个功能完整、架构清晰的精简版深度学习编译器，专为树莓派 4B（Cortex-A72，ARMv8-A，NEON SIMD）优化。

**设计目标：**

- 支持解析 ONNX 和 PyTorch 导出的 CNN 模型
- 设计两级 IR（图级 + 循环级），支持图优化与循环变换
- 实现五种关键优化 Pass
- 针对 ARMv8-A + NEON SIMD 生成高效 C++ 推理代码
- 架构清晰，模块解耦，易于扩展

---

## 编译流水线总览

```
ONNX 模型 / PyTorch 模型
        │
        ▼
┌───────────────────┐
│   前端解析（Python） │  ← onnx / torch.onnx.export
└────────┬──────────┘
         │ High-level IRModule
         ▼
┌───────────────────┐
│   高层 IR（图级）  │  ← 类 TVM Relay，表达算子计算图
└────────┬──────────┘
         │
         ▼
┌───────────────────────────────────────────────┐
│              Pass 管理器（PassManager）         │
│  ┌──────────────┐  ┌──────────────────────┐   │
│  │ ConstFolding │  │  OperatorFusion      │   │
│  ├──────────────┤  ├──────────────────────┤   │
│  │    DCE       │  │  LayoutTransformation│   │
│  └──────────────┘  └──────────────────────┘   │
└────────┬──────────────────────────────────────┘
         │ 优化后 High-level IRModule
         ▼
┌───────────────────┐
│   Lowering 模块   │  ← 图级 IR → 循环级 IR
└────────┬──────────┘
         │ Low-level IRModule（PrimFunc）
         ▼
┌───────────────────┐
│  循环级优化 Pass  │  ← Loop Tiling
└────────┬──────────┘
         │
         ▼
┌───────────────────────┐
│  Codegen（ARMv8-A）   │  ← Low-level IR → NEON Intrinsics C++ 代码
└────────┬──────────────┘
         │
         ▼
┌───────────────────┐
│   Runtime 执行    │  ← 模型加载 + Tensor 推理接口
└───────────────────┘
```

---

## 文档目录大纲

| 文档 | 内容 |
|------|------|
| [01 整体架构设计](01_architecture.md) | 模块依赖关系、数据流、工程目录结构 |
| [02 前端解析模块](02_frontend.md) | ONNX/PyTorch 解析、算子映射、前端→IR 转换 |
| [03 高层 IR 设计](03_high_level_ir.md) | 图级 IR 数据结构（Relay-like）、类型系统、C++ 设计 |
| [04 低层 IR 设计](04_low_level_ir.md) | 循环级 IR 数据结构（TIR-like）、循环原语、C++ 设计 |
| [05 Pass 管理框架](05_pass_manager.md) | PassContext、Pass 抽象接口、PassManager、Pipeline |
| [06 优化 Pass 设计](06_optimization_passes.md) | ConstFolding、OpFusion、DCE、LayoutTransform、LoopTiling |
| [07 Lowering 模块](07_lowering.md) | 图级→循环级 Lowering 流程、算子映射、Conv2D 完整示例 |
| [08 Codegen 与 Runtime](08_codegen.md) | ARMv8-A + NEON Codegen 框架、Intrinsics 映射、Runtime 接口 |

---

## 技术约束

| 维度 | 决策 |
|------|------|
| 目标硬件 | 树莓派 4B（Cortex-A72，ARMv8-A，NEON SIMD，4-core） |
| 精度支持 | FP32 只读推理 |
| 算子覆盖 | CNN 常用算子（Conv2d, BatchNorm, ReLU, Pooling, FC, Concat, Add） |
| 前端语言 | Python（onnx, torch 库） |
| 核心语言 | C++17 |
| 优化级别 | Graph-level（4 Pass）+ Loop-level（Loop Tiling） |
| 向量化 | ARM NEON Intrinsics（128-bit，4×FP32/vector） |

---

## 工程目录预览

```
rasp_compiler/
├── frontend/           # Python 前端解析
│   ├── onnx_frontend.py
│   └── pytorch_frontend.py
├── include/
│   ├── ir/             # IR 头文件
│   ├── pass/           # Pass 头文件
│   ├── lowering/       # Lowering 头文件
│   └── codegen/        # Codegen 头文件
├── src/
│   ├── ir/
│   ├── pass/
│   ├── lowering/
│   └── codegen/
├── runtime/            # Runtime 头文件与实现
├── tests/              # 单元测试
├── docs/               # 设计文档（本目录）
└── CMakeLists.txt
```
