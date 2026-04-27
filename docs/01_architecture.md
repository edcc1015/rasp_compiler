# 01 整体架构设计

## 目录

- [1. 系统架构概览](#1-系统架构概览)
- [2. 模块职责划分](#2-模块职责划分)
- [3. 模块依赖关系](#3-模块依赖关系)
- [4. 数据流与控制流](#4-数据流与控制流)
- [5. 工程目录结构](#5-工程目录结构)
- [6. 关键设计原则](#6-关键设计原则)

---

## 1. 系统架构概览

RASP Compiler 采用经典编译器分层架构，分为**前端（Frontend）**、**中端（Middle-end）**和**后端（Backend）**三个主要阶段，并以**两级 IR** 作为各阶段的数据载体。

```
┌─────────────────────────────────────────────────────────────────┐
│                         RASP Compiler                           │
│                                                                 │
│  ┌─────────────┐    ┌────────────────────────────────────────┐  │
│  │   Frontend  │    │              Middle-end                │  │
│  │  (Python)   │    │                                        │  │
│  │             │    │  ┌─────────────┐  ┌─────────────────┐  │  │
│  │ ONNX Parser │───▶│  │ High-level  │  │  Pass Manager   │  │  │
│  │             │    │  │     IR      │──▶  (Graph-level   │  │  │
│  │ PyTorch     │    │  │  (Relay-    │  │   Passes)       │  │  │
│  │  Parser     │    │  │   like)     │  └────────┬────────┘  │  │
│  └─────────────┘    │  └─────────────┘           │           │  │
│                     │                            │ Lowering  │  │
│                     │  ┌─────────────┐           │           │  │
│                     │  │  Low-level  │◀──────────┘           │  │
│                     │  │     IR      │                        │  │
│                     │  │  (TIR-like) │──▶ Loop Tiling Pass   │  │
│                     │  └─────────────┘                        │  │
│                     └────────────────────────────────────────┘  │
│                                                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                        Backend                             │ │
│  │                                                            │ │
│  │   Codegen (ARMv8-A + NEON)  ──▶  Runtime (C++ 执行引擎)   │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 模块职责划分

### 2.1 前端（Frontend）

**语言**：Python  
**输入**：ONNX `.onnx` 文件 / PyTorch `nn.Module` 对象  
**输出**：序列化的 High-level IRModule（JSON 或自定义二进制格式）

| 子模块 | 职责 |
|--------|------|
| `onnx_frontend.py` | 读取 ONNX protobuf，遍历 NodeProto，构建 High-level IR 节点 |
| `pytorch_frontend.py` | 调用 `torch.onnx.export` 将 PyTorch 模型转为 ONNX，再走 ONNX 前端 |
| `op_registry.py` | 维护 ONNX op → IR Op 的映射表，及参数属性转换规则 |

---

### 2.2 High-level IR（图级 IR）

**语言**：C++  
**描述**：计算图的代数表示，类似 TVM Relay。  
**核心概念**：`IRModule` 包含若干具名 `Function`，`Function` 由嵌套 `Expr` 节点（`Call`, `Var`, `Constant`, `Let`, `Tuple` 等）构成。

---

### 2.3 Pass 管理与图级优化

**语言**：C++  
**输入/输出**：`IRModule` → 优化后 `IRModule`  

| Pass | 级别 | 作用 |
|------|------|------|
| `ConstantFolding` | FunctionPass | 编译期折叠常量子图 |
| `OperatorFusion` | FunctionPass | 合并 Conv+BN+ReLU 等可融合链 |
| `DeadCodeElimination` | FunctionPass | 删除输出不可达的节点 |
| `LayoutTransformation` | FunctionPass | 将 NCHW 转为 NCHWc，优化 NEON 访存 |

---

### 2.4 Lowering 模块

**语言**：C++  
**输入**：优化后 High-level IRModule  
**输出**：Low-level IRModule（每个算子对应一个 `PrimFunc`）  
职责：将算子语义（Call 节点）展开为具体的多重循环 Buffer 访问。

---

### 2.5 Low-level IR（循环级 IR）

**语言**：C++  
**描述**：以循环、Buffer 读写为基本单元的命令式 IR，类似 TVM TIR。  
**核心概念**：`PrimFunc` 由 `Stmt` 树组成，`Stmt` 包含 `For`、`BufferStore`、`Allocate` 等节点。

---

### 2.6 循环级优化（Loop Tiling）

**语言**：C++  
**输入/输出**：`PrimFunc` → 分块后 `PrimFunc`  
针对 Cortex-A72 的 L1/L2 Cache 大小选择分块参数，改善数据局部性。

---

### 2.7 Codegen

**语言**：C++  
**输入**：优化后 Low-level IRModule  
**输出**：C++ 源文件（包含 ARM NEON Intrinsics 调用）  
遍历 `PrimFunc` 中的 `Stmt`/`PrimExpr` 节点，按规则发射 C++ 代码。

---

### 2.8 Runtime

**语言**：C++  
**职责**：加载编译产物，管理 Tensor 内存，提供统一推理调用接口。

---

## 3. 模块依赖关系

```
frontend（Python）
    │
    │ 序列化 IRModule
    ▼
high_level_ir（C++）◀──────────────────────────┐
    │                                          │
    ▼                                          │
pass_manager（C++）                           │
    │  ← 调用各 FunctionPass / ModulePass       │
    │  ConstantFolding                         │ 共享类型定义
    │  OperatorFusion                          │
    │  DCE                                     │
    │  LayoutTransformation                    │
    ▼                                          │
lowering（C++）                               │
    │  ← 算子 Lowering 规则                     │
    ▼                                          │
low_level_ir（C++）◀──────────────────────────┘
    │
    ▼
loop_tiling_pass（C++）
    │
    ▼
codegen（C++）
    │
    ▼
runtime（C++）
```

---

## 4. 数据流与控制流

### 4.1 数据流（IRModule 变换链）

```
ONNX/PyTorch 文件
    │
    │ Python 解析
    ▼
HLIRModule（未优化）
    │
    │ ConstantFolding Pass
    ▼
HLIRModule（常量已折叠）
    │
    │ OperatorFusion Pass
    ▼
HLIRModule（算子已融合）
    │
    │ DCE Pass
    ▼
HLIRModule（死代码已删除）
    │
    │ LayoutTransformation Pass
    ▼
HLIRModule（Layout 已转换）
    │
    │ Lowering
    ▼
LLIRModule（PrimFunc 集合）
    │
    │ LoopTiling Pass
    ▼
LLIRModule（分块后 PrimFunc）
    │
    │ Codegen
    ▼
C++ 源文件（含 NEON Intrinsics）
    │
    │ 编译（clang++ / g++）
    ▼
共享库 / 可执行文件
    │
    │ Runtime 加载
    ▼
推理结果 Tensor
```

### 4.2 控制流

编译过程由 **驱动程序（Driver）** 统一调度：

```
Driver::compile(model_path, output_path)
    ├── Frontend::parse(model_path)       → HLIRModule
    ├── PassManager::run(hlir, ctx)       → HLIRModule（优化）
    ├── Lowering::lower(hlir)             → LLIRModule
    ├── PassManager::run(llir, ctx)       → LLIRModule（Loop Tiling）
    └── Codegen::generate(llir, output_path)
```

---

## 5. 工程目录结构

```
rasp_compiler/
│
├── frontend/                      # Python 前端（模型解析）
│   ├── __init__.py
│   ├── onnx_frontend.py           # ONNX 解析器
│   ├── pytorch_frontend.py        # PyTorch 前端（转 ONNX 后解析）
│   └── op_registry.py             # ONNX Op → IR Op 映射表
│
├── include/                       # C++ 公共头文件
│   ├── ir/
│   │   ├── ir_node.h              # IRNode 基类
│   │   ├── expr.h                 # High-level IR 表达式节点
│   │   ├── type.h                 # 类型系统
│   │   ├── op.h                   # Op 注册
│   │   ├── ir_module.h            # IRModule
│   │   ├── prim_expr.h            # Low-level IR 表达式节点
│   │   ├── stmt.h                 # Low-level IR 语句节点
│   │   ├── buffer.h               # Buffer 描述符
│   │   └── prim_func.h            # PrimFunc
│   ├── pass/
│   │   ├── pass.h                 # Pass 抽象基类
│   │   ├── pass_context.h         # PassContext
│   │   ├── pass_manager.h         # PassManager
│   │   ├── constant_folding.h
│   │   ├── operator_fusion.h
│   │   ├── dead_code_elimination.h
│   │   ├── layout_transformation.h
│   │   └── loop_tiling.h
│   ├── lowering/
│   │   ├── lowering.h             # Lowering 主接口
│   │   └── op_lowering_rules.h    # 各算子 Lowering 规则
│   └── codegen/
│       ├── codegen.h              # Codegen 主接口
│       └── neon_intrinsics.h      # NEON Intrinsics 封装
│
├── src/                           # C++ 实现源文件
│   ├── ir/
│   │   ├── ir_node.cpp
│   │   ├── expr.cpp
│   │   ├── type.cpp
│   │   ├── op.cpp
│   │   ├── ir_module.cpp
│   │   ├── prim_expr.cpp
│   │   ├── stmt.cpp
│   │   ├── buffer.cpp
│   │   └── prim_func.cpp
│   ├── pass/
│   │   ├── pass_manager.cpp
│   │   ├── constant_folding.cpp
│   │   ├── operator_fusion.cpp
│   │   ├── dead_code_elimination.cpp
│   │   ├── layout_transformation.cpp
│   │   └── loop_tiling.cpp
│   ├── lowering/
│   │   ├── lowering.cpp
│   │   └── op_lowering_rules.cpp
│   └── codegen/
│       └── codegen.cpp
│
├── runtime/                       # 推理运行时
│   ├── runtime.h
│   ├── runtime.cpp
│   └── tensor.h
│
├── driver/                        # 编译驱动入口
│   └── compiler_driver.cpp
│
├── tests/                         # 单元测试
│   ├── test_ir.cpp
│   ├── test_passes.cpp
│   ├── test_lowering.cpp
│   └── test_codegen.cpp
│
├── docs/                          # 设计文档
│
└── CMakeLists.txt
```

---

## 6. 关键设计原则

### 6.1 两级 IR 分离

高层 IR 描述**算子语义**（做什么），低层 IR 描述**执行方式**（怎么做）。二者严格分离，Lowering 作为唯一转换通道，保证各阶段职责单一。

### 6.2 Pass 的纯函数语义

每个 Pass 接收 `IRModule`，返回新的 `IRModule`，不做原地修改。这保证了 Pass 的可测试性与可组合性。

### 6.3 节点不可变（Immutable Nodes）

IR 节点一经创建不可修改（类似 LLVM Value）。变换时通过构造新节点替换旧节点，利用引用计数（`std::shared_ptr`）管理生命周期。

### 6.4 算子注册与解耦

所有算子通过 `OpRegistry` 注册，前端、Pass、Lowering 均通过名称查找算子，不硬编码特定算子逻辑，方便后续扩展。

### 6.5 模块间通信以 IR 为边界

各模块（Frontend、PassManager、Lowering、Codegen）之间**只通过 IRModule 传递数据**，不共享内部状态，降低耦合。
