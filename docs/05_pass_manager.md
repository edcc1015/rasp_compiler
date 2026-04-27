# 05 Pass 管理框架

## 目录

- [1. 设计概述](#1-设计概述)
- [2. PassContext](#2-passcontext)
- [3. Pass 抽象基类](#3-pass-抽象基类)
- [4. FunctionPass](#4-functionpass)
- [5. ModulePass](#5-modulepass)
- [6. PassManager](#6-passmanager)
- [7. Pass Pipeline 配置](#7-pass-pipeline-配置)
- [8. Pass 依赖与顺序管理](#8-pass-依赖与顺序管理)
- [9. Pass 注册系统](#9-pass-注册系统)
- [10. 调试与日志](#10-调试与日志)
- [11. 设计示例：完整 Pass 运行流程](#11-设计示例完整-pass-运行流程)

---

## 1. 设计概述

Pass 管理框架负责组织和执行一系列**图变换（Pass）**，将 IRModule 逐步优化。设计目标：

- **统一接口**：所有 Pass 实现相同接口，可被 PassManager 透明管理
- **纯函数语义**：Pass 不修改输入 IRModule，返回新的 IRModule
- **可组合**：Pass 可以任意顺序注册，支持嵌套 PassManager
- **可调试**：Pass 执行前后可 dump IR，便于排查优化错误
- **层次分离**：区分图级（FunctionPass / ModulePass）和循环级（作用于 LLIRModule 的 Pass）

Pass 框架分为两套独立实例：
1. **HLIR Pass 管理器**：管理作用于 High-level IRModule 的 Pass
2. **LLIR Pass 管理器**：管理作用于 Low-level LLIRModule 的 Pass（如 Loop Tiling）

---

## 2. PassContext

`PassContext` 携带 Pass 执行期间共享的全局配置与状态，以栈形式管理（类似 TVM 的 with PassContext）：

```cpp
// include/pass/pass_context.h

struct PassContext {
    // 优化等级（0=不优化，1=基础，2=激进）
    int opt_level = 2;

    // 强制启用的 Pass 名称（即使 opt_level 不满足要求也执行）
    std::vector<std::string> required_passes;

    // 强制禁用的 Pass 名称
    std::vector<std::string> disabled_passes;

    // 是否在每个 Pass 后 dump IR（调试用）
    bool dump_ir = false;

    // dump IR 的输出目录
    std::string dump_dir = "/tmp/rasp_ir_dump";

    // 自定义配置（供特定 Pass 读取）
    std::unordered_map<std::string, std::string> config;

    // 快速查询：某 Pass 是否被禁用
    bool is_disabled(const std::string& pass_name) const;

    // 读取自定义配置（类型安全）
    template <typename T>
    T get_config(const std::string& key, T default_val) const;
};
```

**使用示例**：

```cpp
PassContext ctx;
ctx.opt_level = 2;
ctx.disabled_passes = {"LayoutTransformation"};
ctx.dump_ir = true;

auto result = pass_manager.run(module, ctx);
```

---

## 3. Pass 抽象基类

```cpp
// include/pass/pass.h

class Pass {
public:
    virtual ~Pass() = default;

    // Pass 唯一名称（用于注册、日志、依赖声明）
    virtual std::string name() const = 0;

    // 最低优化等级要求（opt_level < min_opt_level 时跳过此 Pass）
    virtual int min_opt_level() const { return 0; }

    // 此 Pass 依赖哪些 Pass 必须先执行（用于依赖检查）
    virtual std::vector<std::string> dependencies() const { return {}; }

    // 执行变换（对 HLIR Module）
    // 子类选择性实现：FunctionPass 不直接实现此方法
    virtual Ref<IRModule> transform_hlir(Ref<IRModule> mod,
                                          const PassContext& ctx) {
        return mod;  // 默认：不变换
    }

    // 执行变换（对 LLIR Module）
    virtual Ref<LLIRModule> transform_llir(Ref<LLIRModule> mod,
                                            const PassContext& ctx) {
        return mod;
    }

protected:
    Pass() = default;
};
```

---

## 4. FunctionPass

`FunctionPass` 是作用于单个 `Function`（HLIR）的 Pass，PassManager 负责遍历 IRModule 中的所有函数并依次调用：

```cpp
// include/pass/pass.h

class FunctionPass : public Pass {
public:
    // 子类实现：对单个 Function 进行变换
    virtual Ref<Function> transform_function(
        Ref<Function> func,
        Ref<IRModule> mod,
        const PassContext& ctx
    ) = 0;

    // PassManager 调用此方法：遍历所有函数并应用 transform_function
    Ref<IRModule> transform_hlir(
        Ref<IRModule> mod,
        const PassContext& ctx
    ) override final;
};

// 默认实现（在 pass_manager.cpp 中）：
Ref<IRModule> FunctionPass::transform_hlir(Ref<IRModule> mod,
                                            const PassContext& ctx) {
    auto new_mod = IRModule::make();
    for (auto& [name, func] : mod->functions) {
        auto new_func = transform_function(func, mod, ctx);
        new_mod->add_function(name, new_func);
    }
    return new_mod;
}
```

**FunctionPass 示例（ConstantFolding）**：

```cpp
class ConstantFolding : public FunctionPass {
public:
    std::string name() const override { return "ConstantFolding"; }

    Ref<Function> transform_function(
        Ref<Function> func,
        Ref<IRModule> mod,
        const PassContext& ctx
    ) override;
};
```

---

## 5. ModulePass

`ModulePass` 直接操作整个 IRModule，适用于需要跨函数分析（如全局常量传播）或需要增删函数的场景：

```cpp
class ModulePass : public Pass {
public:
    // 子类实现：直接变换 IRModule
    virtual Ref<IRModule> transform_module(
        Ref<IRModule> mod,
        const PassContext& ctx
    ) = 0;

    Ref<IRModule> transform_hlir(
        Ref<IRModule> mod,
        const PassContext& ctx
    ) override final {
        return transform_module(mod, ctx);
    }
};
```

**何时使用 ModulePass**：
- 需要在多个函数间共享信息（如全局常量表）
- 需要增加/删除/重命名函数
- Dead Code Elimination（需要从顶层 main 函数分析可达性）

---

## 6. PassManager

`PassManager` 管理 Pass 列表，按顺序执行，并处理跳过逻辑：

```cpp
// include/pass/pass_manager.h

class PassManager {
public:
    // 添加 Pass（HLIR Pass）
    void add_pass(std::shared_ptr<Pass> pass);

    // 批量添加
    void add_passes(std::initializer_list<std::shared_ptr<Pass>> passes);

    // 执行 HLIR 所有 Pass
    Ref<IRModule> run(Ref<IRModule> mod, const PassContext& ctx) const;

    // 执行 LLIR 所有 Pass
    Ref<LLIRModule> run_llir(Ref<LLIRModule> mod, const PassContext& ctx) const;

    // 清空 Pass 列表
    void clear();

    // 打印已注册的 Pass 列表（调试用）
    void print_pipeline() const;

private:
    std::vector<std::shared_ptr<Pass>> passes_;

    bool should_skip(const Pass& pass, const PassContext& ctx) const;
    void dump_ir_if_needed(Ref<IRModule> mod,
                            const std::string& pass_name,
                            const PassContext& ctx) const;
};
```

### 6.1 run 方法实现逻辑

```
PassManager::run(mod, ctx):
  for each pass in passes_:
    if ctx.is_disabled(pass.name()):
      log("Skipping (disabled): " + pass.name())
      continue
    if ctx.opt_level < pass.min_opt_level():
      log("Skipping (opt_level too low): " + pass.name())
      continue

    log("Running: " + pass.name())
    new_mod = pass.transform_hlir(mod, ctx)

    if ctx.dump_ir:
      dump_ir(new_mod, pass.name(), ctx.dump_dir)

    mod = new_mod

  return mod
```

### 6.2 嵌套 PassManager

PassManager 本身可以实现 `Pass` 接口，从而支持嵌套（如将一组 Pass 打包为一个阶段）：

```cpp
class SequentialPass : public ModulePass {
public:
    SequentialPass(std::vector<std::shared_ptr<Pass>> passes,
                   std::string name)
        : passes_(std::move(passes)), name_(std::move(name)) {}

    std::string name() const override { return name_; }

    Ref<IRModule> transform_module(Ref<IRModule> mod,
                                    const PassContext& ctx) override {
        PassManager inner;
        for (auto& p : passes_) inner.add_pass(p);
        return inner.run(mod, ctx);
    }

private:
    std::vector<std::shared_ptr<Pass>> passes_;
    std::string name_;
};
```

---

## 7. Pass Pipeline 配置

标准优化 Pipeline（HLIR 阶段）：

```cpp
// driver/compiler_driver.cpp

PassManager build_hlir_pass_pipeline(int opt_level) {
    PassManager pm;
    if (opt_level >= 1) {
        pm.add_pass(std::make_shared<ConstantFolding>());
        pm.add_pass(std::make_shared<DeadCodeElimination>());
    }
    if (opt_level >= 2) {
        pm.add_pass(std::make_shared<OperatorFusion>());
        pm.add_pass(std::make_shared<ConstantFolding>());    // 融合后再做一次常量折叠
        pm.add_pass(std::make_shared<DeadCodeElimination>()); // 融合后再做一次 DCE
        pm.add_pass(std::make_shared<LayoutTransformation>());
    }
    return pm;
}

PassManager build_llir_pass_pipeline(int opt_level) {
    PassManager pm;
    if (opt_level >= 2) {
        pm.add_pass(std::make_shared<LoopTiling>());
    }
    return pm;
}
```

**标准 HLIR Pipeline 执行顺序**：

```
ConstantFolding          → 尽早折叠，减少后续处理的节点数量
DeadCodeElimination      → 删除因常量折叠产生的死节点
OperatorFusion           → 在干净的图上执行融合，避免死节点干扰模式匹配
ConstantFolding          → 融合后可能产生新的常量计算（如 BN 的参数融合进 Conv）
DeadCodeElimination      → 清理融合产生的孤立子图
LayoutTransformation     → 最后做布局转换，避免与融合规则冲突
```

---

## 8. Pass 依赖与顺序管理

当前采用**简单顺序约束**（非自动依赖解析）：

- Pass 在 `dependencies()` 中声明前置依赖
- PassManager 在 `run()` 前调用 `verify_dependencies()` 检查所有依赖是否已添加
- 若依赖未满足，抛出 `std::runtime_error` 提示用户修正 Pipeline

```cpp
void PassManager::verify_dependencies() const {
    std::unordered_set<std::string> seen;
    for (auto& pass : passes_) {
        for (auto& dep : pass->dependencies()) {
            if (seen.find(dep) == seen.end()) {
                throw std::runtime_error(
                    "Pass '" + pass->name() +
                    "' depends on '" + dep + "' which hasn't been added before it."
                );
            }
        }
        seen.insert(pass->name());
    }
}
```

---

## 9. Pass 注册系统

为便于通过配置文件或命令行动态组装 Pipeline，提供全局 Pass 注册表：

```cpp
// include/pass/pass_registry.h

class PassRegistry {
public:
    using PassFactory = std::function<std::shared_ptr<Pass>()>;

    static void register_pass(const std::string& name, PassFactory factory);
    static std::shared_ptr<Pass> create(const std::string& name);
    static bool has(const std::string& name);
    static std::vector<std::string> list_all();

private:
    static std::unordered_map<std::string, PassFactory>& registry();
};

// 注册宏（在各 Pass 的 .cpp 文件末尾调用）
#define REGISTER_PASS(cls) \
    static bool _##cls##_registered = []() { \
        PassRegistry::register_pass(#cls, []() { \
            return std::make_shared<cls>(); \
        }); \
        return true; \
    }()
```

**使用示例**：

```cpp
// 注册（在各 pass .cpp 末尾）
REGISTER_PASS(ConstantFolding);
REGISTER_PASS(OperatorFusion);

// 动态创建
auto pass = PassRegistry::create("ConstantFolding");
pm.add_pass(pass);
```

---

## 10. 调试与日志

### 10.1 IR Dump

当 `PassContext::dump_ir = true` 时，每个 Pass 执行后将当前 IRModule 以 JSON 格式 dump 到 `dump_dir`，文件命名为 `<序号>_<pass_name>.json`：

```
/tmp/rasp_ir_dump/
├── 00_input.json
├── 01_ConstantFolding.json
├── 02_DeadCodeElimination.json
├── 03_OperatorFusion.json
├── 04_ConstantFolding.json
├── 05_DeadCodeElimination.json
└── 06_LayoutTransformation.json
```

### 10.2 Pass 执行日志

```
[PassManager] Running: ConstantFolding
[PassManager]   Input:  42 nodes
[PassManager]   Output: 38 nodes (folded 4 constants)
[PassManager]   Time:   1.2 ms
[PassManager] Running: OperatorFusion
[PassManager]   Input:  38 nodes
[PassManager]   Output: 31 nodes (fused 7 patterns)
[PassManager]   Time:   3.5 ms
```

---

## 11. 设计示例：完整 Pass 运行流程

```cpp
// 1. 读取前端生成的 IRModule
auto mod = IRModule::from_json(read_file("model_ir.json"));

// 2. 配置 PassContext
PassContext ctx;
ctx.opt_level = 2;
ctx.dump_ir = false;

// 3. 构建并执行 HLIR Pass Pipeline
PassManager hlir_pm = build_hlir_pass_pipeline(ctx.opt_level);
auto optimized_mod = hlir_pm.run(mod, ctx);

// 4. Lowering：HLIR → LLIR
auto llir_mod = Lowering::lower(optimized_mod);

// 5. 构建并执行 LLIR Pass Pipeline
PassManager llir_pm = build_llir_pass_pipeline(ctx.opt_level);
auto optimized_llir = llir_pm.run_llir(llir_mod, ctx);

// 6. Codegen
Codegen::generate(optimized_llir, "output/model_gen.cpp");
```
