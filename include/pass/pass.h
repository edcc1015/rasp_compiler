#ifndef RASP_INCLUDE_PASS_PASS_H_
#define RASP_INCLUDE_PASS_PASS_H_

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "pass_context.h"
#include "high_level_ir/ir_module.h"
#include "low_level_ir/ll_ir_module.h"

namespace rasp {

/* ──────────────────────────────────────────────────────────────────────────
 * Pass
 * Abstract base class for all IR transformation passes.
 * Subclasses override transform_hlir or transform_llir (or both).
 * ────────────────────────────────────────────────────────────────────────── */
class Pass {
 public:
  virtual ~Pass() = default;

  /* Unique name used for registration, logging, and dependency checks. */
  virtual std::string name() const = 0;

  /* Minimum opt_level required to run this pass (default: always runs). */
  virtual int min_opt_level() const { return 0; }

  /* Passes that must appear before this one in the pipeline. */
  virtual std::vector<std::string> dependencies() const { return {}; }

  /* Transform a HLIR module; default is identity. */
  virtual Ref<IRModule> transform_hlir(Ref<IRModule> mod,
                                       const PassContext& ctx) {
    (void)ctx;
    return mod;
  }

  /* Transform a LLIR module; default is identity. */
  virtual Ref<LLIRModule> transform_llir(Ref<LLIRModule> mod,
                                         const PassContext& ctx) {
    (void)ctx;
    return mod;
  }

 protected:
  Pass() = default;
};

/* ──────────────────────────────────────────────────────────────────────────
 * FunctionPass
 * Applies a per-Function transformation across every function in an IRModule.
 * Subclasses implement transform_function; the iteration is handled here.
 * ────────────────────────────────────────────────────────────────────────── */
class FunctionPass : public Pass {
 public:
  virtual Ref<Function> transform_function(Ref<Function> func,
                                           Ref<IRModule> mod,
                                           const PassContext& ctx) = 0;

  Ref<IRModule> transform_hlir(Ref<IRModule> mod,
                               const PassContext& ctx) override final {
    auto new_mod = IRModule::make();
    for (auto& [name, func] : mod->functions) {
      auto new_func = transform_function(func, mod, ctx);
      new_mod->add_function(name, new_func);
    }
    return new_mod;
  }
};

/* ──────────────────────────────────────────────────────────────────────────
 * ModulePass
 * Applies a whole-module transformation.
 * Use when analysis or mutations span multiple functions.
 * ────────────────────────────────────────────────────────────────────────── */
class ModulePass : public Pass {
 public:
  virtual Ref<IRModule> transform_module(Ref<IRModule> mod,
                                         const PassContext& ctx) = 0;

  Ref<IRModule> transform_hlir(Ref<IRModule> mod,
                               const PassContext& ctx) override final {
    return transform_module(mod, ctx);
  }
};

} /* namespace rasp */

#endif /* RASP_INCLUDE_PASS_PASS_H_ */
