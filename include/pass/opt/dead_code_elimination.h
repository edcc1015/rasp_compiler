#ifndef RASP_INCLUDE_PASS_OPT_DEAD_CODE_ELIMINATION_H_
#define RASP_INCLUDE_PASS_OPT_DEAD_CODE_ELIMINATION_H_

#pragma once

#include <unordered_map>

#include "high_level_ir/hl_ir.h"
#include "pass/pass.h"

namespace rasp {

/* ──────────────────────────────────────────────────────────────────────────
 * DeadCodeElimination  (FunctionPass, min_opt_level = 1)
 *
 * Eliminates Let-bindings whose bound Var is never referenced in the body,
 * reducing graph complexity and run-time memory consumption.
 *
 * Two-phase algorithm per function:
 *   1. RefCounter (ExprVisitor): counts how many times each Var node is
 *      referenced inside func->body.
 *   2. DCEMutator (ExprMutator): walks Let nodes inside-out; if the bound
 *      Var has ref_count == 0 the binding is eliminated.  When a binding is
 *      dropped, a DecrementVisitor decrements the ref_counts of every Var
 *      that appeared in the discarded value, so cascaded dead bindings are
 *      resolved in a single pass.
 *
 * All RASP CNN operators are pure (no side-effects), so every unreferenced
 * Let binding is safe to remove.
 * ────────────────────────────────────────────────────────────────────────── */
class DeadCodeElimination : public FunctionPass {
 public:
  std::string name() const override { return "DeadCodeElimination"; }
  int min_opt_level() const override { return 1; }

  Ref<Function> transform_function(Ref<Function> func,
                                   Ref<IRModule> mod,
                                   const PassContext& ctx) override;

 private:
  /* Phase 1: count how many times each Var is referenced as an expression. */
  class RefCounter : public ExprVisitor {
   public:
    std::unordered_map<IRNode*, int> ref_count;
    void visit_var(Ref<Var> node) override;
  };

  /* Helper: decrement ref_count for every Var that appears inside expr.
   * Used when a dead Let value is discarded to keep ref_counts consistent,
   * enabling cascaded elimination in a single pass. */
  class DecrementVisitor : public ExprVisitor {
   public:
    explicit DecrementVisitor(std::unordered_map<IRNode*, int>& rc)
        : ref_count_(rc) {}
    void visit_var(Ref<Var> node) override;

   private:
    std::unordered_map<IRNode*, int>& ref_count_;
  };

  /* Phase 2: remove Let bindings whose Var has ref_count == 0. */
  class DCEMutator : public ExprMutator {
   public:
    explicit DCEMutator(std::unordered_map<IRNode*, int>& rc)
        : ref_count_(rc) {}
    Ref<Expr> mutate_let(Ref<Let> node) override;

   private:
    std::unordered_map<IRNode*, int>& ref_count_;
  };
};

} /* namespace rasp */

#endif /* RASP_INCLUDE_PASS_OPT_DEAD_CODE_ELIMINATION_H_ */
