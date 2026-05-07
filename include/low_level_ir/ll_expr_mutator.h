#ifndef RASP_INCLUDE_LOW_LEVEL_IR_LL_EXPR_MUTATOR_H_
#define RASP_INCLUDE_LOW_LEVEL_IR_LL_EXPR_MUTATOR_H_

#pragma once

#include <vector>
#include <stdexcept>

#include "low_level_ir/stmt.h"

namespace rasp {

/* ──────────────────────────────────────────────────────────────────────────
 * PrimExprMutator
 * Tree-rewriting traversal of a PrimExpr tree.
 * Override mutate_xxx methods to replace specific node kinds.
 *
 * Default leaf implementations return the node unchanged (identity).
 * Default composite implementations rebuild the node only when at least one
 * child pointer has changed (structural sharing).
 * mutate() is the entry point and dispatches based on node type.
 * ────────────────────────────────────────────────────────────────────────── */
class PrimExprMutator {
 public:
  virtual ~PrimExprMutator() = default;

  /* Entry point: dispatch to the appropriate mutate_xxx. */
  virtual Ref<PrimExpr> mutate(Ref<PrimExpr> expr);

  /* Leaf nodes — default: return unchanged. */
  virtual Ref<PrimExpr> mutate_ll_var(Ref<LLVar> node)       { return node; }
  virtual Ref<PrimExpr> mutate_int_imm(Ref<IntImm> node)     { return node; }
  virtual Ref<PrimExpr> mutate_float_imm(Ref<FloatImm> node) { return node; }

  virtual Ref<PrimExpr> mutate_buffer_load(Ref<BufferLoad> node);

  /* Helper template: rebuild a BinaryOp subclass only when a child changed.
   * Must live in the header so the compiler can instantiate it for each T. */
  template <typename T>
  Ref<PrimExpr> mutate_binary(Ref<T> node) {
    auto na = mutate(node->a);
    auto nb = mutate(node->b);
    if (na == node->a && nb == node->b) return node;
    return T::make(std::move(na), std::move(nb));
  }

  virtual Ref<PrimExpr> mutate_add(Ref<Add> node);
  virtual Ref<PrimExpr> mutate_sub(Ref<Sub> node);
  virtual Ref<PrimExpr> mutate_mul(Ref<Mul> node);
  virtual Ref<PrimExpr> mutate_div(Ref<Div> node);
  virtual Ref<PrimExpr> mutate_mod(Ref<Mod> node);
  virtual Ref<PrimExpr> mutate_floor_div(Ref<FloorDiv> node);
  virtual Ref<PrimExpr> mutate_floor_mod(Ref<FloorMod> node);
  virtual Ref<PrimExpr> mutate_min(Ref<Min> node);
  virtual Ref<PrimExpr> mutate_max(Ref<Max> node);
  virtual Ref<PrimExpr> mutate_select(Ref<Select> node);
  virtual Ref<PrimExpr> mutate_cast(Ref<Cast> node);
  virtual Ref<PrimExpr> mutate_ramp(Ref<Ramp> node);
  virtual Ref<PrimExpr> mutate_broadcast(Ref<Broadcast> node);
  virtual Ref<PrimExpr> mutate_prim_call(Ref<PrimCall> node);

 protected:
  PrimExprMutator() = default;
};

/* ──────────────────────────────────────────────────────────────────────────
 * StmtMutator
 * Tree-rewriting traversal of a Stmt tree.
 * Override mutate_xxx methods to substitute specific statement kinds.
 *
 * Default implementations rebuild a node only when a child has changed.
 * Leaf statement nodes (BufferStore, Evaluate) default to identity.
 * mutate_stmt() is the entry point and dispatches based on node type.
 * ────────────────────────────────────────────────────────────────────────── */
class StmtMutator : public PrimExprMutator {
 public:
  virtual ~StmtMutator() = default;

  /* Entry point: dispatch to the appropriate mutate_xxx. */
  virtual Ref<Stmt> mutate_stmt(Ref<Stmt> stmt);

  virtual Ref<Stmt> mutate_seq_stmt(Ref<SeqStmt> node);
  virtual Ref<Stmt> mutate_for(Ref<For> node);
  virtual Ref<Stmt> mutate_if_then_else(Ref<IfThenElse> node);
  virtual Ref<Stmt> mutate_allocate(Ref<Allocate> node);

  virtual Ref<Stmt> mutate_buffer_store(Ref<BufferStore> node);
  virtual Ref<Stmt> mutate_evaluate(Ref<Evaluate> node);

 protected:
  StmtMutator() = default;
};

} /* namespace rasp */

#endif /* RASP_INCLUDE_LOW_LEVEL_IR_LL_EXPR_MUTATOR_H_ */
