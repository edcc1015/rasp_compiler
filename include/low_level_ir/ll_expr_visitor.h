#ifndef RASP_INCLUDE_LOW_LEVEL_IR_LL_EXPR_VISITOR_H_
#define RASP_INCLUDE_LOW_LEVEL_IR_LL_EXPR_VISITOR_H_

#pragma once

#include "low_level_ir/stmt.h"

namespace rasp {

/* ──────────────────────────────────────────────────────────────────────────
 * PrimExprVisitor
 * Read-only recursive traversal of a PrimExpr tree.
 * Override specific visit_xxx methods to inspect nodes of interest.
 * visit() dispatches to the appropriate virtual method based on node type.
 *
 * Default leaf implementations are no-ops.
 * Default composite implementations recurse into all child expressions.
 * ────────────────────────────────────────────────────────────────────────── */
class PrimExprVisitor {
 public:
  virtual ~PrimExprVisitor() = default;

  /* Entry point: dispatch to the appropriate visit_xxx. */
  virtual void visit(Ref<PrimExpr> expr);

  /* Leaf nodes — default: no-op. */
  virtual void visit_ll_var(Ref<LLVar> node)       { (void)node; }
  virtual void visit_int_imm(Ref<IntImm> node)     { (void)node; }
  virtual void visit_float_imm(Ref<FloatImm> node) { (void)node; }

  /* Composite nodes — default: recurse into children. */
  virtual void visit_buffer_load(Ref<BufferLoad> node);
  virtual void visit_add(Ref<Add> node);
  virtual void visit_sub(Ref<Sub> node);
  virtual void visit_mul(Ref<Mul> node);
  virtual void visit_div(Ref<Div> node);
  virtual void visit_mod(Ref<Mod> node);
  virtual void visit_floor_div(Ref<FloorDiv> node);
  virtual void visit_floor_mod(Ref<FloorMod> node);
  virtual void visit_min(Ref<Min> node);
  virtual void visit_max(Ref<Max> node);
  virtual void visit_select(Ref<Select> node);
  virtual void visit_cast(Ref<Cast> node);
  virtual void visit_ramp(Ref<Ramp> node);
  virtual void visit_broadcast(Ref<Broadcast> node);
  virtual void visit_prim_call(Ref<PrimCall> node);

 protected:
  PrimExprVisitor() = default;
};

/* ──────────────────────────────────────────────────────────────────────────
 * StmtVisitor
 * Read-only recursive traversal of a Stmt tree.
 * Override specific visit_xxx methods to inspect nodes of interest.
 * visit_stmt() dispatches to the appropriate virtual method.
 *
 * Default implementations recurse into child statements.
 * Leaf statement nodes (BufferStore, Evaluate) default to no-op.
 * ────────────────────────────────────────────────────────────────────────── */
class StmtVisitor : public PrimExprVisitor {
 public:
  virtual ~StmtVisitor() = default;

  /* Entry point: dispatch to the appropriate visit_xxx. */
  virtual void visit_stmt(Ref<Stmt> stmt);

  virtual void visit_seq_stmt(Ref<SeqStmt> node);
  virtual void visit_for(Ref<For> node);
  virtual void visit_if_then_else(Ref<IfThenElse> node);
  virtual void visit_allocate(Ref<Allocate> node);

  virtual void visit_buffer_store(Ref<BufferStore> node);
  virtual void visit_evaluate(Ref<Evaluate> node);

 protected:
  StmtVisitor() = default;
};

} /* namespace rasp */

#endif /* RASP_INCLUDE_LOW_LEVEL_IR_LL_EXPR_VISITOR_H_ */
