#ifndef RASP_INCLUDE_HIGH_LEVEL_IR_HL_EXPR_MUTATOR_H_
#define RASP_INCLUDE_HIGH_LEVEL_IR_HL_EXPR_MUTATOR_H_

#pragma once

#include "expr.h"

namespace rasp {

/* ──────────────────────────────────────────────────────────────────────────
 * ExprMutator
 * Tree-rewriting traversal of an Expr tree.
 * Override mutate_xxx methods to replace specific node kinds.
 *
 * Default composite-node implementations recurse into children and return
 * the original node unchanged when no child pointer was replaced (structural
 * sharing), or a freshly constructed copy when at least one child changed.
 * mutate() is the entry point and dispatches to the appropriate virtual method.
 * ────────────────────────────────────────────────────────────────────────── */
class ExprMutator {
 public:
  virtual ~ExprMutator() = default;

  /* Entry point: dispatch to the appropriate mutate_xxx. */
  virtual Ref<Expr> mutate(Ref<Expr> expr);

  /* Leaf nodes: default returns the node unchanged (identity). */
  virtual Ref<Expr> mutate_var(Ref<Var> node) { return node; }
  virtual Ref<Expr> mutate_constant(Ref<Constant> node) { return node; }

  /* Composite nodes: default recurses into children and reconstructs only
   * when at least one child pointer changes. */
  virtual Ref<Expr> mutate_call(Ref<Call> node);
  virtual Ref<Expr> mutate_let(Ref<Let> node);
  virtual Ref<Expr> mutate_tuple(Ref<Tuple> node);
  virtual Ref<Expr> mutate_tuple_get_item(Ref<TupleGetItem> node);
  virtual Ref<Expr> mutate_function(Ref<Function> node);

 protected:
  ExprMutator() = default;
};

} /* namespace rasp */

#endif /* RASP_INCLUDE_HIGH_LEVEL_IR_HL_EXPR_MUTATOR_H_ */
