#pragma once

#include "expr.h"

namespace rasp {

/* ──────────────────────────────────────────────────────────────────────────
 * ExprVisitor
 * Read-only traversal of an Expr tree.
 * Override visit_xxx methods for nodes of interest; all others are no-ops.
 * visit() dispatches to the appropriate virtual method based on node type.
 *
 * Default composite-node implementations recurse into all children so that
 * subclasses only need to override the nodes they care about.
 * ────────────────────────────────────────────────────────────────────────── */
class ExprVisitor {
 public:
  virtual ~ExprVisitor() = default;

  /* Entry point: dispatch to the appropriate visit_xxx. */
  virtual void visit(Ref<Expr> expr);

  /* Leaf nodes: default is a no-op. */
  virtual void visit_var(Ref<Var> node) { (void)node; }
  virtual void visit_constant(Ref<Constant> node) { (void)node; }

  /* Composite nodes: default recursively visits children. */
  virtual void visit_call(Ref<Call> node);
  virtual void visit_let(Ref<Let> node);
  virtual void visit_tuple(Ref<Tuple> node);
  virtual void visit_tuple_get_item(Ref<TupleGetItem> node);
  virtual void visit_function(Ref<Function> node);

 protected:
  ExprVisitor() = default;
};

} /* namespace rasp */
