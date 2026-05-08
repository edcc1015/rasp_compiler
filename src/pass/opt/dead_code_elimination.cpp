#include "pass/opt/dead_code_elimination.h"

#include "pass/pass_manager.h"
#include "utils/log.h"

namespace rasp {

/* ── RefCounter::visit_var ──────────────────────────────────────────────── */

void DeadCodeElimination::RefCounter::visit_var(Ref<Var> node) {
  ref_count[node.get()]++;
}

/* ── DecrementVisitor::visit_var ────────────────────────────────────────── */

void DeadCodeElimination::DecrementVisitor::visit_var(Ref<Var> node) {
  auto it = ref_count_.find(node.get());
  if (it != ref_count_.end()) --it->second;
}

/* ── DCEMutator::mutate_let ─────────────────────────────────────────────── */

Ref<Expr> DeadCodeElimination::DCEMutator::mutate_let(Ref<Let> node) {
  /* Process body first (inside-out) so that when we check a var's count
   * below, any inner dead bindings have already been processed and their
   * DecrementVisitors have updated the counts. */
  auto new_body = mutate(node->body);

  auto it = ref_count_.find(node->var.get());
  int count = (it != ref_count_.end()) ? it->second : 0;

  if (count == 0) {
    /* Dead binding: walk the value and decrement every Var's ref_count so
     * outer Let nodes that referenced the same Vars may become dead too. */
    DecrementVisitor dec(ref_count_);
    dec.visit(node->value);
    return new_body;
  }

  auto new_value = mutate(node->value);
  if (new_value == node->value && new_body == node->body) return node;
  return Let::make(node->var, std::move(new_value), std::move(new_body));
}

/* ── DeadCodeElimination::transform_function ────────────────────────────── */

Ref<Function> DeadCodeElimination::transform_function(
    Ref<Function> func, Ref<IRModule> /*mod*/, const PassContext& /*ctx*/) {
  /* Phase 1: count variable references in the function body. */
  RefCounter counter;
  counter.visit(func->body);

  /* Phase 2: eliminate dead Let bindings. */
  DCEMutator dce(counter.ref_count);
  auto new_body = dce.mutate(func->body);

  if (new_body == func->body) return func;
  return Function::make(func->params, std::move(new_body),
                        func->ret_type, func->attrs);
}

REGISTER_PASS(DeadCodeElimination);

} /* namespace rasp */
