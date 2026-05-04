#include <stdexcept>
#include <vector>

#include "high_level_ir/hl_expr_mutator.h"

namespace rasp {

Ref<Expr> ExprMutator::mutate(Ref<Expr> expr) {
  switch (expr->node_type()) {
    case IRNodeType::kVar:
      return mutate_var(std::static_pointer_cast<Var>(expr));
    case IRNodeType::kConstant:
      return mutate_constant(std::static_pointer_cast<Constant>(expr));
    case IRNodeType::kCall:
      return mutate_call(std::static_pointer_cast<Call>(expr));
    case IRNodeType::kLet:
      return mutate_let(std::static_pointer_cast<Let>(expr));
    case IRNodeType::kTuple:
      return mutate_tuple(std::static_pointer_cast<Tuple>(expr));
    case IRNodeType::kTupleGetItem:
      return mutate_tuple_get_item(std::static_pointer_cast<TupleGetItem>(expr));
    case IRNodeType::kFunction:
      return mutate_function(std::static_pointer_cast<Function>(expr));
    default:
      throw std::runtime_error("ExprMutator::mutate: unsupported node type");
  }
}

Ref<Expr> ExprMutator::mutate_call(Ref<Call> node) {
  bool changed = false;
  std::vector<Ref<Expr>> new_args;
  new_args.reserve(node->args.size());
  for (auto& arg : node->args) {
    auto new_arg = mutate(arg);
    if (new_arg != arg) changed = true;
    new_args.push_back(std::move(new_arg));
  }
  if (!changed) return node;
  return Call::make(node->op, std::move(new_args), node->attrs);
}

Ref<Expr> ExprMutator::mutate_let(Ref<Let> node) {
  auto new_value = mutate(node->value);
  auto new_body = mutate(node->body);
  if (new_value == node->value && new_body == node->body) return node;
  return Let::make(node->var, std::move(new_value), std::move(new_body));
}

Ref<Expr> ExprMutator::mutate_tuple(Ref<Tuple> node) {
  bool changed = false;
  std::vector<Ref<Expr>> new_fields;
  new_fields.reserve(node->fields.size());
  for (auto& field : node->fields) {
    auto new_field = mutate(field);
    if (new_field != field) changed = true;
    new_fields.push_back(std::move(new_field));
  }
  if (!changed) return node;
  return Tuple::make(std::move(new_fields));
}

Ref<Expr> ExprMutator::mutate_tuple_get_item(Ref<TupleGetItem> node) {
  auto new_tuple = mutate(node->tuple);
  if (new_tuple == node->tuple) return node;
  return TupleGetItem::make(std::move(new_tuple), node->index);
}

Ref<Expr> ExprMutator::mutate_function(Ref<Function> node) {
  auto new_body = mutate(node->body);
  if (new_body == node->body) return node;
  return Function::make(node->params, std::move(new_body), node->ret_type, node->attrs);
}

} /* namespace rasp */
