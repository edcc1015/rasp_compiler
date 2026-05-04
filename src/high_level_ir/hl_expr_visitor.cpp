#include <stdexcept>

#include "high_level_ir/hl_expr_visitor.h"

namespace rasp {

void ExprVisitor::visit(Ref<Expr> expr) {
  switch (expr->node_type()) {
    case IRNodeType::kVar:
      visit_var(std::static_pointer_cast<Var>(expr));
      break;
    case IRNodeType::kConstant:
      visit_constant(std::static_pointer_cast<Constant>(expr));
      break;
    case IRNodeType::kCall:
      visit_call(std::static_pointer_cast<Call>(expr));
      break;
    case IRNodeType::kLet:
      visit_let(std::static_pointer_cast<Let>(expr));
      break;
    case IRNodeType::kTuple:
      visit_tuple(std::static_pointer_cast<Tuple>(expr));
      break;
    case IRNodeType::kTupleGetItem:
      visit_tuple_get_item(std::static_pointer_cast<TupleGetItem>(expr));
      break;
    case IRNodeType::kFunction:
      visit_function(std::static_pointer_cast<Function>(expr));
      break;
    default:
      throw std::runtime_error("ExprVisitor::visit: unsupported node type");
  }
}

void ExprVisitor::visit_call(Ref<Call> node) {
  for (auto& arg : node->args) {
    visit(arg);
  }
}

void ExprVisitor::visit_let(Ref<Let> node) {
  visit(node->value);
  visit(node->body);
}

void ExprVisitor::visit_tuple(Ref<Tuple> node) {
  for (auto& field : node->fields) {
    visit(field);
  }
}

void ExprVisitor::visit_tuple_get_item(Ref<TupleGetItem> node) {
  visit(node->tuple);
}

void ExprVisitor::visit_function(Ref<Function> node) {
  visit(node->body);
}

} /* namespace rasp */
