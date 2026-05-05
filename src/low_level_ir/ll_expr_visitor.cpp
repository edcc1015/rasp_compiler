#include <stdexcept>

#include "low_level_ir/ll_expr_visitor.h"

namespace rasp {

/* ── PrimExprVisitor dispatch ─────────────────────────────────────────────── */

void PrimExprVisitor::visit(Ref<PrimExpr> expr) {
  if (!expr) return;
  switch (expr->node_type()) {
    case IRNodeType::kLLVar:
      return visit_ll_var(std::static_pointer_cast<LLVar>(expr));
    case IRNodeType::kIntImm:
      return visit_int_imm(std::static_pointer_cast<IntImm>(expr));
    case IRNodeType::kFloatImm:
      return visit_float_imm(std::static_pointer_cast<FloatImm>(expr));
    case IRNodeType::kBufferLoad:
      return visit_buffer_load(std::static_pointer_cast<BufferLoad>(expr));
    case IRNodeType::kAdd:
      return visit_add(std::static_pointer_cast<Add>(expr));
    case IRNodeType::kSub:
      return visit_sub(std::static_pointer_cast<Sub>(expr));
    case IRNodeType::kMul:
      return visit_mul(std::static_pointer_cast<Mul>(expr));
    case IRNodeType::kDiv:
      return visit_div(std::static_pointer_cast<Div>(expr));
    case IRNodeType::kMod:
      return visit_mod(std::static_pointer_cast<Mod>(expr));
    case IRNodeType::kFloorDiv:
      return visit_floor_div(std::static_pointer_cast<FloorDiv>(expr));
    case IRNodeType::kFloorMod:
      return visit_floor_mod(std::static_pointer_cast<FloorMod>(expr));
    case IRNodeType::kMin:
      return visit_min(std::static_pointer_cast<Min>(expr));
    case IRNodeType::kMax:
      return visit_max(std::static_pointer_cast<Max>(expr));
    case IRNodeType::kSelect:
      return visit_select(std::static_pointer_cast<Select>(expr));
    case IRNodeType::kCast:
      return visit_cast(std::static_pointer_cast<Cast>(expr));
    case IRNodeType::kRamp:
      return visit_ramp(std::static_pointer_cast<Ramp>(expr));
    case IRNodeType::kBroadcast:
      return visit_broadcast(std::static_pointer_cast<Broadcast>(expr));
    case IRNodeType::kLLCall:
      return visit_prim_call(std::static_pointer_cast<PrimCall>(expr));
    default:
      throw std::runtime_error("PrimExprVisitor::visit: unsupported node type");
  }
}

/* ── PrimExprVisitor defaults ─────────────────────────────────────────────── */

void PrimExprVisitor::visit_buffer_load(Ref<BufferLoad> node) {
  for (auto& idx : node->indices) visit(idx);
}

void PrimExprVisitor::visit_add(Ref<Add> node) { visit(node->a); visit(node->b); }
void PrimExprVisitor::visit_sub(Ref<Sub> node) { visit(node->a); visit(node->b); }
void PrimExprVisitor::visit_mul(Ref<Mul> node) { visit(node->a); visit(node->b); }
void PrimExprVisitor::visit_div(Ref<Div> node) { visit(node->a); visit(node->b); }
void PrimExprVisitor::visit_mod(Ref<Mod> node) { visit(node->a); visit(node->b); }
void PrimExprVisitor::visit_floor_div(Ref<FloorDiv> node) { visit(node->a); visit(node->b); }
void PrimExprVisitor::visit_floor_mod(Ref<FloorMod> node) { visit(node->a); visit(node->b); }
void PrimExprVisitor::visit_min(Ref<Min> node) { visit(node->a); visit(node->b); }
void PrimExprVisitor::visit_max(Ref<Max> node) { visit(node->a); visit(node->b); }

void PrimExprVisitor::visit_select(Ref<Select> node) {
  visit(node->condition);
  visit(node->true_value);
  visit(node->false_value);
}

void PrimExprVisitor::visit_cast(Ref<Cast> node) { visit(node->value); }

void PrimExprVisitor::visit_ramp(Ref<Ramp> node) {
  visit(node->base);
  visit(node->stride);
}

void PrimExprVisitor::visit_broadcast(Ref<Broadcast> node) { visit(node->value); }

void PrimExprVisitor::visit_prim_call(Ref<PrimCall> node) {
  for (auto& arg : node->args) visit(arg);
}

/* ── StmtVisitor dispatch ─────────────────────────────────────────────────── */

void StmtVisitor::visit_stmt(Ref<Stmt> stmt) {
  if (!stmt) return;
  switch (stmt->node_type()) {
    case IRNodeType::kSeqStmt:
      return visit_seq_stmt(std::static_pointer_cast<SeqStmt>(stmt));
    case IRNodeType::kFor:
      return visit_for(std::static_pointer_cast<For>(stmt));
    case IRNodeType::kIfThenElse:
      return visit_if_then_else(std::static_pointer_cast<IfThenElse>(stmt));
    case IRNodeType::kAllocate:
      return visit_allocate(std::static_pointer_cast<Allocate>(stmt));
    case IRNodeType::kBufferStore:
      return visit_buffer_store(std::static_pointer_cast<BufferStore>(stmt));
    case IRNodeType::kEvaluate:
      return visit_evaluate(std::static_pointer_cast<Evaluate>(stmt));
    default:
      throw std::runtime_error("StmtVisitor::visit_stmt: unsupported node type");
  }
}

/* ── StmtVisitor defaults ─────────────────────────────────────────────────── */

void StmtVisitor::visit_seq_stmt(Ref<SeqStmt> node) {
  for (auto& s : node->seq) visit_stmt(s);
}

void StmtVisitor::visit_for(Ref<For> node) {
  visit(node->min);
  visit(node->extent);
  visit_stmt(node->body);
}

void StmtVisitor::visit_if_then_else(Ref<IfThenElse> node) {
  visit(node->condition);
  visit_stmt(node->then_case);
  if (node->else_case) visit_stmt(node->else_case);
}

void StmtVisitor::visit_allocate(Ref<Allocate> node) {
  for (auto& extent : node->extents) visit(extent);
  visit_stmt(node->body);
}

void StmtVisitor::visit_buffer_store(Ref<BufferStore> node) {
  for (auto& idx : node->indices) visit(idx);
  visit(node->value);
}

void StmtVisitor::visit_evaluate(Ref<Evaluate> node) {
  visit(node->value);
}

} /* namespace rasp */
