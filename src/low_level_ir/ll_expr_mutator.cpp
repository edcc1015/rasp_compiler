#include <stdexcept>
#include <vector>

#include "low_level_ir/ll_expr_mutator.h"

namespace rasp {

/* ── PrimExprMutator dispatch ─────────────────────────────────────────────── */

Ref<PrimExpr> PrimExprMutator::mutate(Ref<PrimExpr> expr) {
  if (!expr) return nullptr;
  switch (expr->node_type()) {
    case IRNodeType::kLLVar:
      return mutate_ll_var(std::static_pointer_cast<LLVar>(expr));
    case IRNodeType::kIntImm:
      return mutate_int_imm(std::static_pointer_cast<IntImm>(expr));
    case IRNodeType::kFloatImm:
      return mutate_float_imm(std::static_pointer_cast<FloatImm>(expr));
    case IRNodeType::kBufferLoad:
      return mutate_buffer_load(std::static_pointer_cast<BufferLoad>(expr));
    case IRNodeType::kAdd:
      return mutate_add(std::static_pointer_cast<Add>(expr));
    case IRNodeType::kSub:
      return mutate_sub(std::static_pointer_cast<Sub>(expr));
    case IRNodeType::kMul:
      return mutate_mul(std::static_pointer_cast<Mul>(expr));
    case IRNodeType::kDiv:
      return mutate_div(std::static_pointer_cast<Div>(expr));
    case IRNodeType::kMod:
      return mutate_mod(std::static_pointer_cast<Mod>(expr));
    case IRNodeType::kFloorDiv:
      return mutate_floor_div(std::static_pointer_cast<FloorDiv>(expr));
    case IRNodeType::kFloorMod:
      return mutate_floor_mod(std::static_pointer_cast<FloorMod>(expr));
    case IRNodeType::kMin:
      return mutate_min(std::static_pointer_cast<Min>(expr));
    case IRNodeType::kMax:
      return mutate_max(std::static_pointer_cast<Max>(expr));
    case IRNodeType::kSelect:
      return mutate_select(std::static_pointer_cast<Select>(expr));
    case IRNodeType::kCast:
      return mutate_cast(std::static_pointer_cast<Cast>(expr));
    case IRNodeType::kRamp:
      return mutate_ramp(std::static_pointer_cast<Ramp>(expr));
    case IRNodeType::kBroadcast:
      return mutate_broadcast(std::static_pointer_cast<Broadcast>(expr));
    case IRNodeType::kLLCall:
      return mutate_prim_call(std::static_pointer_cast<PrimCall>(expr));
    default:
      throw std::runtime_error("PrimExprMutator::mutate: unsupported node type");
  }
}

/* ── PrimExprMutator defaults ─────────────────────────────────────────────── */

Ref<PrimExpr> PrimExprMutator::mutate_buffer_load(Ref<BufferLoad> node) {
  std::vector<Ref<PrimExpr>> new_idx;
  new_idx.reserve(node->indices.size());
  bool changed = false;
  for (auto& idx : node->indices) {
    auto ni = mutate(idx);
    if (ni != idx) changed = true;
    new_idx.push_back(std::move(ni));
  }
  if (!changed) return node;
  return BufferLoad::make(node->buffer, std::move(new_idx));
}

Ref<PrimExpr> PrimExprMutator::mutate_add(Ref<Add> node)            { return mutate_binary(node); }
Ref<PrimExpr> PrimExprMutator::mutate_sub(Ref<Sub> node)            { return mutate_binary(node); }
Ref<PrimExpr> PrimExprMutator::mutate_mul(Ref<Mul> node)            { return mutate_binary(node); }
Ref<PrimExpr> PrimExprMutator::mutate_div(Ref<Div> node)            { return mutate_binary(node); }
Ref<PrimExpr> PrimExprMutator::mutate_mod(Ref<Mod> node)            { return mutate_binary(node); }
Ref<PrimExpr> PrimExprMutator::mutate_floor_div(Ref<FloorDiv> node) { return mutate_binary(node); }
Ref<PrimExpr> PrimExprMutator::mutate_floor_mod(Ref<FloorMod> node) { return mutate_binary(node); }
Ref<PrimExpr> PrimExprMutator::mutate_min(Ref<Min> node)            { return mutate_binary(node); }
Ref<PrimExpr> PrimExprMutator::mutate_max(Ref<Max> node)            { return mutate_binary(node); }

Ref<PrimExpr> PrimExprMutator::mutate_select(Ref<Select> node) {
  auto nc = mutate(node->condition);
  auto nt = mutate(node->true_value);
  auto nf = mutate(node->false_value);
  if (nc == node->condition &&
      nt == node->true_value &&
      nf == node->false_value) return node;
  return Select::make(std::move(nc), std::move(nt), std::move(nf));
}

Ref<PrimExpr> PrimExprMutator::mutate_cast(Ref<Cast> node) {
  auto nv = mutate(node->value);
  if (nv == node->value) return node;
  return Cast::make(node->dtype, std::move(nv));
}

Ref<PrimExpr> PrimExprMutator::mutate_ramp(Ref<Ramp> node) {
  auto nb = mutate(node->base);
  auto ns = mutate(node->stride);
  if (nb == node->base && ns == node->stride) return node;
  return Ramp::make(std::move(nb), std::move(ns), node->lanes);
}

Ref<PrimExpr> PrimExprMutator::mutate_broadcast(Ref<Broadcast> node) {
  auto nv = mutate(node->value);
  if (nv == node->value) return node;
  return Broadcast::make(std::move(nv), node->lanes);
}

Ref<PrimExpr> PrimExprMutator::mutate_prim_call(Ref<PrimCall> node) {
  std::vector<Ref<PrimExpr>> new_args;
  new_args.reserve(node->args.size());
  bool changed = false;
  for (auto& arg : node->args) {
    auto na = mutate(arg);
    if (na != arg) changed = true;
    new_args.push_back(std::move(na));
  }
  if (!changed) return node;
  return PrimCall::make(node->dtype, node->func_name,
                        node->call_type, std::move(new_args));
}

/* ── StmtMutator dispatch ─────────────────────────────────────────────────── */

Ref<Stmt> StmtMutator::mutate_stmt(Ref<Stmt> stmt) {
  if (!stmt) return nullptr;
  switch (stmt->node_type()) {
    case IRNodeType::kSeqStmt:
      return mutate_seq_stmt(std::static_pointer_cast<SeqStmt>(stmt));
    case IRNodeType::kFor:
      return mutate_for(std::static_pointer_cast<For>(stmt));
    case IRNodeType::kIfThenElse:
      return mutate_if_then_else(std::static_pointer_cast<IfThenElse>(stmt));
    case IRNodeType::kAllocate:
      return mutate_allocate(std::static_pointer_cast<Allocate>(stmt));
    case IRNodeType::kBufferStore:
      return mutate_buffer_store(std::static_pointer_cast<BufferStore>(stmt));
    case IRNodeType::kEvaluate:
      return mutate_evaluate(std::static_pointer_cast<Evaluate>(stmt));
    default:
      throw std::runtime_error("StmtMutator::mutate_stmt: unsupported node type");
  }
}

/* ── StmtMutator defaults ─────────────────────────────────────────────────── */

Ref<Stmt> StmtMutator::mutate_seq_stmt(Ref<SeqStmt> node) {
  std::vector<Ref<Stmt>> new_seq;
  new_seq.reserve(node->seq.size());
  bool changed = false;
  for (auto& s : node->seq) {
    auto ns = mutate_stmt(s);
    if (ns != s) changed = true;
    new_seq.push_back(std::move(ns));
  }
  if (!changed) return node;
  return SeqStmt::make(std::move(new_seq));
}

Ref<Stmt> StmtMutator::mutate_for(Ref<For> node) {
  auto new_min = mutate(node->min);
  auto new_extent = mutate(node->extent);
  auto new_body = mutate_stmt(node->body);
  if (new_min == node->min &&
      new_extent == node->extent &&
      new_body == node->body) return node;
  return For::make(node->loop_var, std::move(new_min),
                   std::move(new_extent), node->kind,
                   std::move(new_body));
}

Ref<Stmt> StmtMutator::mutate_if_then_else(Ref<IfThenElse> node) {
  auto new_condition = mutate(node->condition);
  auto new_then = mutate_stmt(node->then_case);
  auto new_else = node->else_case ? mutate_stmt(node->else_case) : nullptr;
  if (new_condition == node->condition &&
      new_then == node->then_case &&
      new_else == node->else_case) return node;
  return IfThenElse::make(std::move(new_condition),
                          std::move(new_then),
                          std::move(new_else));
}

Ref<Stmt> StmtMutator::mutate_allocate(Ref<Allocate> node) {
  std::vector<Ref<PrimExpr>> new_extents;
  new_extents.reserve(node->extents.size());
  bool changed = false;
  for (auto& extent : node->extents) {
    auto new_extent = mutate(extent);
    if (new_extent != extent) changed = true;
    new_extents.push_back(std::move(new_extent));
  }
  auto new_body = mutate_stmt(node->body);
  if (!changed && new_body == node->body) return node;
  return Allocate::make(node->buffer_var, node->dtype,
                        std::move(new_extents), std::move(new_body));
}

Ref<Stmt> StmtMutator::mutate_buffer_store(Ref<BufferStore> node) {
  std::vector<Ref<PrimExpr>> new_indices;
  new_indices.reserve(node->indices.size());
  bool changed = false;
  for (auto& idx : node->indices) {
    auto new_idx = mutate(idx);
    if (new_idx != idx) changed = true;
    new_indices.push_back(std::move(new_idx));
  }
  auto new_value = mutate(node->value);
  if (!changed && new_value == node->value) return node;
  return BufferStore::make(node->buffer, std::move(new_indices), std::move(new_value));
}

Ref<Stmt> StmtMutator::mutate_evaluate(Ref<Evaluate> node) {
  auto new_value = mutate(node->value);
  if (new_value == node->value) return node;
  return Evaluate::make(std::move(new_value));
}

} /* namespace rasp */
