#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "low_level_ir/buffer.h"

namespace rasp {

/* ──────────────────────────────────────────────────────────────────────────
 * Stmt
 * Abstract base class for all LLIR statement nodes.
 * Stmt nodes form a tree that represents the imperative body of a PrimFunc.
 * ────────────────────────────────────────────────────────────────────────── */
class Stmt : public IRNode {
 protected:
  Stmt() = default;
};

/* ──────────────────────────────────────────────────────────────────────────
 * SeqStmt — an ordered sequence of statements executed top-to-bottom.
 * flatten() merges adjacent SeqStmt nodes to avoid unnecessary nesting.
 * ────────────────────────────────────────────────────────────────────────── */
class SeqStmt : public Stmt {
 public:
  std::vector<Ref<Stmt>> seq;

  IRNodeType node_type() const override { return IRNodeType::kSeqStmt; }

  static Ref<SeqStmt> make(std::vector<Ref<Stmt>> seq);

  /* Flatten nested SeqStmt nodes; returns a single Ref<Stmt>.
   * Implemented in stmt.cpp. */
  static Ref<Stmt> flatten(std::vector<Ref<Stmt>> stmts);

 private:
  explicit SeqStmt(std::vector<Ref<Stmt>> seq)
    : seq(std::move(seq)) {}
};

inline Ref<SeqStmt> SeqStmt::make(std::vector<Ref<Stmt>> seq) {
  return std::shared_ptr<SeqStmt>(new SeqStmt(std::move(seq)));
}

/* ──────────────────────────────────────────────────────────────────────────
 * ForKind — execution strategy for a For loop.
 * ────────────────────────────────────────────────────────────────────────── */
enum class ForKind {
  kSerial,    /* ordinary sequential loop                              */
  kVectorized,/* body executed as a single NEON vector instruction     */
  kUnrolled,  /* loop fully unrolled at code-generation time           */
};

/* ──────────────────────────────────────────────────────────────────────────
 * For — a counted loop with an explicit loop variable.
 *   for (loop_var = min; loop_var < min + extent; loop_var++)
 * ────────────────────────────────────────────────────────────────────────── */
class For : public Stmt {
 public:
  Ref<LLVar> loop_var; /* integer loop counter */
  Ref<PrimExpr> min;      /* first value (usually IntImm(0)) */
  Ref<PrimExpr> extent;   /* number of iterations */
  ForKind kind;
  Ref<Stmt> body;

  IRNodeType node_type() const override { return IRNodeType::kFor; }

  static Ref<For> make(Ref<LLVar> loop_var,
                       Ref<PrimExpr> min,
                       Ref<PrimExpr> extent,
                       ForKind kind,
                       Ref<Stmt> body);

 private:
  For(Ref<LLVar> loop_var,
      Ref<PrimExpr> min,
      Ref<PrimExpr> extent,
      ForKind kind,
      Ref<Stmt> body)
    : loop_var(std::move(loop_var)),
      min(std::move(min)),
      extent(std::move(extent)),
      kind(kind),
      body(std::move(body)) {}
};

inline Ref<For> For::make(Ref<LLVar> loop_var,
                          Ref<PrimExpr> min,
                          Ref<PrimExpr> extent,
                          ForKind kind,
                          Ref<Stmt> body) {
  return std::shared_ptr<For>(
    new For(std::move(loop_var), std::move(min), std::move(extent),
            kind, std::move(body)));
}

/* ──────────────────────────────────────────────────────────────────────────
 * IfThenElse — conditional branch; else_case may be nullptr.
 * ────────────────────────────────────────────────────────────────────────── */
class IfThenElse : public Stmt {
 public:
  Ref<PrimExpr> condition;
  Ref<Stmt> then_case;
  Ref<Stmt> else_case; /* nullptr if there is no else branch */

  IRNodeType node_type() const override { return IRNodeType::kIfThenElse; }

  static Ref<IfThenElse> make(Ref<PrimExpr> condition,
                              Ref<Stmt> then_case,
                              Ref<Stmt> else_case = nullptr);

 private:
  IfThenElse(Ref<PrimExpr> condition,
             Ref<Stmt> then_case,
             Ref<Stmt> else_case)
    : condition(std::move(condition)),
      then_case(std::move(then_case)),
      else_case(std::move(else_case)) {}
};

inline Ref<IfThenElse> IfThenElse::make(Ref<PrimExpr> condition,
                                        Ref<Stmt> then_case,
                                        Ref<Stmt> else_case) {
  return std::shared_ptr<IfThenElse>(
    new IfThenElse(std::move(condition),
                   std::move(then_case),
                   std::move(else_case)));
}

/* ──────────────────────────────────────────────────────────────────────────
 * Allocate — declares a local buffer within a scoped body.
 * buffer_var is the pointer variable used to access the allocation.
 * The allocation is live only within body.
 * ────────────────────────────────────────────────────────────────────────── */
class Allocate : public Stmt {
 public:
  Ref<LLVar> buffer_var; /* data-pointer variable */
  DataType dtype;
  std::vector<Ref<PrimExpr>> extents; /* size of each dimension */
  Ref<Stmt> body; /* allocation scope */

  IRNodeType node_type() const override { return IRNodeType::kAllocate; }

  static Ref<Allocate> make(Ref<LLVar> buffer_var,
                            DataType dtype,
                            std::vector<Ref<PrimExpr>> extents,
                            Ref<Stmt> body);

 private:
  Allocate(Ref<LLVar> buffer_var,
           DataType dtype,
           std::vector<Ref<PrimExpr>> extents,
           Ref<Stmt> body)
    : buffer_var(std::move(buffer_var)),
      dtype(dtype),
      extents(std::move(extents)),
      body(std::move(body)) {}
};

inline Ref<Allocate> Allocate::make(Ref<LLVar> buffer_var,
                                    DataType dtype,
                                    std::vector<Ref<PrimExpr>> extents,
                                    Ref<Stmt> body) {
  return std::shared_ptr<Allocate>(
    new Allocate(std::move(buffer_var), dtype,
                 std::move(extents), std::move(body)));
}

/* ──────────────────────────────────────────────────────────────────────────
 * BufferStore — write a scalar (or vector) value into a Buffer.
 * value may be a scalar PrimExpr or a vector produced by a Ramp/Broadcast.
 * ────────────────────────────────────────────────────────────────────────── */
class BufferStore : public Stmt {
 public:
  Ref<Buffer> buffer;
  std::vector<Ref<PrimExpr>> indices;
  Ref<PrimExpr> value;

  IRNodeType node_type() const override { return IRNodeType::kBufferStore; }

  static Ref<BufferStore> make(Ref<Buffer> buffer,
                               std::vector<Ref<PrimExpr>> indices,
                               Ref<PrimExpr> value);

 private:
  BufferStore(Ref<Buffer> buffer,
              std::vector<Ref<PrimExpr>> indices,
              Ref<PrimExpr> value)
    : buffer(std::move(buffer)),
      indices(std::move(indices)),
      value(std::move(value)) {}
};

inline Ref<BufferStore> BufferStore::make(Ref<Buffer> buffer,
                                          std::vector<Ref<PrimExpr>> indices,
                                          Ref<PrimExpr> value) {
  return std::shared_ptr<BufferStore>(
    new BufferStore(std::move(buffer), std::move(indices), std::move(value)));
}

/* ──────────────────────────────────────────────────────────────────────────
 * Evaluate — execute a PrimExpr for its side effects (e.g. void intrinsics).
 * ────────────────────────────────────────────────────────────────────────── */
class Evaluate : public Stmt {
 public:
  Ref<PrimExpr> value;

  IRNodeType node_type() const override { return IRNodeType::kEvaluate; }

  static Ref<Evaluate> make(Ref<PrimExpr> value);

 private:
  explicit Evaluate(Ref<PrimExpr> value)
    : value(std::move(value)) {}
};

inline Ref<Evaluate> Evaluate::make(Ref<PrimExpr> value) {
  return std::shared_ptr<Evaluate>(new Evaluate(std::move(value)));
}

/* ══════════════════════════════════════════════════════════════════════════
 * Visitor / Mutator patterns for Stmt
 * ══════════════════════════════════════════════════════════════════════════ */

/* ──────────────────────────────────────────────────────────────────────────
 * StmtVisitor — read-only recursive traversal of a Stmt tree.
 * Override specific visit_* methods to inspect nodes of interest.
 * Default implementations recurse into child statements.
 * ────────────────────────────────────────────────────────────────────────── */
class StmtVisitor {
 public:
  virtual ~StmtVisitor() = default;

  /* Entry point: dispatch to the appropriate visit_* method. */
  virtual void visit_stmt(Ref<Stmt> stmt);

  virtual void visit_seq_stmt(Ref<SeqStmt> node) {
    for (auto& s : node->seq) visit_stmt(s);
  }

  virtual void visit_for(Ref<For> node) {
    visit_stmt(node->body);
  }

  virtual void visit_if_then_else(Ref<IfThenElse> node) {
    visit_stmt(node->then_case);
    if (node->else_case) visit_stmt(node->else_case);
  }

  virtual void visit_allocate(Ref<Allocate> node) {
    visit_stmt(node->body);
  }

  virtual void visit_buffer_store(Ref<BufferStore>) {}

  virtual void visit_evaluate(Ref<Evaluate>) {}
};

/* ──────────────────────────────────────────────────────────────────────────
 * StmtMutator — rewriting traversal of a Stmt tree.
 * Override specific mutate_* methods to substitute nodes.
 * Default implementations rebuild a node only when a child has changed.
 * Loop Tiling pass uses this to split For nodes into tiled pairs.
 * ────────────────────────────────────────────────────────────────────────── */
class StmtMutator {
 public:
  virtual ~StmtMutator() = default;

  /* Entry point: dispatch to the appropriate mutate_* method. */
  virtual Ref<Stmt> mutate_stmt(Ref<Stmt> stmt);

  virtual Ref<Stmt> mutate_seq_stmt(Ref<SeqStmt> node) {
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

  virtual Ref<Stmt> mutate_for(Ref<For> node) {
    auto new_body = mutate_stmt(node->body);
    if (new_body == node->body) return node;
    return For::make(node->loop_var, node->min,
                     node->extent, node->kind,
                     std::move(new_body));
  }

  virtual Ref<Stmt> mutate_if_then_else(Ref<IfThenElse> node) {
    auto new_then = mutate_stmt(node->then_case);
    auto new_else = node->else_case ? mutate_stmt(node->else_case) : nullptr;
    if (new_then == node->then_case && new_else == node->else_case) return node;
    return IfThenElse::make(node->condition,
                             std::move(new_then),
                             std::move(new_else));
  }

  virtual Ref<Stmt> mutate_allocate(Ref<Allocate> node) {
    auto new_body = mutate_stmt(node->body);
    if (new_body == node->body) return node;
    return Allocate::make(node->buffer_var, node->dtype,
                          node->extents, std::move(new_body));
  }

  virtual Ref<Stmt> mutate_buffer_store(Ref<BufferStore> node) { return node; }

  virtual Ref<Stmt> mutate_evaluate(Ref<Evaluate> node) { return node; }
};

/* ──────────────────────────────────────────────────────────────────────────
 * StmtVisitor::visit_stmt  — type-dispatch (inline after all class defs)
 * StmtMutator::mutate_stmt — type-dispatch (inline after all class defs)
 * ────────────────────────────────────────────────────────────────────────── */
inline void StmtVisitor::visit_stmt(Ref<Stmt> stmt) {
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
      throw std::runtime_error("StmtVisitor: unknown Stmt node type");
  }
}

inline Ref<Stmt> StmtMutator::mutate_stmt(Ref<Stmt> stmt) {
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
      throw std::runtime_error("StmtMutator: unknown Stmt node type");
  }
}

} /* namespace rasp */
