#pragma once

#include <memory>

namespace rasp {

/* ──────────────────────────────────────────────────────────────────────────
 * IRNodeType
 * Enumerates every concrete IR node class for type dispatch.
 * ────────────────────────────────────────────────────────────────────────── */
enum class IRNodeType {
  /* Type nodes */
  kTensorType,
  kTupleType,
  kFuncType,
  /* Expr nodes */
  kVar,
  kConstant,
  kCall,
  kLet,
  kTuple,
  kTupleGetItem,
  kFunction,
  /* Module */
  kIRModule,
  /* Op */
  kOp,
};

/* ──────────────────────────────────────────────────────────────────────────
 * IRNode
 * Abstract base class for all IR nodes.
 * All nodes are heap-allocated and managed via shared_ptr (Ref<T>).
 * ────────────────────────────────────────────────────────────────────────── */
class IRNode {
 public:
  virtual ~IRNode() = default;
  virtual IRNodeType node_type() const = 0;

  /* Nodes are non-copyable; ownership is transferred through Ref<T>. */
  IRNode(const IRNode&) = delete;
  IRNode& operator=(const IRNode&) = delete;

 protected:
  IRNode() = default;
};

/* ──────────────────────────────────────────────────────────────────────────
 * Ref<T>
 * Alias for shared_ptr used throughout the IR.
 * ────────────────────────────────────────────────────────────────────────── */
template <typename T>
using Ref = std::shared_ptr<T>;

} /* namespace rasp */
