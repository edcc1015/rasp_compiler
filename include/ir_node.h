#ifndef RASP_INCLUDE_IR_NODE_H_
#define RASP_INCLUDE_IR_NODE_H_

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
  /* LLIR PrimExpr nodes */
  kLLVar,
  kIntImm,
  kFloatImm,
  kBufferLoad,
  kAdd,
  kSub,
  kMul,
  kDiv,
  kMod,
  kFloorDiv,
  kFloorMod,
  kMin,
  kMax,
  kSelect,
  kCast,
  kRamp,
  kBroadcast,
  kLLCall,
  /* LLIR Stmt nodes */
  kSeqStmt,
  kFor,
  kIfThenElse,
  kAllocate,
  kBufferStore,
  kEvaluate,
  /* LLIR Buffer / PrimFunc / Module */
  kBuffer,
  kPrimFunc,
  kLLIRModule,
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

#endif /* RASP_INCLUDE_IR_NODE_H_ */
