#ifndef RASP_INCLUDE_HIGH_LEVEL_IR_HL_IR_H_
#define RASP_INCLUDE_HIGH_LEVEL_IR_HL_IR_H_

#pragma once

/*
 * Convenience header: include all High-level IR definitions in dependency order.
 *
 *   ir_node.h      — IRNode base, IRNodeType, Ref<T>
 *   type.h         — DataType, TensorType, TupleType, FuncType
 *   op.h           — AttrSchema, Op, OpRegistry
 *   expr.h         — Expr, Var, Constant, Attrs, Call, Let, Tuple, TupleGetItem, Function
 *   ir_module.h    — IRModule
 *   expr_visitor.h — ExprVisitor (read-only traversal)
 *   expr_mutator.h — ExprMutator (tree-rewriting traversal)
 */
#include "ir_node.h"
#include "type.h"
#include "op.h"
#include "expr.h"
#include "ir_module.h"
#include "hl_expr_visitor.h"
#include "hl_expr_mutator.h"

#endif /* RASP_INCLUDE_HIGH_LEVEL_IR_HL_IR_H_ */
