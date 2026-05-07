#include "pass/opt/constant_folding.h"

#include <algorithm>
#include <cstring>
#include <variant>

#include "pass/pass_manager.h"
#include "utils/log.h"

namespace rasp {

/* compile-time evaluation
 * TODO:
 *   algebraic simplification
 *   shape folding
 *   symbolic folding
 *   dead branch folding
 *   constant propagation + folding
 * */

/* ── Folder::mutate_call ────────────────────────────────────────────────── */

Ref<Expr> ConstantFolding::Folder::mutate_call(Ref<Call> node) {
  /* Post-order: recurse into args first so children are folded before us. */
  auto base = ExprMutator::mutate_call(node);
  auto call = std::static_pointer_cast<Call>(base);

  /* Only fold calls to registered Ops, not inline Functions. */
  if (!call->op || call->op->node_type() != IRNodeType::kOp) return base;

  /* All args must be Constants. */
  std::vector<Ref<Constant>> const_args;
  const_args.reserve(call->args.size());
  for (auto& arg : call->args) {
    if (arg->node_type() != IRNodeType::kConstant) return base;
    const_args.push_back(std::static_pointer_cast<Constant>(arg));
  }

  auto op = std::static_pointer_cast<Op>(call->op);
  auto result = eval_op(op->name, const_args, call->attrs);
  if (!result) return base;
  return result;
}

/* ── eval_op dispatcher ─────────────────────────────────────────────────── */

Ref<Expr> ConstantFolding::Folder::eval_op(const std::string& op_name,
                                           const std::vector<Ref<Constant>>& args,
                                           const Attrs& attrs) {
  if (args.empty()) return nullptr;

  if (op_name == "nn.relu") {
    if (args.size() != 1) return nullptr;
    return eval_relu(args[0]);
  }

  if (op_name == "nn.flatten") {
    if (args.size() != 1) return nullptr;
    return eval_flatten(args[0], attrs);
  }

  if (op_name == "nn.reshape") {
    Ref<Constant> shape_arg = (args.size() >= 2) ? args[1] : nullptr;
    return eval_reshape(args[0], shape_arg, attrs);
  }

  /* Conv2d, Dense, BatchNorm, etc. — not evaluated at compile time. */
  return nullptr;
}

/* ── eval_relu ──────────────────────────────────────────────────────────── */

Ref<Constant> ConstantFolding::Folder::eval_relu(Ref<Constant> x) {
  if (!x->tensor_type) return nullptr;
  if (x->tensor_type->dtype != DataType::kFloat32) return nullptr;
  int64_t n = x->tensor_type->numel();
  if (n < 0) return nullptr;

  auto new_data = x->data; /* copy */
  if (n > 0) {
    float* ptr = reinterpret_cast<float*>(new_data.data());
    for (int64_t i = 0; i < n; ++i) {
      if (ptr[i] < 0.0f) ptr[i] = 0.0f;
    }
  }
  return Constant::make(std::move(new_data), x->tensor_type);
}

/* ── eval_flatten ───────────────────────────────────────────────────────── */

Ref<Constant> ConstantFolding::Folder::eval_flatten(Ref<Constant> x,
                                                     const Attrs& attrs) {
  if (!x->tensor_type) return nullptr;
  int64_t ndim = x->tensor_type->ndim();
  if (ndim == 0) return nullptr;

  int64_t start_dim = attrs.get_or<int64_t>("start_dim", 1LL);
  int64_t end_dim   = attrs.get_or<int64_t>("end_dim",  -1LL);

  /* Normalize negative indices. */
  if (start_dim < 0) start_dim += ndim;
  if (end_dim   < 0) end_dim   += ndim;
  start_dim = std::max(int64_t{0},  start_dim);
  end_dim   = std::min(ndim - 1,    end_dim);
  if (start_dim > end_dim) return nullptr;

  /* Build new shape. */
  std::vector<int64_t> new_shape;
  new_shape.reserve(static_cast<size_t>(ndim - (end_dim - start_dim)));
  for (int64_t i = 0; i < start_dim; ++i)
    new_shape.push_back(x->tensor_type->shape[i]);
  int64_t flat = 1;
  for (int64_t i = start_dim; i <= end_dim; ++i)
    flat *= x->tensor_type->shape[i];
  new_shape.push_back(flat);
  for (int64_t i = end_dim + 1; i < ndim; ++i)
    new_shape.push_back(x->tensor_type->shape[i]);

  auto new_type = TensorType::make(std::move(new_shape), x->tensor_type->dtype);
  return Constant::make(x->data, std::move(new_type));
}

/* ── eval_reshape ───────────────────────────────────────────────────────── */

Ref<Constant> ConstantFolding::Folder::eval_reshape(Ref<Constant> x,
                                                     Ref<Constant> shape_arg,
                                                     const Attrs& attrs) {
  if (!x->tensor_type) return nullptr;
  int64_t total = x->tensor_type->numel();
  if (total < 0) return nullptr;

  std::vector<int64_t> new_shape;

  if (shape_arg) {
    /* Shape provided as a second Constant argument (int64 data). */
    if (!shape_arg->tensor_type) return nullptr;
    if (shape_arg->tensor_type->dtype != DataType::kInt64) return nullptr;
    int64_t ndim = shape_arg->tensor_type->numel();
    if (ndim <= 0) return nullptr;
    const int64_t* p =
        reinterpret_cast<const int64_t*>(shape_arg->data.data());
    new_shape.assign(p, p + ndim);
  } else {
    /* Shape from attrs["newshape"]. */
    auto it = attrs.values.find("newshape");
    if (it == attrs.values.end()) return nullptr;
    if (!std::holds_alternative<std::vector<int64_t>>(it->second)) return nullptr;
    new_shape = std::get<std::vector<int64_t>>(it->second);
    if (new_shape.empty()) return nullptr;
  }

  /* Resolve single -1 wildcard dimension. */
  int64_t wildcard_idx = -1;
  int64_t known = 1;
  for (int64_t i = 0; i < static_cast<int64_t>(new_shape.size()); ++i) {
    if (new_shape[i] == -1) {
      if (wildcard_idx != -1) return nullptr; /* more than one -1 */
      wildcard_idx = i;
    } else {
      known *= new_shape[i];
    }
  }
  if (wildcard_idx != -1) {
    if (known == 0) return nullptr;
    new_shape[wildcard_idx] = total / known;
  }

  auto new_type = TensorType::make(std::move(new_shape), x->tensor_type->dtype);
  return Constant::make(x->data, std::move(new_type)); /* same bytes, new shape */
}

/* ── ConstantFolding::transform_function ─────────────────────────────────── */

Ref<Function> ConstantFolding::transform_function(Ref<Function> func,
                                                    Ref<IRModule> /*mod*/,
                                                    const PassContext& /*ctx*/) {
  Folder folder;
  auto new_body = folder.mutate(func->body);
  if (new_body == func->body) return func;
  return Function::make(func->params, std::move(new_body),
                        func->ret_type, func->attrs);
}

REGISTER_PASS(ConstantFolding);

} /* namespace rasp */
