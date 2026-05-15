#include "pass/analysis/type_inference.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "high_level_ir/op.h"
#include "pass/pass_manager.h"

namespace rasp {

namespace {

Ref<TensorType> as_tensor(const Ref<IRNode>& type, const std::string& ctx) {
  if (!type || type->node_type() != IRNodeType::kTensorType) {
    throw std::runtime_error("TypeInference: expected TensorType for " + ctx);
  }
  return std::static_pointer_cast<TensorType>(type);
}

int64_t positive_axis(int64_t axis, int64_t ndim) {
  if (axis < 0) axis += ndim;
  if (axis < 0 || axis >= ndim) {
    throw std::runtime_error("TypeInference: axis out of range");
  }
  return axis;
}

std::vector<int64_t> get_ints(const Attrs& attrs,
                              const std::string& key,
                              std::vector<int64_t> def) {
  return attrs.get_or<std::vector<int64_t>>(key, std::move(def));
}

int64_t get_int(const Attrs& attrs, const std::string& key, int64_t def) {
  return attrs.get_or<int64_t>(key, def);
}

std::vector<int64_t> broadcast_shape(const std::vector<int64_t>& a,
                                     const std::vector<int64_t>& b) {
  size_t n = std::max(a.size(), b.size());
  std::vector<int64_t> out(n, 1);
  for (size_t i = 0; i < n; ++i) {
    int64_t da = 1;
    int64_t db = 1;
    if (i < a.size()) da = a[a.size() - 1 - i];
    if (i < b.size()) db = b[b.size() - 1 - i];
    if (da != db && da != 1 && db != 1 && da >= 0 && db >= 0) {
      throw std::runtime_error("TypeInference: incompatible broadcast shapes");
    }
    out[n - 1 - i] = (da == 1) ? db : da;
    if (da < 0 || db < 0) out[n - 1 - i] = -1;
  }
  return out;
}

int64_t conv_out_dim(int64_t in_dim, int64_t k, int64_t stride,
                     int64_t pad_before, int64_t pad_after, int64_t dilation) {
  if (in_dim < 0 || k < 0) return -1;
  int64_t kernel = dilation * (k - 1) + 1;
  return (in_dim + pad_before + pad_after - kernel) / stride + 1;
}

} /* anonymous namespace */

class TypeInference::Inferencer {
 public:
  explicit Inferencer(Ref<Function> func) {
    for (auto& param : func->params) {
      if (!param->type_annotation) {
        throw std::runtime_error(
          "TypeInference: parameter '" + param->name + "' has no type annotation");
      }
      env_[param.get()] = param->type_annotation;
      param->checked_type = param->type_annotation;
    }
  }

  Ref<IRNode> infer(Ref<Expr> expr) {
    switch (expr->node_type()) {
      case IRNodeType::kVar:
        return infer_var(std::static_pointer_cast<Var>(expr));
      case IRNodeType::kConstant:
        return infer_constant(std::static_pointer_cast<Constant>(expr));
      case IRNodeType::kCall:
        return infer_call(std::static_pointer_cast<Call>(expr));
      case IRNodeType::kLet:
        return infer_let(std::static_pointer_cast<Let>(expr));
      case IRNodeType::kTuple:
        return infer_tuple(std::static_pointer_cast<Tuple>(expr));
      case IRNodeType::kTupleGetItem:
        return infer_tuple_get_item(std::static_pointer_cast<TupleGetItem>(expr));
      case IRNodeType::kFunction:
        return infer_function(std::static_pointer_cast<Function>(expr));
      default:
        throw std::runtime_error("TypeInference: unsupported expression node");
    }
  }

 private:
  std::unordered_map<Var*, Ref<IRNode>> env_;

  Ref<IRNode> set_type(const Ref<Expr>& expr, Ref<IRNode> type) {
    expr->checked_type = type;
    return type;
  }

  Ref<IRNode> infer_var(Ref<Var> var) {
    auto it = env_.find(var.get());
    if (it != env_.end()) return set_type(var, it->second);
    if (var->type_annotation) return set_type(var, var->type_annotation);
    throw std::runtime_error("TypeInference: unbound variable '" + var->name + "'");
  }

  Ref<IRNode> infer_constant(Ref<Constant> c) {
    return set_type(c, c->tensor_type);
  }

  Ref<IRNode> infer_tuple(Ref<Tuple> tuple) {
    std::vector<Ref<IRNode>> fields;
    fields.reserve(tuple->fields.size());
    for (auto& field : tuple->fields) fields.push_back(infer(field));
    return set_type(tuple, TupleType::make(std::move(fields)));
  }

  Ref<IRNode> infer_tuple_get_item(Ref<TupleGetItem> tgi) {
    auto tuple_type = infer(tgi->tuple);
    if (!tuple_type || tuple_type->node_type() != IRNodeType::kTupleType) {
      throw std::runtime_error("TypeInference: TupleGetItem source is not tuple");
    }
    auto tt = std::static_pointer_cast<TupleType>(tuple_type);
    if (tgi->index < 0 || tgi->index >= static_cast<int>(tt->fields.size())) {
      throw std::runtime_error("TypeInference: TupleGetItem index out of range");
    }
    return set_type(tgi, tt->fields[tgi->index]);
  }

  Ref<IRNode> infer_let(Ref<Let> let) {
    auto value_type = infer(let->value);
    env_[let->var.get()] = value_type;
    let->var->checked_type = value_type;
    return set_type(let, infer(let->body));
  }

  Ref<IRNode> infer_function(Ref<Function> fn) {
    Inferencer nested(fn);
    auto ret = nested.infer(fn->body);
    fn->ret_type = ret;
    std::vector<Ref<IRNode>> arg_types;
    for (auto& param : fn->params) arg_types.push_back(param->checked_type);
    return set_type(fn, FuncType::make(std::move(arg_types), ret));
  }

  Ref<IRNode> infer_call(Ref<Call> call) {
    std::vector<Ref<IRNode>> arg_types;
    arg_types.reserve(call->args.size());
    for (auto& arg : call->args) arg_types.push_back(infer(arg));

    Ref<IRNode> out;
    if (call->op && call->op->node_type() == IRNodeType::kFunction) {
      auto func = std::static_pointer_cast<Function>(call->op);
      if (func->params.size() != arg_types.size()) {
        throw std::runtime_error("TypeInference: function call arity mismatch");
      }
      for (size_t i = 0; i < func->params.size(); ++i) {
        if (!func->params[i]->type_annotation) {
          func->params[i]->type_annotation = arg_types[i];
        }
        func->params[i]->checked_type = arg_types[i];
      }
      infer_function(func);
      out = func->ret_type;
    } else if (call->op && call->op->node_type() == IRNodeType::kOp) {
      auto op = std::static_pointer_cast<Op>(call->op);
      out = infer_op(op->name, call->attrs, arg_types);
    } else {
      throw std::runtime_error("TypeInference: Call has invalid op");
    }
    return set_type(call, out);
  }

  Ref<IRNode> infer_op(const std::string& op_name,
                       const Attrs& attrs,
                       const std::vector<Ref<IRNode>>& args) {
    if (op_name == "nn.relu" || op_name == "nn.clip" ||
        op_name == "nn.sigmoid" || op_name == "nn.softmax" ||
        op_name == "nn.batch_norm") {
      return args.at(0);
    }
    if (op_name == "nn.add") {
      auto a = as_tensor(args.at(0), op_name + " lhs");
      auto b = as_tensor(args.at(1), op_name + " rhs");
      return TensorType::make(broadcast_shape(a->shape, b->shape), a->dtype);
    }
    if (op_name == "nn.conv2d") {
      auto data = as_tensor(args.at(0), "conv2d data");
      auto weight = as_tensor(args.at(1), "conv2d weight");
      auto strides = get_ints(attrs, "strides", {1, 1});
      auto padding = get_ints(attrs, "padding", {0, 0, 0, 0});
      auto dilation = get_ints(attrs, "dilation", {1, 1});
      int64_t oh = conv_out_dim(data->shape.at(2), weight->shape.at(2),
                                strides.at(0), padding.at(0),
                                padding.at(2), dilation.at(0));
      int64_t ow = conv_out_dim(data->shape.at(3), weight->shape.at(3),
                                strides.at(1), padding.at(1),
                                padding.at(3), dilation.at(1));
      return TensorType::make({data->shape.at(0), weight->shape.at(0), oh, ow},
                              data->dtype);
    }
    if (op_name == "nn.max_pool2d" || op_name == "nn.avg_pool2d") {
      auto data = as_tensor(args.at(0), op_name + " data");
      auto pool = get_ints(attrs, "pool_size", {1, 1});
      auto strides = get_ints(attrs, "strides", pool);
      auto padding = get_ints(attrs, "padding", {0, 0, 0, 0});
      int64_t oh = conv_out_dim(data->shape.at(2), pool.at(0), strides.at(0),
                                padding.at(0), padding.at(2), 1);
      int64_t ow = conv_out_dim(data->shape.at(3), pool.at(1), strides.at(1),
                                padding.at(1), padding.at(3), 1);
      return TensorType::make({data->shape.at(0), data->shape.at(1), oh, ow},
                              data->dtype);
    }
    if (op_name == "nn.global_avg_pool2d") {
      auto data = as_tensor(args.at(0), "global_avg_pool2d data");
      return TensorType::make({data->shape.at(0), data->shape.at(1), 1, 1},
                              data->dtype);
    }
    if (op_name == "nn.dense") {
      auto data = as_tensor(args.at(0), "dense data");
      auto weight = as_tensor(args.at(1), "dense weight");
      return TensorType::make({data->shape.at(0), weight->shape.at(0)},
                              data->dtype);
    }
    if (op_name == "nn.matmul") {
      auto a = as_tensor(args.at(0), "matmul lhs");
      auto b = as_tensor(args.at(1), "matmul rhs");
      bool ta = get_int(attrs, "transpose_a", 0) != 0;
      bool tb = get_int(attrs, "transpose_b", 0) != 0;
      int64_t m = ta ? a->shape.at(1) : a->shape.at(0);
      int64_t n = tb ? b->shape.at(0) : b->shape.at(1);
      return TensorType::make({m, n}, a->dtype);
    }
    if (op_name == "nn.concatenate") {
      auto first = as_tensor(args.at(0), "concatenate input");
      auto shape = first->shape;
      int64_t axis = positive_axis(get_int(attrs, "axis", 0), first->ndim());
      shape[axis] = 0;
      for (auto& arg : args) {
        auto tt = as_tensor(arg, "concatenate input");
        shape[axis] += tt->shape[axis];
      }
      return TensorType::make(std::move(shape), first->dtype);
    }
    if (op_name == "nn.flatten") {
      auto data = as_tensor(args.at(0), "flatten data");
      int64_t ndim = data->ndim();
      int64_t start = positive_axis(get_int(attrs, "start_dim", 1), ndim);
      int64_t end = get_int(attrs, "end_dim", -1);
      end = positive_axis(end, ndim);
      std::vector<int64_t> shape;
      for (int64_t i = 0; i < start; ++i) shape.push_back(data->shape[i]);
      int64_t flat = 1;
      for (int64_t i = start; i <= end; ++i) {
        if (data->shape[i] < 0) flat = -1;
        if (flat >= 0) flat *= data->shape[i];
      }
      shape.push_back(flat);
      for (int64_t i = end + 1; i < ndim; ++i) shape.push_back(data->shape[i]);
      return TensorType::make(std::move(shape), data->dtype);
    }
    if (op_name == "nn.reshape") {
      auto data = as_tensor(args.at(0), "reshape data");
      auto shape = get_ints(attrs, "newshape", {});
      int64_t infer_idx = -1;
      int64_t known = 1;
      for (int64_t i = 0; i < static_cast<int64_t>(shape.size()); ++i) {
        if (shape[i] == -1) {
          infer_idx = i;
        } else {
          known *= shape[i];
        }
      }
      if (infer_idx >= 0 && data->numel() >= 0) shape[infer_idx] = data->numel() / known;
      return TensorType::make(std::move(shape), data->dtype);
    }
    throw std::runtime_error("TypeInference: unsupported op '" + op_name + "'");
  }
};

Ref<Function> TypeInference::transform_function(Ref<Function> func,
                                                 Ref<IRModule> /*mod*/,
                                                 const PassContext& /*ctx*/) {
  Inferencer inferencer(func);
  auto ret = inferencer.infer(func->body);
  func->ret_type = ret;
  std::vector<Ref<IRNode>> arg_types;
  arg_types.reserve(func->params.size());
  for (auto& param : func->params) arg_types.push_back(param->checked_type);
  func->checked_type = FuncType::make(std::move(arg_types), ret);
  return func;
}

REGISTER_PASS(TypeInference);

} /* namespace rasp */
