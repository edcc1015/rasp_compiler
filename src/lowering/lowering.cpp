#include "lowering/lowering.h"

#include <algorithm>
#include <cctype>
#include <functional>
#include <stdexcept>
#include <unordered_set>

#include "lowering/op_lowering_rules.h"

namespace rasp {

namespace {

Ref<TensorType> checked_tensor(const Ref<Expr>& expr) {
  if (!expr->checked_type || expr->checked_type->node_type() != IRNodeType::kTensorType) {
    throw std::runtime_error("Lowering: expression is missing TensorType checked_type");
  }
  return std::static_pointer_cast<TensorType>(expr->checked_type);
}

Ref<TensorType> checked_tensor_type(const Ref<IRNode>& type,
                                    const std::string& ctx) {
  if (!type || type->node_type() != IRNodeType::kTensorType) {
    throw std::runtime_error("Lowering: expected TensorType for " + ctx);
  }
  return std::static_pointer_cast<TensorType>(type);
}

std::string op_name_of(const Ref<Call>& call) {
  if (call->op && call->op->node_type() == IRNodeType::kOp) {
    return std::static_pointer_cast<Op>(call->op)->name;
  }
  if (call->op && call->op->node_type() == IRNodeType::kFunction) {
    auto fn = std::static_pointer_cast<Function>(call->op);
    auto it = fn->attrs.find("Composite");
    if (it != fn->attrs.end()) return it->second;
  }
  throw std::runtime_error("Lowering: unsupported call op");
}

std::string sanitize_name(std::string name) {
  for (char& c : name) {
    if (!std::isalnum(static_cast<unsigned char>(c))) c = '_';
  }
  return name;
}

} /* anonymous namespace */

Ref<LLIRModule> Lowering::lower(Ref<IRModule> hlir_mod) {
  if (!hlir_mod->has_function("main")) {
    throw std::runtime_error("Lowering: IRModule has no main function");
  }
  Lowering lowering;
  return lowering.lower_function(hlir_mod->get_function("main"));
}

Ref<LLIRModule> Lowering::lower_function(Ref<Function> func) {
  auto mod = LLIRModule::make();
  for (auto& param : func->params) {
    auto ttype = checked_tensor_type(param->checked_type, "lowering parameter");
    buffer_map_[param.get()] = alloc_buffer("input_" + param->name, ttype,
                                            BufferScope::kGlobal);
  }
  auto calls = topological_sort(func->body);
  for (auto& call : calls) {
    bool is_output = call.get() == func->body.get();
    auto out_buf = alloc_buffer(is_output ? "output" : "buf_" + gen_func_name(op_name_of(call)),
                                checked_tensor(call),
                                is_output ? BufferScope::kGlobal : BufferScope::kLocal);
    buffer_map_[call.get()] = out_buf;
    mod->add_func(lower_call(call, out_buf));
  }
  return mod;
}

std::vector<Ref<Call>> Lowering::topological_sort(Ref<Expr> body) {
  std::vector<Ref<Call>> result;
  std::unordered_set<IRNode*> visited;
  std::function<void(Ref<Expr>)> dfs = [&](Ref<Expr> expr) {
    if (!expr || visited.count(expr.get())) return;
    visited.insert(expr.get());
    switch (expr->node_type()) {
      case IRNodeType::kCall: {
        auto call = std::static_pointer_cast<Call>(expr);
        for (auto& arg : call->args) dfs(arg);
        result.push_back(call);
        break;
      }
      case IRNodeType::kLet: {
        auto let = std::static_pointer_cast<Let>(expr);
        dfs(let->value);
        dfs(let->body);
        break;
      }
      case IRNodeType::kTuple: {
        auto tuple = std::static_pointer_cast<Tuple>(expr);
        for (auto& field : tuple->fields) dfs(field);
        break;
      }
      case IRNodeType::kTupleGetItem: {
        dfs(std::static_pointer_cast<TupleGetItem>(expr)->tuple);
        break;
      }
      default:
        break;
    }
  };
  dfs(body);
  return result;
}

Ref<Buffer> Lowering::buffer_for_expr(Ref<Expr> expr, const std::string& hint) {
  auto it = buffer_map_.find(expr.get());
  if (it != buffer_map_.end()) return it->second;
  if (expr->node_type() == IRNodeType::kConstant) {
    auto c = std::static_pointer_cast<Constant>(expr);
    auto name = "const_" + sanitize_name(hint) + "_" + std::to_string(const_counter_++);
    auto buf = alloc_buffer(name, c->tensor_type, BufferScope::kGlobal);
    buf->data = c->data;
    buffer_map_[expr.get()] = buf;
    return buf;
  }
  if (expr->node_type() == IRNodeType::kVar) {
    auto v = std::static_pointer_cast<Var>(expr);
    auto ttype = checked_tensor(v);
    auto buf = alloc_buffer("input_" + v->name, ttype, BufferScope::kGlobal);
    buffer_map_[expr.get()] = buf;
    return buf;
  }
  throw std::runtime_error("Lowering: missing buffer for expression");
}

Ref<Buffer> Lowering::alloc_buffer(const std::string& name,
                                   Ref<TensorType> ttype,
                                   BufferScope scope) {
  std::vector<Ref<PrimExpr>> shape;
  shape.reserve(ttype->shape.size());
  for (int64_t dim : ttype->shape) {
    if (dim < 0) {
      throw std::runtime_error("Lowering: dynamic shapes are not supported");
    }
    shape.push_back(IntImm::make(dim));
  }
  return Buffer::make(name, ttype->dtype, std::move(shape), {}, 16, scope);
}

Ref<PrimFunc> Lowering::lower_call(Ref<Call> call, Ref<Buffer> out_buf) {
  auto op_name = op_name_of(call);
  std::vector<Ref<Buffer>> in_bufs;
  in_bufs.reserve(call->args.size());
  for (size_t i = 0; i < call->args.size(); ++i) {
    in_bufs.push_back(buffer_for_expr(call->args[i], op_name + "_" + std::to_string(i)));
  }

  Ref<Stmt> body;
  if (call->op && call->op->node_type() == IRNodeType::kFunction) {
    auto fn = std::static_pointer_cast<Function>(call->op);
    std::unordered_map<IRNode*, Ref<Buffer>> local_buffers;
    for (size_t i = 0; i < fn->params.size(); ++i) {
      local_buffers[fn->params[i].get()] = in_bufs.at(i);
    }
    std::vector<Ref<Stmt>> stmts;
    auto inner_calls = topological_sort(fn->body);
    for (auto& inner : inner_calls) {
      bool is_output = inner.get() == fn->body.get();
      auto inner_out = is_output
        ? out_buf
        : alloc_buffer("buf_" + gen_func_name(op_name_of(inner)),
                       checked_tensor(inner), BufferScope::kLocal);
      local_buffers[inner.get()] = inner_out;

      std::vector<Ref<Buffer>> inner_inputs;
      inner_inputs.reserve(inner->args.size());
      for (size_t i = 0; i < inner->args.size(); ++i) {
        auto arg = inner->args[i];
        auto it = local_buffers.find(arg.get());
        if (it != local_buffers.end()) {
          inner_inputs.push_back(it->second);
        } else if (arg->node_type() == IRNodeType::kConstant) {
          inner_inputs.push_back(buffer_for_expr(arg, op_name_of(inner) + "_" + std::to_string(i)));
        } else {
          throw std::runtime_error("Lowering: composite function has unbound input");
        }
      }

      auto* rule = OpLoweringRegistry::get(op_name_of(inner));
      if (!rule) {
        throw std::runtime_error("Lowering: no lowering rule for op '" +
                                 op_name_of(inner) + "'");
      }
      stmts.push_back(rule->lower(inner_inputs, inner_out, inner->attrs));
    }
    body = lower_utils::seq(std::move(stmts));
  } else {
    auto* rule = OpLoweringRegistry::get(op_name);
    if (!rule) {
      throw std::runtime_error("Lowering: no lowering rule for op '" + op_name + "'");
    }
    body = rule->lower(in_bufs, out_buf, call->attrs);
  }

  auto params = in_bufs;
  params.push_back(out_buf);
  return PrimFunc::make(gen_func_name(op_name), std::move(params), body,
                        {{"op", op_name}});
}

std::string Lowering::gen_func_name(const std::string& op_name) {
  return sanitize_name(op_name) + "_" + std::to_string(func_counter_++);
}

} /* namespace rasp */
