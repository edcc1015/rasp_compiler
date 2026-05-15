#include "runtime/reference_runtime.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace rasp {

namespace {

Ref<TensorType> tensor_type_of(Ref<Expr> expr) {
  if (!expr->checked_type || expr->checked_type->node_type() != IRNodeType::kTensorType) {
    throw std::runtime_error("ReferenceRuntime: expression has no TensorType");
  }
  return std::static_pointer_cast<TensorType>(expr->checked_type);
}

int64_t offset(const std::vector<int64_t>& shape,
               const std::vector<int64_t>& idx) {
  int64_t out = 0;
  int64_t stride = 1;
  for (int i = static_cast<int>(idx.size()) - 1; i >= 0; --i) {
    out += idx[i] * stride;
    stride *= shape[i];
  }
  return out;
}

Tensor clone_tensor(const Tensor& src) {
  Tensor dst = Tensor::allocate(src.shape, src.dtype);
  std::memcpy(dst.data, src.data, src.nbytes());
  return dst;
}

Tensor constant_to_tensor(Ref<Constant> c) {
  Tensor t = Tensor::allocate(c->tensor_type->shape, DType::kFloat32);
  std::memcpy(t.data, c->data.data(), c->data.size());
  return t;
}

std::string op_name(Ref<Call> call) {
  if (call->op && call->op->node_type() == IRNodeType::kOp) {
    return std::static_pointer_cast<Op>(call->op)->name;
  }
  throw std::runtime_error("ReferenceRuntime: unsupported call op");
}

} /* anonymous namespace */

std::vector<Tensor> ReferenceRuntime::run(Ref<IRModule> mod,
                                          const std::vector<Tensor>& inputs) {
  auto fn = mod->get_function("main");
  if (inputs.size() != fn->params.size()) {
    throw std::runtime_error("ReferenceRuntime: wrong input count");
  }
  Env env;
  std::vector<Tensor> owned;
  for (size_t i = 0; i < fn->params.size(); ++i) {
    env[fn->params[i].get()] = const_cast<Tensor*>(&inputs[i]);
  }
  std::vector<Tensor> outputs;
  outputs.push_back(eval(fn->body, env, owned));
  return outputs;
}

Tensor ReferenceRuntime::eval(Ref<Expr> expr, Env& env, std::vector<Tensor>& owned) {
  (void)owned;
  switch (expr->node_type()) {
    case IRNodeType::kVar: {
      auto it = env.find(expr.get());
      if (it == env.end()) throw std::runtime_error("ReferenceRuntime: unbound var");
      return clone_tensor(*it->second);
    }
    case IRNodeType::kConstant:
      return constant_to_tensor(std::static_pointer_cast<Constant>(expr));
    case IRNodeType::kCall: {
      auto call = std::static_pointer_cast<Call>(expr);
      auto name = op_name(call);
      std::vector<Tensor> args;
      for (auto& arg : call->args) args.push_back(eval(arg, env, owned));
      auto tt = tensor_type_of(call);
      Tensor out = Tensor::allocate(tt->shape, DType::kFloat32);
      float* y = out.data_f32();
      if (name == "nn.relu") {
        const float* x = args[0].data_f32();
        for (int64_t i = 0; i < out.numel(); ++i) y[i] = std::max(0.0f, x[i]);
        return out;
      }
      if (name == "nn.add") {
        const float* a = args[0].data_f32();
        const float* b = args[1].data_f32();
        if (args[0].numel() == args[1].numel()) {
          for (int64_t i = 0; i < out.numel(); ++i) y[i] = a[i] + b[i];
        } else {
          for (int64_t i = 0; i < out.numel(); ++i) y[i] = a[i] + b[i % args[1].numel()];
        }
        return out;
      }
      if (name == "nn.dense") {
        const float* x = args[0].data_f32();
        const float* w = args[1].data_f32();
        const float* b = args.size() > 2 ? args[2].data_f32() : nullptr;
        int64_t n = args[0].shape[0];
        int64_t ic = args[0].shape[1];
        int64_t oc = args[1].shape[0];
        for (int64_t i = 0; i < n; ++i) {
          for (int64_t o = 0; o < oc; ++o) {
            float acc = b ? b[o] : 0.0f;
            for (int64_t c = 0; c < ic; ++c) acc += x[i * ic + c] * w[o * ic + c];
            y[i * oc + o] = acc;
          }
        }
        return out;
      }
      if (name == "nn.conv2d") {
        const float* x = args[0].data_f32();
        const float* w = args[1].data_f32();
        auto strides = call->attrs.get_or<std::vector<int64_t>>("strides", {1, 1});
        auto padding = call->attrs.get_or<std::vector<int64_t>>("padding", {0, 0, 0, 0});
        int64_t n = out.shape[0], oc = out.shape[1], oh = out.shape[2], ow = out.shape[3];
        int64_t ic = args[0].shape[1], ih = args[0].shape[2], iw = args[0].shape[3];
        int64_t kh = args[1].shape[2], kw = args[1].shape[3];
        for (int64_t ni = 0; ni < n; ++ni) {
          for (int64_t oo = 0; oo < oc; ++oo) {
            for (int64_t ho = 0; ho < oh; ++ho) {
              for (int64_t wo = 0; wo < ow; ++wo) {
                float acc = 0.0f;
                for (int64_t ci = 0; ci < ic; ++ci) {
                  for (int64_t hk = 0; hk < kh; ++hk) {
                    for (int64_t wk = 0; wk < kw; ++wk) {
                      int64_t hi = ho * strides[0] + hk - padding[0];
                      int64_t wi = wo * strides[1] + wk - padding[1];
                      if (hi >= 0 && hi < ih && wi >= 0 && wi < iw) {
                        acc += x[offset(args[0].shape, {ni, ci, hi, wi})] *
                               w[offset(args[1].shape, {oo, ci, hk, wk})];
                      }
                    }
                  }
                }
                y[offset(out.shape, {ni, oo, ho, wo})] = acc;
              }
            }
          }
        }
        return out;
      }
      throw std::runtime_error("ReferenceRuntime: unsupported op " + name);
    }
    default:
      throw std::runtime_error("ReferenceRuntime: unsupported expression");
  }
}

bool allclose(const Tensor& a, const Tensor& b, float rtol, float atol) {
  if (a.numel() != b.numel()) return false;
  const float* x = a.data_f32();
  const float* y = b.data_f32();
  for (int64_t i = 0; i < a.numel(); ++i) {
    float diff = std::fabs(x[i] - y[i]);
    if (diff > atol + rtol * std::fabs(y[i])) return false;
  }
  return true;
}

} /* namespace rasp */
