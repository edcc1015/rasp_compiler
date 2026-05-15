#ifndef RASP_INCLUDE_RUNTIME_REFERENCE_RUNTIME_H_
#define RASP_INCLUDE_RUNTIME_REFERENCE_RUNTIME_H_

#pragma once

#include <unordered_map>
#include <vector>

#include "high_level_ir/hl_ir.h"
#include "runtime/tensor.h"

namespace rasp {

class ReferenceRuntime {
 public:
  static std::vector<Tensor> run(Ref<IRModule> mod,
                                 const std::vector<Tensor>& inputs);

 private:
  using Env = std::unordered_map<IRNode*, Tensor*>;

  static Tensor eval(Ref<Expr> expr, Env& env, std::vector<Tensor>& owned);
};

bool allclose(const Tensor& a, const Tensor& b,
              float rtol = 1e-4f,
              float atol = 1e-5f);

} /* namespace rasp */

#endif /* RASP_INCLUDE_RUNTIME_REFERENCE_RUNTIME_H_ */
