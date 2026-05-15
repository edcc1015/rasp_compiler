#ifndef RASP_INCLUDE_LOWERING_LOWERING_H_
#define RASP_INCLUDE_LOWERING_LOWERING_H_

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "high_level_ir/hl_ir.h"
#include "low_level_ir/ll_ir_module.h"

namespace rasp {

class Lowering {
 public:
  static Ref<LLIRModule> lower(Ref<IRModule> hlir_mod);

 private:
  Lowering() = default;

  std::unordered_map<IRNode*, Ref<Buffer>> buffer_map_;
  int func_counter_ = 0;
  int const_counter_ = 0;

  Ref<LLIRModule> lower_function(Ref<Function> func);
  std::vector<Ref<Call>> topological_sort(Ref<Expr> body);
  Ref<Buffer> buffer_for_expr(Ref<Expr> expr, const std::string& hint);
  Ref<Buffer> alloc_buffer(const std::string& name,
                           Ref<TensorType> ttype,
                           BufferScope scope);
  Ref<PrimFunc> lower_call(Ref<Call> call, Ref<Buffer> out_buf);
  std::string gen_func_name(const std::string& op_name);
};

} /* namespace rasp */

#endif /* RASP_INCLUDE_LOWERING_LOWERING_H_ */
