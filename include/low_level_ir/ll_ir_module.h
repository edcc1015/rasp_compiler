#ifndef RASP_INCLUDE_LOW_LEVEL_IR_LL_IR_MODULE_H_
#define RASP_INCLUDE_LOW_LEVEL_IR_LL_IR_MODULE_H_

#pragma once

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "low_level_ir/prim_func.h"

namespace rasp {

/* ──────────────────────────────────────────────────────────────────────────
 * LLIRModule
 * Top-level container for all PrimFuncs produced by the lowering pass.
 *
 * exec_order preserves the topological execution order inherited from the
 * HLIR computation graph, ensuring that data dependencies are respected when
 * functions are dispatched at runtime.
 * ────────────────────────────────────────────────────────────────────────── */
class LLIRModule : public IRNode {
 public:
  std::unordered_map<std::string, Ref<PrimFunc>> functions;
  std::vector<std::string> exec_order;

  IRNodeType node_type() const override { return IRNodeType::kLLIRModule; }

  static Ref<LLIRModule> make();

  void add_func(Ref<PrimFunc> func) {
    const std::string& n = func->name;
    functions[n] = func;
    exec_order.push_back(n);
  }

  Ref<PrimFunc> get_func(const std::string& name) const {
    auto it = functions.find(name);
    if (it == functions.end())
      throw std::runtime_error("LLIRModule: function not found: " + name);
    return it->second;
  }

  bool has_func(const std::string& name) const {
    return functions.count(name) > 0;
  }

 private:
  LLIRModule() = default;
};

inline Ref<LLIRModule> LLIRModule::make() {
  return std::shared_ptr<LLIRModule>(new LLIRModule());
}

} /* namespace rasp */

#endif /* RASP_INCLUDE_LOW_LEVEL_IR_LL_IR_MODULE_H_ */
