#ifndef RASP_INCLUDE_HIGH_LEVEL_IR_IR_MODULE_H_
#define RASP_INCLUDE_HIGH_LEVEL_IR_IR_MODULE_H_

#pragma once

#include <stdexcept>
#include <string>
#include <unordered_map>

#include "ir_node.h"
#include "expr.h"

namespace rasp {

/* ──────────────────────────────────────────────────────────────────────────
 * IRModule
 * Top-level container for named functions.
 * A typical model has exactly one function, "main".
 * ────────────────────────────────────────────────────────────────────────── */
class IRModule : public IRNode {
 public:
  std::unordered_map<std::string, Ref<Function>> functions;

  IRNodeType node_type() const override { return IRNodeType::kIRModule; }

  static Ref<IRModule> make();

  void add_function(const std::string& name, Ref<Function> func) {
    functions[name] = std::move(func);
  }

  Ref<Function> get_function(const std::string& name) const {
    auto it = functions.find(name);
    if (it == functions.end()) {
      throw std::runtime_error("Function not found in IRModule: " + name);
    }
    return it->second;
  }

  bool has_function(const std::string& name) const {
    return functions.count(name) > 0;
  }

  /* Serialize the module to a JSON string (compact). */
  std::string to_json() const;

  /* Deserialize an IRModule from a JSON string produced by to_json(). */
  static Ref<IRModule> from_json(const std::string& json_str);

 private:
  IRModule() = default;
};

inline Ref<IRModule> IRModule::make() {
  return std::shared_ptr<IRModule>(new IRModule());
}

} /* namespace rasp */

#endif /* RASP_INCLUDE_HIGH_LEVEL_IR_IR_MODULE_H_ */
