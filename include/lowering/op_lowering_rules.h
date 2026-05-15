#ifndef RASP_INCLUDE_LOWERING_OP_LOWERING_RULES_H_
#define RASP_INCLUDE_LOWERING_OP_LOWERING_RULES_H_

#pragma once

#include <initializer_list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "high_level_ir/expr.h"
#include "low_level_ir/stmt.h"

namespace rasp {

class OpLoweringRule {
 public:
  virtual ~OpLoweringRule() = default;
  virtual std::string op_name() const = 0;
  virtual Ref<Stmt> lower(const std::vector<Ref<Buffer>>& in_bufs,
                          Ref<Buffer> out_buf,
                          const Attrs& attrs) = 0;
};

class OpLoweringRegistry {
 public:
  static void register_rule(std::shared_ptr<OpLoweringRule> rule);
  static OpLoweringRule* get(const std::string& op_name);
  static bool has(const std::string& op_name);

 private:
  static std::unordered_map<std::string, std::shared_ptr<OpLoweringRule>>& registry();
};

namespace lower_utils {

Ref<IntImm> ci(int64_t v);
Ref<FloatImm> cf(double v);
Ref<BufferLoad> load(Ref<Buffer> buf,
                     std::initializer_list<Ref<PrimExpr>> indices);
Ref<BufferStore> store(Ref<Buffer> buf,
                       std::initializer_list<Ref<PrimExpr>> indices,
                       Ref<PrimExpr> value);
Ref<Stmt> seq(std::vector<Ref<Stmt>> stmts);
Ref<PrimExpr> call_bool(const std::string& name,
                        std::vector<Ref<PrimExpr>> args);

} /* namespace lower_utils */

#define REGISTER_LOWERING_RULE(cls) \
  static bool _##cls##_rule_registered = []() { \
    ::rasp::OpLoweringRegistry::register_rule(std::make_shared<cls>()); \
    return true; \
  }()

} /* namespace rasp */

#endif /* RASP_INCLUDE_LOWERING_OP_LOWERING_RULES_H_ */
