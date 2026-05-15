#ifndef RASP_INCLUDE_PASS_ANALYSIS_TYPE_INFERENCE_H_
#define RASP_INCLUDE_PASS_ANALYSIS_TYPE_INFERENCE_H_

#pragma once

#include <string>
#include <unordered_map>

#include "pass/pass.h"

namespace rasp {

/* ──────────────────────────────────────────────────────────────────────────
 * TypeInference
 * Fills Expr::checked_type for every HLIR expression and Function::ret_type.
 * ────────────────────────────────────────────────────────────────────────── */
class TypeInference : public FunctionPass {
 public:
  std::string name() const override { return "TypeInference"; }
  int min_opt_level() const override { return 0; }

  Ref<Function> transform_function(Ref<Function> func,
                                   Ref<IRModule> mod,
                                   const PassContext& ctx) override;

 private:
  class Inferencer;
};

} /* namespace rasp */

#endif /* RASP_INCLUDE_PASS_ANALYSIS_TYPE_INFERENCE_H_ */
