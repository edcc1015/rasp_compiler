#ifndef RASP_INCLUDE_PASS_OPT_CONSTANT_FOLDING_H_
#define RASP_INCLUDE_PASS_OPT_CONSTANT_FOLDING_H_

#pragma once

#include "high_level_ir/hl_ir.h"
#include "pass/pass.h"

namespace rasp {

/* ──────────────────────────────────────────────────────────────────────────
 * ConstantFolding  (FunctionPass, min_opt_level = 1)
 *
 * Post-order ExprMutator pass: evaluates Call nodes whose every argument is
 * a Constant at compile time and replaces them with a Constant result.
 *
 * Supported ops: nn.relu, nn.flatten, nn.reshape.
 * Unsupported ops (e.g. nn.conv2d) are left unchanged even if all inputs
 * happen to be constant.
 * ────────────────────────────────────────────────────────────────────────── */
class ConstantFolding : public FunctionPass {
 public:
  std::string name() const override { return "ConstantFolding"; }
  int min_opt_level() const override { return 1; }

  Ref<Function> transform_function(Ref<Function> func,
                                   Ref<IRModule> mod,
                                   const PassContext& ctx) override;

 private:
  class Folder : public ExprMutator {
   public:
    Ref<Expr> mutate_call(Ref<Call> node) override;

   private:
    /* Dispatch to the appropriate eval_xxx based on op_name.
     * Returns nullptr if the op is not supported for compile-time evaluation. */
    Ref<Expr> eval_op(const std::string& op_name,
                      const std::vector<Ref<Constant>>& args,
                      const Attrs& attrs);

    Ref<Constant> eval_relu(Ref<Constant> x);
    Ref<Constant> eval_flatten(Ref<Constant> x, const Attrs& attrs);
    /* shape_arg: second Constant arg when reshape is called as (data, shape);
     * nullptr when the shape is taken from attrs["newshape"]. */
    Ref<Constant> eval_reshape(Ref<Constant> x,
                               Ref<Constant> shape_arg,
                               const Attrs& attrs);
  };
};

} /* namespace rasp */

#endif /* RASP_INCLUDE_PASS_OPT_CONSTANT_FOLDING_H_ */
