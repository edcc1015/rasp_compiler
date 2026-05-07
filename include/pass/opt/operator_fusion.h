#ifndef RASP_INCLUDE_PASS_OPT_OPERATOR_FUSION_H_
#define RASP_INCLUDE_PASS_OPT_OPERATOR_FUSION_H_

#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "high_level_ir/hl_ir.h"
#include "pass/pass.h"

namespace rasp {

/* ──────────────────────────────────────────────────────────────────────────
 * FusionPattern
 * Describes a linear operator chain that can be fused into a single
 * CompositeFunction.  op_chain lists ops from head (first) to tail (last).
 * ────────────────────────────────────────────────────────────────────────── */
struct FusionPattern {
  std::string name;      /* e.g. "conv2d_add_relu"              */
  std::vector<std::string> op_chain;  /* e.g. {"nn.conv2d","nn.add","nn.relu"} */
};

/* ──────────────────────────────────────────────────────────────────────────
 * OperatorFusion  (FunctionPass, min_opt_level = 2)
 *
 * Fuses linear operator chains that match a registered FusionPattern into a
 * single  Call(CompositeFunction, external_args)  node.
 *
 * Two-phase algorithm per function:
 *   1. UsageCounter (ExprVisitor) builds an arg-usage-count map.
 *   2. FusionMutator (ExprMutator) matches patterns at every Call node;
 *      intermediate (non-tail) chain nodes must have usage_count == 1.
 *
 * The composite function carries attrs["Composite"] = pattern.name.
 * Chain connectivity is assumed to flow through args[0] of each node.
 *
 * Built-in patterns (ordered longest-first so greedy matching prefers them):
 *   conv2d_add_relu, conv2d_bn_relu, dense_add_relu  (length 3)
 *   conv2d_add, add_relu                             (length 2)
 * ────────────────────────────────────────────────────────────────────────── */
class OperatorFusion : public FunctionPass {
 public:
  std::string name() const override { return "OperatorFusion"; }
  int min_opt_level() const override { return 2; }
  std::vector<std::string> dependencies() const override {
    return {"ConstantFolding"};
  }

  Ref<Function> transform_function(Ref<Function> func,
                                   Ref<IRModule> mod,
                                   const PassContext& ctx) override;

  static void register_pattern(FusionPattern pattern);
  static const std::vector<FusionPattern>& all_patterns();

 private:
  static std::vector<FusionPattern> patterns_;

  /* Matched chain of Call nodes together with the collected external inputs. */
  struct MatchedChain {
    std::vector<Ref<Call>> chain;           /* head ... tail               */
    std::vector<Ref<Expr>> external_inputs; /* in head-to-tail order       */
  };

  class FusionMutator : public ExprMutator {
   public:
    explicit FusionMutator(const std::unordered_map<IRNode*, int>& usage_count)
        : usage_count_(usage_count) {}

    Ref<Expr> mutate_call(Ref<Call> node) override;

   private:
    const std::unordered_map<IRNode*, int>& usage_count_;

    /* Try to match pattern at tail; returns nullopt on failure. */
    std::optional<MatchedChain> try_match(Ref<Call> tail,
                                          const FusionPattern& pattern) const;

    /* Build Call(CompositeFunction, mutated_external_args).
     * External inputs are processed through *this so nested patterns inside
     * them are also fused. */
    Ref<Expr> build_composite(const MatchedChain& m, const FusionPattern& p);
  };
};

} /* namespace rasp */

#endif /* RASP_INCLUDE_PASS_OPT_OPERATOR_FUSION_H_ */
