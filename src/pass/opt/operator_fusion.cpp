#include "pass/opt/operator_fusion.h"

#include <string>

#include "pass/pass_manager.h"
#include "utils/log.h"

namespace rasp {

/* ── Built-in fusion patterns (longest-first; greedy matching prefers them) */

std::vector<FusionPattern> OperatorFusion::patterns_ = {
  {"conv2d_add_relu", {"nn.conv2d",     "nn.add",        "nn.relu"}},
  {"conv2d_bn_relu",  {"nn.conv2d",     "nn.batch_norm", "nn.relu"}},
  {"dense_add_relu",  {"nn.dense",      "nn.add",        "nn.relu"}},
  {"conv2d_add",      {"nn.conv2d",     "nn.add"}},
  {"add_relu",        {"nn.add",        "nn.relu"}},
};

void OperatorFusion::register_pattern(FusionPattern p) {
  patterns_.push_back(std::move(p));
}

const std::vector<FusionPattern>& OperatorFusion::all_patterns() {
  return patterns_;
}

/* ── UsageCounter ─────────────────────────────────────────────────────────── */

namespace {

/* Counts how many times each Expr* appears as an argument in any Call. */
class UsageCounter : public ExprVisitor {
 public:
  std::unordered_map<IRNode*, int> count;

  void visit_call(Ref<Call> node) override {
    for (auto& arg : node->args)
      count[arg.get()]++;
    ExprVisitor::visit_call(node); /* recurse into children */
  }
};

/* Returns true iff node is a Call to a registered Op with the given name. */
bool is_op_call(const Ref<Call>& node, const std::string& op_name) {
  if (!node->op || node->op->node_type() != IRNodeType::kOp) return false;
  return std::static_pointer_cast<Op>(node->op)->name == op_name;
}

} /* anonymous namespace */

/* ── FusionMutator::try_match ─────────────────────────────────────────────── */

std::optional<OperatorFusion::MatchedChain>
OperatorFusion::FusionMutator::try_match(Ref<Call> tail,
                                          const FusionPattern& pattern) const {
  const auto& ops = pattern.op_chain;
  int n = static_cast<int>(ops.size());
  if (n == 0) return std::nullopt;

  /* Current node must match the tail op. */
  if (!is_op_call(tail, ops[n - 1])) return std::nullopt;

  std::vector<Ref<Call>> chain(n);
  chain[n - 1] = tail;

  /* Walk backward: each node's args[0] must match the preceding op. */
  for (int i = n - 2; i >= 0; --i) {
    auto& inner_expr = chain[i + 1]->args[0];
    if (!inner_expr || inner_expr->node_type() != IRNodeType::kCall)
      return std::nullopt;
    auto inner = std::static_pointer_cast<Call>(inner_expr);
    if (!is_op_call(inner, ops[i])) return std::nullopt;

    /* Intermediate node must be consumed by exactly one other node. */
    auto it = usage_count_.find(inner.get());
    if (it != usage_count_.end() && it->second > 1) return std::nullopt;

    chain[i] = inner;
  }

  /* Collect external inputs:
   *   head  (chain[0]):   all args are external.
   *   inner (chain[i>0]): args[0] is the chain edge; args[1..] are external. */
  std::vector<Ref<Expr>> ext;
  for (int i = 0; i < n; ++i) {
    int start = (i == 0) ? 0 : 1;
    for (int j = start; j < static_cast<int>(chain[i]->args.size()); ++j)
      ext.push_back(chain[i]->args[j]);
  }

  return MatchedChain{std::move(chain), std::move(ext)};
}

/* ── FusionMutator::build_composite ──────────────────────────────────────── */

Ref<Expr> OperatorFusion::FusionMutator::build_composite(
    const MatchedChain& m, const FusionPattern& p) {
  int n       = static_cast<int>(m.chain.size());
  int num_ext = static_cast<int>(m.external_inputs.size());

  /* Create one fresh Var param per external input. */
  std::vector<Ref<Var>> params;
  params.reserve(num_ext);
  for (int i = 0; i < num_ext; ++i)
    params.push_back(Var::make("p" + std::to_string(i)));

  /* Rebuild the chain body, replacing external inputs with params. */
  int param_idx = 0;

  /* Head: all its args are external. */
  std::vector<Ref<Expr>> head_args;
  for (size_t j = 0; j < m.chain[0]->args.size(); ++j)
    head_args.push_back(params[param_idx++]);
  Ref<Expr> prev = Call::make(m.chain[0]->op,
                               std::move(head_args),
                               m.chain[0]->attrs);

  /* Inner and tail nodes: args[0] is the chain edge; args[1..] use params. */
  for (int i = 1; i < n; ++i) {
    std::vector<Ref<Expr>> args;
    args.push_back(prev);
    for (size_t j = 1; j < m.chain[i]->args.size(); ++j)
      args.push_back(params[param_idx++]);
    prev = Call::make(m.chain[i]->op, std::move(args), m.chain[i]->attrs);
  }

  auto composite_func = Function::make(
      params, prev, nullptr, {{"Composite", p.name}});

  /* Mutate external inputs so nested fusable patterns inside them are handled. */
  std::vector<Ref<Expr>> mutated_ext;
  mutated_ext.reserve(num_ext);
  for (auto& inp : m.external_inputs)
    mutated_ext.push_back(mutate(inp));

  return Call::make(composite_func, std::move(mutated_ext));
}

/* ── FusionMutator::mutate_call ───────────────────────────────────────────── */

Ref<Expr> OperatorFusion::FusionMutator::mutate_call(Ref<Call> node) {
  /* Try each pattern (longest-first) before falling back to normal recursion. */
  for (const auto& pattern : OperatorFusion::all_patterns()) {
    auto matched = try_match(node, pattern);
    if (matched)
      return build_composite(*matched, pattern);
  }
  return ExprMutator::mutate_call(node);
}

/* ── OperatorFusion::transform_function ──────────────────────────────────── */

Ref<Function> OperatorFusion::transform_function(Ref<Function> func,
                                                   Ref<IRModule> /*mod*/,
                                                   const PassContext& /*ctx*/) {
  /* Phase 1: count how many times each expression is used as an argument. */
  UsageCounter counter;
  counter.visit(func->body);

  /* Phase 2: rewrite with fusion. */
  FusionMutator mutator(counter.count);
  auto new_body = mutator.mutate(func->body);
  if (new_body == func->body) return func;
  return Function::make(func->params, std::move(new_body),
                        func->ret_type, func->attrs);
}

REGISTER_PASS(OperatorFusion);

} /* namespace rasp */
