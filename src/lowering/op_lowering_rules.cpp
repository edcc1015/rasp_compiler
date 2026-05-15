#include "lowering/op_lowering_rules.h"

#include <functional>
#include <stdexcept>

namespace rasp {

namespace {

int64_t int_value(const Ref<PrimExpr>& expr) {
  if (!expr || expr->node_type() != IRNodeType::kIntImm) {
    throw std::runtime_error("Lowering: expected static IntImm shape");
  }
  return std::static_pointer_cast<IntImm>(expr)->value;
}

std::vector<int64_t> shape_of(const Ref<Buffer>& buf) {
  std::vector<int64_t> shape;
  shape.reserve(buf->shape.size());
  for (auto& dim : buf->shape) shape.push_back(int_value(dim));
  return shape;
}

Ref<Stmt> loop_nest(const std::vector<int64_t>& extents,
                    const std::vector<std::string>& names,
                    const std::function<Ref<Stmt>(const std::vector<Ref<PrimExpr>>&)>& body) {
  std::vector<Ref<LLVar>> vars;
  vars.reserve(extents.size());
  for (auto& name : names) vars.push_back(LLVar::make(name));
  Ref<Stmt> stmt = body(std::vector<Ref<PrimExpr>>(vars.begin(), vars.end()));
  for (int i = static_cast<int>(extents.size()) - 1; i >= 0; --i) {
    stmt = For::make(vars[i], lower_utils::ci(0), lower_utils::ci(extents[i]),
                     ForKind::kSerial, stmt);
  }
  return stmt;
}

std::vector<Ref<PrimExpr>> broadcast_indices(const std::vector<Ref<PrimExpr>>& out_idx,
                                             const std::vector<int64_t>& in_shape) {
  std::vector<Ref<PrimExpr>> idx(in_shape.size());
  int offset = static_cast<int>(out_idx.size()) - static_cast<int>(in_shape.size());
  for (int i = 0; i < static_cast<int>(in_shape.size()); ++i) {
    idx[i] = (in_shape[i] == 1) ? lower_utils::ci(0) : out_idx[i + offset];
  }
  return idx;
}

Ref<Stmt> elementwise_unary(Ref<Buffer> in,
                            Ref<Buffer> out,
                            const std::function<Ref<PrimExpr>(Ref<PrimExpr>)>& map) {
  auto out_shape = shape_of(out);
  std::vector<std::string> names;
  for (size_t i = 0; i < out_shape.size(); ++i) names.push_back("i" + std::to_string(i));
  return loop_nest(out_shape, names, [&](const std::vector<Ref<PrimExpr>>& idx) {
    return BufferStore::make(out, idx, map(BufferLoad::make(in, idx)));
  });
}

class ReluLoweringRule : public OpLoweringRule {
 public:
  std::string op_name() const override { return "nn.relu"; }

  Ref<Stmt> lower(const std::vector<Ref<Buffer>>& in_bufs,
                  Ref<Buffer> out_buf,
                  const Attrs& /*attrs*/) override {
    return elementwise_unary(in_bufs.at(0), out_buf, [](Ref<PrimExpr> v) {
      return Max::make(lower_utils::cf(0.0), v);
    });
  }
};

class ClipLoweringRule : public OpLoweringRule {
 public:
  std::string op_name() const override { return "nn.clip"; }

  Ref<Stmt> lower(const std::vector<Ref<Buffer>>& in_bufs,
                  Ref<Buffer> out_buf,
                  const Attrs& attrs) override {
    double lo = attrs.get_or<double>("a_min", 0.0);
    double hi = attrs.get_or<double>("a_max", 6.0);
    return elementwise_unary(in_bufs.at(0), out_buf, [lo, hi](Ref<PrimExpr> v) {
      return Max::make(lower_utils::cf(lo), Min::make(lower_utils::cf(hi), v));
    });
  }
};

class AddLoweringRule : public OpLoweringRule {
 public:
  std::string op_name() const override { return "nn.add"; }

  Ref<Stmt> lower(const std::vector<Ref<Buffer>>& in_bufs,
                  Ref<Buffer> out_buf,
                  const Attrs& /*attrs*/) override {
    auto out_shape = shape_of(out_buf);
    auto lhs_shape = shape_of(in_bufs.at(0));
    auto rhs_shape = shape_of(in_bufs.at(1));
    std::vector<std::string> names;
    for (size_t i = 0; i < out_shape.size(); ++i) names.push_back("i" + std::to_string(i));
    return loop_nest(out_shape, names, [&](const std::vector<Ref<PrimExpr>>& idx) {
      auto lhs_idx = broadcast_indices(idx, lhs_shape);
      auto rhs_idx = broadcast_indices(idx, rhs_shape);
      auto value = Add::make(BufferLoad::make(in_bufs[0], lhs_idx),
                             BufferLoad::make(in_bufs[1], rhs_idx));
      return BufferStore::make(out_buf, idx, value);
    });
  }
};

class DenseLoweringRule : public OpLoweringRule {
 public:
  std::string op_name() const override { return "nn.dense"; }

  Ref<Stmt> lower(const std::vector<Ref<Buffer>>& in_bufs,
                  Ref<Buffer> out_buf,
                  const Attrs& /*attrs*/) override {
    auto data_shape = shape_of(in_bufs.at(0));
    auto weight_shape = shape_of(in_bufs.at(1));
    int64_t n = data_shape.at(0);
    int64_t ic = data_shape.at(1);
    int64_t oc = weight_shape.at(0);
    auto vn = LLVar::make("n");
    auto voc = LLVar::make("oc");
    auto vic = LLVar::make("ic");
    Ref<PrimExpr> init = lower_utils::cf(0.0);
    if (in_bufs.size() > 2) init = BufferLoad::make(in_bufs[2], {voc});
    auto init_store = BufferStore::make(out_buf, {vn, voc}, init);
    auto update = BufferStore::make(
      out_buf, {vn, voc},
      Add::make(BufferLoad::make(out_buf, {vn, voc}),
                Mul::make(BufferLoad::make(in_bufs[0], {vn, vic}),
                          BufferLoad::make(in_bufs[1], {voc, vic}))));
    auto init_loop = For::make(vn, lower_utils::ci(0), lower_utils::ci(n),
      ForKind::kSerial, For::make(voc, lower_utils::ci(0), lower_utils::ci(oc),
      ForKind::kSerial, init_store));
    auto main_loop = For::make(vn, lower_utils::ci(0), lower_utils::ci(n),
      ForKind::kSerial, For::make(voc, lower_utils::ci(0), lower_utils::ci(oc),
      ForKind::kSerial, For::make(vic, lower_utils::ci(0), lower_utils::ci(ic),
      ForKind::kSerial, update)));
    return lower_utils::seq({init_loop, main_loop});
  }
};

class BatchNormLoweringRule : public OpLoweringRule {
 public:
  std::string op_name() const override { return "nn.batch_norm"; }

  Ref<Stmt> lower(const std::vector<Ref<Buffer>>& in_bufs,
                  Ref<Buffer> out_buf,
                  const Attrs& attrs) override {
    if (in_bufs.size() < 5) {
      return elementwise_unary(in_bufs.at(0), out_buf, [](Ref<PrimExpr> v) {
        return v;
      });
    }
    double eps = attrs.get_or<double>("epsilon", 1e-5);
    auto out_shape = shape_of(out_buf);
    std::vector<std::string> names;
    for (size_t i = 0; i < out_shape.size(); ++i) names.push_back("i" + std::to_string(i));
    return loop_nest(out_shape, names, [&](const std::vector<Ref<PrimExpr>>& idx) {
      auto c = idx.at(1);
      auto x = BufferLoad::make(in_bufs[0], idx);
      auto gamma = BufferLoad::make(in_bufs[1], {c});
      auto beta = BufferLoad::make(in_bufs[2], {c});
      auto mean = BufferLoad::make(in_bufs[3], {c});
      auto var = BufferLoad::make(in_bufs[4], {c});
      auto denom = PrimCall::make(out_buf->dtype, "sqrt", CallType::kBuiltin,
                                  {Add::make(var, lower_utils::cf(eps))});
      auto norm = Div::make(Sub::make(x, mean), denom);
      return BufferStore::make(out_buf, idx, Add::make(Mul::make(gamma, norm), beta));
    });
  }
};

class MaxPool2DLoweringRule : public OpLoweringRule {
 public:
  std::string op_name() const override { return "nn.max_pool2d"; }

  Ref<Stmt> lower(const std::vector<Ref<Buffer>>& in_bufs,
                  Ref<Buffer> out_buf,
                  const Attrs& attrs) override {
    auto data_shape = shape_of(in_bufs.at(0));
    auto out_shape = shape_of(out_buf);
    auto pool = attrs.get_or<std::vector<int64_t>>("pool_size", {1, 1});
    auto strides = attrs.get_or<std::vector<int64_t>>("strides", pool);
    auto padding = attrs.get_or<std::vector<int64_t>>("padding", {0, 0, 0, 0});
    auto vn = LLVar::make("n");
    auto vc = LLVar::make("c");
    auto voh = LLVar::make("oh");
    auto vow = LLVar::make("ow");
    auto vkh = LLVar::make("kh");
    auto vkw = LLVar::make("kw");
    auto init = BufferStore::make(out_buf, {vn, vc, voh, vow},
                                  lower_utils::cf(-3.402823e38));
    auto in_h = Add::make(Mul::make(voh, lower_utils::ci(strides[0])),
                          Sub::make(vkh, lower_utils::ci(padding[0])));
    auto in_w = Add::make(Mul::make(vow, lower_utils::ci(strides[1])),
                          Sub::make(vkw, lower_utils::ci(padding[1])));
    auto cond = lower_utils::call_bool("and", {
      lower_utils::call_bool("and", {
        lower_utils::call_bool("ge", {in_h, lower_utils::ci(0)}),
        lower_utils::call_bool("lt", {in_h, lower_utils::ci(data_shape[2])})}),
      lower_utils::call_bool("and", {
        lower_utils::call_bool("ge", {in_w, lower_utils::ci(0)}),
        lower_utils::call_bool("lt", {in_w, lower_utils::ci(data_shape[3])})})});
    auto update = BufferStore::make(
      out_buf, {vn, vc, voh, vow},
      Max::make(BufferLoad::make(out_buf, {vn, vc, voh, vow}),
                BufferLoad::make(in_bufs[0], {vn, vc, in_h, in_w})));
    auto init_loop = For::make(vn, lower_utils::ci(0), lower_utils::ci(out_shape[0]),
      ForKind::kSerial, For::make(vc, lower_utils::ci(0), lower_utils::ci(out_shape[1]),
      ForKind::kSerial, For::make(voh, lower_utils::ci(0), lower_utils::ci(out_shape[2]),
      ForKind::kSerial, For::make(vow, lower_utils::ci(0), lower_utils::ci(out_shape[3]),
      ForKind::kSerial, init))));
    auto main_loop = For::make(vn, lower_utils::ci(0), lower_utils::ci(out_shape[0]),
      ForKind::kSerial, For::make(vc, lower_utils::ci(0), lower_utils::ci(out_shape[1]),
      ForKind::kSerial, For::make(voh, lower_utils::ci(0), lower_utils::ci(out_shape[2]),
      ForKind::kSerial, For::make(vow, lower_utils::ci(0), lower_utils::ci(out_shape[3]),
      ForKind::kSerial, For::make(vkh, lower_utils::ci(0), lower_utils::ci(pool[0]),
      ForKind::kSerial, For::make(vkw, lower_utils::ci(0), lower_utils::ci(pool[1]),
      ForKind::kSerial, IfThenElse::make(cond, update)))))));
    return lower_utils::seq({init_loop, main_loop});
  }
};

class AvgPool2DLoweringRule : public OpLoweringRule {
 public:
  std::string op_name() const override { return "nn.avg_pool2d"; }

  Ref<Stmt> lower(const std::vector<Ref<Buffer>>& in_bufs,
                  Ref<Buffer> out_buf,
                  const Attrs& attrs) override {
    auto out_shape = shape_of(out_buf);
    auto pool = attrs.get_or<std::vector<int64_t>>("pool_size", {1, 1});
    auto strides = attrs.get_or<std::vector<int64_t>>("strides", pool);
    auto vn = LLVar::make("n");
    auto vc = LLVar::make("c");
    auto voh = LLVar::make("oh");
    auto vow = LLVar::make("ow");
    auto vkh = LLVar::make("kh");
    auto vkw = LLVar::make("kw");
    auto init = BufferStore::make(out_buf, {vn, vc, voh, vow}, lower_utils::cf(0.0));
    auto in_h = Add::make(Mul::make(voh, lower_utils::ci(strides[0])), vkh);
    auto in_w = Add::make(Mul::make(vow, lower_utils::ci(strides[1])), vkw);
    auto update = BufferStore::make(
      out_buf, {vn, vc, voh, vow},
      Add::make(BufferLoad::make(out_buf, {vn, vc, voh, vow}),
                Div::make(BufferLoad::make(in_bufs[0], {vn, vc, in_h, in_w}),
                          lower_utils::cf(static_cast<double>(pool[0] * pool[1])))));
    auto init_loop = For::make(vn, lower_utils::ci(0), lower_utils::ci(out_shape[0]),
      ForKind::kSerial, For::make(vc, lower_utils::ci(0), lower_utils::ci(out_shape[1]),
      ForKind::kSerial, For::make(voh, lower_utils::ci(0), lower_utils::ci(out_shape[2]),
      ForKind::kSerial, For::make(vow, lower_utils::ci(0), lower_utils::ci(out_shape[3]),
      ForKind::kSerial, init))));
    auto main_loop = For::make(vn, lower_utils::ci(0), lower_utils::ci(out_shape[0]),
      ForKind::kSerial, For::make(vc, lower_utils::ci(0), lower_utils::ci(out_shape[1]),
      ForKind::kSerial, For::make(voh, lower_utils::ci(0), lower_utils::ci(out_shape[2]),
      ForKind::kSerial, For::make(vow, lower_utils::ci(0), lower_utils::ci(out_shape[3]),
      ForKind::kSerial, For::make(vkh, lower_utils::ci(0), lower_utils::ci(pool[0]),
      ForKind::kSerial, For::make(vkw, lower_utils::ci(0), lower_utils::ci(pool[1]),
      ForKind::kSerial, update))))));
    return lower_utils::seq({init_loop, main_loop});
  }
};

class GlobalAvgPool2DLoweringRule : public AvgPool2DLoweringRule {
 public:
  std::string op_name() const override { return "nn.global_avg_pool2d"; }

  Ref<Stmt> lower(const std::vector<Ref<Buffer>>& in_bufs,
                  Ref<Buffer> out_buf,
                  const Attrs& /*attrs*/) override {
    auto data_shape = shape_of(in_bufs.at(0));
    Attrs attrs;
    attrs.values["pool_size"] = std::vector<int64_t>{data_shape[2], data_shape[3]};
    attrs.values["strides"] = std::vector<int64_t>{data_shape[2], data_shape[3]};
    return AvgPool2DLoweringRule::lower(in_bufs, out_buf, attrs);
  }
};

class ReshapeLikeLoweringRule : public OpLoweringRule {
 public:
  explicit ReshapeLikeLoweringRule(std::string name) : name_(std::move(name)) {}
  std::string op_name() const override { return name_; }

  Ref<Stmt> lower(const std::vector<Ref<Buffer>>& in_bufs,
                  Ref<Buffer> out_buf,
                  const Attrs& /*attrs*/) override {
    auto out_shape = shape_of(out_buf);
    int64_t numel = 1;
    for (int64_t dim : out_shape) numel *= dim;
    auto vi = LLVar::make("i");
    auto store = BufferStore::make(out_buf, {vi}, BufferLoad::make(in_bufs[0], {vi}));
    return For::make(vi, lower_utils::ci(0), lower_utils::ci(numel),
                     ForKind::kSerial, store);
  }

 private:
  std::string name_;
};

class Conv2DLoweringRule : public OpLoweringRule {
 public:
  std::string op_name() const override { return "nn.conv2d"; }

  Ref<Stmt> lower(const std::vector<Ref<Buffer>>& in_bufs,
                  Ref<Buffer> out_buf,
                  const Attrs& attrs) override {
    auto data_shape = shape_of(in_bufs.at(0));
    auto weight_shape = shape_of(in_bufs.at(1));
    auto out_shape = shape_of(out_buf);
    auto strides = attrs.get_or<std::vector<int64_t>>("strides", {1, 1});
    auto padding = attrs.get_or<std::vector<int64_t>>("padding", {0, 0, 0, 0});
    int64_t n = out_shape.at(0);
    int64_t oc = out_shape.at(1);
    int64_t oh = out_shape.at(2);
    int64_t ow = out_shape.at(3);
    int64_t ic = data_shape.at(1);
    int64_t ih = data_shape.at(2);
    int64_t iw = data_shape.at(3);
    int64_t kh = weight_shape.at(2);
    int64_t kw = weight_shape.at(3);

    auto vn = LLVar::make("n");
    auto voc = LLVar::make("oc");
    auto voh = LLVar::make("oh");
    auto vow = LLVar::make("ow");
    auto vic = LLVar::make("ic");
    auto vkh = LLVar::make("kh");
    auto vkw = LLVar::make("kw");
    Ref<PrimExpr> init = lower_utils::cf(0.0);
    if (in_bufs.size() > 2) init = BufferLoad::make(in_bufs[2], {voc});
    auto init_store = BufferStore::make(out_buf, {vn, voc, voh, vow}, init);
    auto in_h = Add::make(Mul::make(voh, lower_utils::ci(strides[0])),
                          Sub::make(vkh, lower_utils::ci(padding[0])));
    auto in_w = Add::make(Mul::make(vow, lower_utils::ci(strides[1])),
                          Sub::make(vkw, lower_utils::ci(padding[1])));
    auto cond = lower_utils::call_bool("and", {
      lower_utils::call_bool("and", {
        lower_utils::call_bool("ge", {in_h, lower_utils::ci(0)}),
        lower_utils::call_bool("lt", {in_h, lower_utils::ci(ih)})}),
      lower_utils::call_bool("and", {
        lower_utils::call_bool("ge", {in_w, lower_utils::ci(0)}),
        lower_utils::call_bool("lt", {in_w, lower_utils::ci(iw)})})});
    auto update = BufferStore::make(
      out_buf, {vn, voc, voh, vow},
      Add::make(BufferLoad::make(out_buf, {vn, voc, voh, vow}),
                Mul::make(BufferLoad::make(in_bufs[0], {vn, vic, in_h, in_w}),
                          BufferLoad::make(in_bufs[1], {voc, vic, vkh, vkw}))));
    auto guarded = IfThenElse::make(cond, update);
    auto init_loop = For::make(vn, lower_utils::ci(0), lower_utils::ci(n),
      ForKind::kSerial, For::make(voc, lower_utils::ci(0), lower_utils::ci(oc),
      ForKind::kSerial, For::make(voh, lower_utils::ci(0), lower_utils::ci(oh),
      ForKind::kSerial, For::make(vow, lower_utils::ci(0), lower_utils::ci(ow),
      ForKind::kSerial, init_store))));
    auto main_loop = For::make(vn, lower_utils::ci(0), lower_utils::ci(n),
      ForKind::kSerial, For::make(voc, lower_utils::ci(0), lower_utils::ci(oc),
      ForKind::kSerial, For::make(voh, lower_utils::ci(0), lower_utils::ci(oh),
      ForKind::kSerial, For::make(vow, lower_utils::ci(0), lower_utils::ci(ow),
      ForKind::kSerial, For::make(vic, lower_utils::ci(0), lower_utils::ci(ic),
      ForKind::kSerial, For::make(vkh, lower_utils::ci(0), lower_utils::ci(kh),
      ForKind::kSerial, For::make(vkw, lower_utils::ci(0), lower_utils::ci(kw),
      ForKind::kSerial, guarded)))))));
    return lower_utils::seq({init_loop, main_loop});
  }
};

} /* anonymous namespace */

std::unordered_map<std::string, std::shared_ptr<OpLoweringRule>>&
OpLoweringRegistry::registry() {
  static std::unordered_map<std::string, std::shared_ptr<OpLoweringRule>> reg;
  return reg;
}

void OpLoweringRegistry::register_rule(std::shared_ptr<OpLoweringRule> rule) {
  registry()[rule->op_name()] = std::move(rule);
}

OpLoweringRule* OpLoweringRegistry::get(const std::string& op_name) {
  auto it = registry().find(op_name);
  if (it == registry().end()) return nullptr;
  return it->second.get();
}

bool OpLoweringRegistry::has(const std::string& op_name) {
  return registry().count(op_name) > 0;
}

namespace lower_utils {

Ref<IntImm> ci(int64_t v) {
  return IntImm::make(v);
}

Ref<FloatImm> cf(double v) {
  return FloatImm::make(v);
}

Ref<BufferLoad> load(Ref<Buffer> buf,
                     std::initializer_list<Ref<PrimExpr>> indices) {
  return BufferLoad::make(std::move(buf), std::vector<Ref<PrimExpr>>(indices));
}

Ref<BufferStore> store(Ref<Buffer> buf,
                       std::initializer_list<Ref<PrimExpr>> indices,
                       Ref<PrimExpr> value) {
  return BufferStore::make(std::move(buf), std::vector<Ref<PrimExpr>>(indices),
                           std::move(value));
}

Ref<Stmt> seq(std::vector<Ref<Stmt>> stmts) {
  return SeqStmt::flatten(std::move(stmts));
}

Ref<PrimExpr> call_bool(const std::string& name,
                        std::vector<Ref<PrimExpr>> args) {
  return PrimCall::make(DataType::kBool, name, CallType::kBuiltin, std::move(args));
}

} /* namespace lower_utils */

REGISTER_LOWERING_RULE(ReluLoweringRule);
REGISTER_LOWERING_RULE(ClipLoweringRule);
REGISTER_LOWERING_RULE(AddLoweringRule);
REGISTER_LOWERING_RULE(DenseLoweringRule);
REGISTER_LOWERING_RULE(BatchNormLoweringRule);
REGISTER_LOWERING_RULE(MaxPool2DLoweringRule);
REGISTER_LOWERING_RULE(AvgPool2DLoweringRule);
REGISTER_LOWERING_RULE(GlobalAvgPool2DLoweringRule);
REGISTER_LOWERING_RULE(Conv2DLoweringRule);

static bool _reshape_like_rules_registered = []() {
  OpLoweringRegistry::register_rule(std::make_shared<ReshapeLikeLoweringRule>("nn.flatten"));
  OpLoweringRegistry::register_rule(std::make_shared<ReshapeLikeLoweringRule>("nn.reshape"));
  return true;
}();

} /* namespace rasp */
