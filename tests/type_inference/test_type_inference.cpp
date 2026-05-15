#include <iostream>
#include <string>

#include "high_level_ir/hl_ir.h"
#include "pass/analysis/type_inference.h"
#include "pass/pass_manager.h"

static bool check(const char* name, bool cond) {
  std::cout << (cond ? "[PASS] " : "[FAIL] ") << name << "\n";
  return cond;
}

static rasp::Ref<rasp::TensorType> tensor_type(const rasp::Ref<rasp::Expr>& expr) {
  return std::static_pointer_cast<rasp::TensorType>(expr->checked_type);
}

static bool test_conv2d_shape() {
  auto data_t = rasp::TensorType::make({1, 3, 32, 32}, rasp::DataType::kFloat32);
  auto weight_t = rasp::TensorType::make({16, 3, 3, 3}, rasp::DataType::kFloat32);
  auto data = rasp::Var::make("data", data_t);
  auto weight = rasp::Var::make("weight", weight_t);
  rasp::Attrs attrs;
  attrs.values["strides"] = std::vector<int64_t>{2, 2};
  attrs.values["padding"] = std::vector<int64_t>{1, 1, 1, 1};
  auto conv = rasp::Call::make(rasp::OpRegistry::get("nn.conv2d"),
                               {data, weight}, attrs);
  auto mod = rasp::IRModule::make();
  mod->add_function("main", rasp::Function::make({data, weight}, conv));

  rasp::PassContext ctx;
  rasp::PassManager pm;
  pm.add_pass(std::make_shared<rasp::TypeInference>());
  auto out = pm.run(mod, ctx)->get_function("main")->body;
  auto tt = tensor_type(out);
  return check("test_conv2d_shape",
               tt->shape == std::vector<int64_t>({1, 16, 16, 16}));
}

static bool test_add_broadcast_shape() {
  auto x_t = rasp::TensorType::make({1, 16, 8, 8}, rasp::DataType::kFloat32);
  auto b_t = rasp::TensorType::make({16, 1, 1}, rasp::DataType::kFloat32);
  auto x = rasp::Var::make("x", x_t);
  auto b = rasp::Var::make("b", b_t);
  auto add = rasp::Call::make(rasp::OpRegistry::get("nn.add"), {x, b});
  auto mod = rasp::IRModule::make();
  mod->add_function("main", rasp::Function::make({x, b}, add));

  rasp::PassContext ctx;
  rasp::PassManager pm;
  pm.add_pass(std::make_shared<rasp::TypeInference>());
  auto out = pm.run(mod, ctx)->get_function("main")->body;
  auto tt = tensor_type(out);
  return check("test_add_broadcast_shape",
               tt->shape == std::vector<int64_t>({1, 16, 8, 8}));
}

int main() {
  bool ok = true;
  ok &= test_conv2d_shape();
  ok &= test_add_broadcast_shape();
  return ok ? 0 : 1;
}
