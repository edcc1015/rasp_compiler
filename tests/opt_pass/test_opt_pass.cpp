#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "high_level_ir/hl_ir.h"
#include "pass/opt/constant_folding.h"
#include "pass/opt/operator_fusion.h"
#include "pass/pass_manager.h"
#include "utils/utils.h"

/* ── helpers ─────────────────────────────────────────────────────────────── */

static rasp::Ref<rasp::Constant> make_float32_constant(
    const std::vector<float>& vals,
    const std::vector<int64_t>& shape) {
  std::vector<uint8_t> data(vals.size() * sizeof(float));
  std::memcpy(data.data(), vals.data(), data.size());
  auto ttype = rasp::TensorType::make(shape, rasp::DataType::kFloat32);
  return rasp::Constant::make(std::move(data), std::move(ttype));
}

static bool check(const char* name, bool cond) {
  LOG_I(cond ? (std::string(name) + " PASS").c_str()
             : (std::string(name) + " FAIL").c_str());
  return cond;
}

/* ── tests ───────────────────────────────────────────────────────────────── */

/*
 * Test 1 – ConstantFolding
 * Graph : fn(x) { add(x, relu(Constant([-1, 2, -3]))) }
 * Expect: relu(Constant) folded to Constant([0, 2, 0]); no "nn.relu" in JSON.
 */
static bool test_constant_folding() {
  auto c    = make_float32_constant({-1.0f, 2.0f, -3.0f}, {3});
  auto x    = rasp::Var::make("x");
  auto relu = rasp::Call::make(rasp::OpRegistry::get("nn.relu"), {c});
  auto add  = rasp::Call::make(rasp::OpRegistry::get("nn.add"), {x, relu});
  auto mod  = rasp::IRModule::make();
  mod->add_function("main", rasp::Function::make({x}, add));

  rasp::PassContext ctx;
  ctx.opt_level = 1;
  rasp::PassManager pm;
  pm.add_pass(std::make_shared<rasp::ConstantFolding>());
  std::string out = pm.run(mod, ctx)->to_json();

  return check("test_constant_folding",
               out.find("\"nn.relu\"") == std::string::npos);
}

/*
 * Test 2 – OperatorFusion: add_relu pattern
 * Graph : fn(data, bias) { relu(add(data, bias)) }
 * Expect: fused into Composite="add_relu".
 */
static bool test_fusion_add_relu() {
  auto data = rasp::Var::make("data");
  auto bias = rasp::Var::make("bias");
  auto add  = rasp::Call::make(rasp::OpRegistry::get("nn.add"), {data, bias});
  auto relu = rasp::Call::make(rasp::OpRegistry::get("nn.relu"), {add});
  auto mod  = rasp::IRModule::make();
  mod->add_function("main", rasp::Function::make({data, bias}, relu));

  rasp::PassContext ctx;
  ctx.opt_level = 2;
  rasp::PassManager pm;
  pm.add_pass(std::make_shared<rasp::ConstantFolding>());
  pm.add_pass(std::make_shared<rasp::OperatorFusion>());
  std::string out = pm.run(mod, ctx)->to_json();

  return check("test_fusion_add_relu",
               out.find("\"Composite\"") != std::string::npos);
}

/*
 * Test 3 – OperatorFusion: conv2d_add_relu pattern
 * Graph : fn(data, weight, bias) { relu(add(conv2d(data, weight), bias)) }
 * Expect: fused into Composite="conv2d_add_relu".
 */
static bool test_fusion_conv2d_add_relu() {
  auto data   = rasp::Var::make("data");
  auto weight = rasp::Var::make("weight");
  auto bias   = rasp::Var::make("bias");

  rasp::Attrs attrs;
  attrs.values["strides"] = std::vector<int64_t>{1, 1};
  attrs.values["padding"] = std::vector<int64_t>{0, 0, 0, 0};

  auto conv = rasp::Call::make(rasp::OpRegistry::get("nn.conv2d"),
                                {data, weight}, attrs);
  auto add  = rasp::Call::make(rasp::OpRegistry::get("nn.add"), {conv, bias});
  auto relu = rasp::Call::make(rasp::OpRegistry::get("nn.relu"), {add});
  auto mod  = rasp::IRModule::make();
  mod->add_function("main", rasp::Function::make({data, weight, bias}, relu));

  rasp::PassContext ctx;
  ctx.opt_level = 2;
  rasp::PassManager pm;
  pm.add_pass(std::make_shared<rasp::ConstantFolding>());
  pm.add_pass(std::make_shared<rasp::OperatorFusion>());
  std::string out = pm.run(mod, ctx)->to_json();

  return check("test_fusion_conv2d_add_relu",
               out.find("\"conv2d_add_relu\"") != std::string::npos);
}

/* ── main ────────────────────────────────────────────────────────────────── */

int main() {
  bool ok = true;
  ok &= test_constant_folding();
  ok &= test_fusion_add_relu();
  ok &= test_fusion_conv2d_add_relu();
  return ok ? 0 : 1;
}
