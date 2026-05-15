#include <iostream>
#include <string>

#include "high_level_ir/hl_ir.h"
#include "lowering/lowering.h"
#include "pass/opt/constant_folding.h"
#include "pass/opt/operator_fusion.h"
#include "pass/analysis/type_inference.h"
#include "pass/pass_manager.h"

static bool check(const char* name, bool cond) {
  std::cout << (cond ? "[PASS] " : "[FAIL] ") << name << "\n";
  return cond;
}

static rasp::Ref<rasp::IRModule> infer(rasp::Ref<rasp::IRModule> mod) {
  rasp::PassContext ctx;
  rasp::PassManager pm;
  pm.add_pass(std::make_shared<rasp::TypeInference>());
  return pm.run(mod, ctx);
}

static bool test_lowering_topological_order() {
  auto x_t = rasp::TensorType::make({1, 4}, rasp::DataType::kFloat32);
  auto b_t = rasp::TensorType::make({1, 4}, rasp::DataType::kFloat32);
  auto x = rasp::Var::make("x", x_t);
  auto b = rasp::Var::make("b", b_t);
  auto add = rasp::Call::make(rasp::OpRegistry::get("nn.add"), {x, b});
  auto relu = rasp::Call::make(rasp::OpRegistry::get("nn.relu"), {add});
  auto mod = rasp::IRModule::make();
  mod->add_function("main", rasp::Function::make({x, b}, relu));

  auto llir = rasp::Lowering::lower(infer(mod));
  bool order_ok = llir->exec_order.size() == 2 &&
                  llir->get_func(llir->exec_order[0])->attrs["op"] == "nn.add" &&
                  llir->get_func(llir->exec_order[1])->attrs["op"] == "nn.relu";
  bool output_ok = llir->get_func(llir->exec_order[1])->params.back()->name == "output";
  return check("test_lowering_topological_order", order_ok && output_ok);
}

static bool test_lowering_conv2d_body() {
  auto data_t = rasp::TensorType::make({1, 3, 8, 8}, rasp::DataType::kFloat32);
  auto weight_t = rasp::TensorType::make({4, 3, 3, 3}, rasp::DataType::kFloat32);
  auto data = rasp::Var::make("data", data_t);
  auto weight = rasp::Var::make("weight", weight_t);
  rasp::Attrs attrs;
  attrs.values["strides"] = std::vector<int64_t>{1, 1};
  attrs.values["padding"] = std::vector<int64_t>{1, 1, 1, 1};
  auto conv = rasp::Call::make(rasp::OpRegistry::get("nn.conv2d"),
                               {data, weight}, attrs);
  auto mod = rasp::IRModule::make();
  mod->add_function("main", rasp::Function::make({data, weight}, conv));

  auto llir = rasp::Lowering::lower(infer(mod));
  auto fn = llir->get_func(llir->exec_order[0]);
  bool ok = fn->attrs["op"] == "nn.conv2d" &&
            fn->body->node_type() == rasp::IRNodeType::kSeqStmt &&
            fn->params.back()->shape[1]->node_type() == rasp::IRNodeType::kIntImm;
  return check("test_lowering_conv2d_body", ok);
}

static bool test_lowering_composite_add_relu() {
  auto x_t = rasp::TensorType::make({1, 4}, rasp::DataType::kFloat32);
  auto b_t = rasp::TensorType::make({1, 4}, rasp::DataType::kFloat32);
  auto x = rasp::Var::make("x", x_t);
  auto b = rasp::Var::make("b", b_t);
  auto add = rasp::Call::make(rasp::OpRegistry::get("nn.add"), {x, b});
  auto relu = rasp::Call::make(rasp::OpRegistry::get("nn.relu"), {add});
  auto mod = rasp::IRModule::make();
  mod->add_function("main", rasp::Function::make({x, b}, relu));

  rasp::PassContext ctx;
  ctx.opt_level = 2;
  rasp::PassManager pm;
  pm.add_pass(std::make_shared<rasp::TypeInference>());
  pm.add_pass(std::make_shared<rasp::ConstantFolding>());
  pm.add_pass(std::make_shared<rasp::OperatorFusion>());
  pm.add_pass(std::make_shared<rasp::TypeInference>());
  auto llir = rasp::Lowering::lower(pm.run(mod, ctx));

  auto fn = llir->get_func(llir->exec_order[0]);
  bool ok = llir->exec_order.size() == 1 &&
            fn->attrs["op"] == "add_relu" &&
            fn->body->node_type() == rasp::IRNodeType::kSeqStmt;
  return check("test_lowering_composite_add_relu", ok);
}

int main() {
  bool ok = true;
  ok &= test_lowering_topological_order();
  ok &= test_lowering_conv2d_body();
  ok &= test_lowering_composite_add_relu();
  return ok ? 0 : 1;
}
