#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "codegen/codegen.h"
#include "high_level_ir/hl_ir.h"
#include "lowering/lowering.h"
#include "pass/analysis/type_inference.h"
#include "pass/pass_manager.h"
#include "runtime/reference_runtime.h"
#include "runtime/runtime.h"

static bool check(const char* name, bool cond) {
  std::cout << (cond ? "[PASS] " : "[FAIL] ") << name << "\n";
  return cond;
}

static rasp::Ref<rasp::Constant> make_const(const std::vector<float>& vals,
                                            const std::vector<int64_t>& shape) {
  std::vector<uint8_t> data(vals.size() * sizeof(float));
  std::memcpy(data.data(), vals.data(), data.size());
  return rasp::Constant::make(std::move(data),
                              rasp::TensorType::make(shape, rasp::DataType::kFloat32));
}

static bool test_codegen_runtime_compare() {
  auto x_t = rasp::TensorType::make({1, 4}, rasp::DataType::kFloat32);
  auto x = rasp::Var::make("x", x_t);
  auto w = make_const({1.0f, 2.0f, -1.0f, 0.5f,
                       -0.5f, 1.0f, 2.0f, 1.0f}, {2, 4});
  auto b = make_const({0.25f, -1.0f}, {2});
  auto dense = rasp::Call::make(rasp::OpRegistry::get("nn.dense"), {x, w, b});
  auto relu = rasp::Call::make(rasp::OpRegistry::get("nn.relu"), {dense});
  auto mod = rasp::IRModule::make();
  mod->add_function("main", rasp::Function::make({x}, relu));

  rasp::PassContext ctx;
  rasp::PassManager pm;
  pm.add_pass(std::make_shared<rasp::TypeInference>());
  auto typed = pm.run(mod, ctx);

  std::vector<float> in_data = {1.0f, -2.0f, 3.0f, 0.5f};
  std::vector<rasp::Tensor> inputs;
  inputs.push_back(rasp::Tensor::from_data(in_data.data(), {1, 4}, rasp::DType::kFloat32));
  auto expected = rasp::ReferenceRuntime::run(typed, inputs);

  auto llir = rasp::Lowering::lower(typed);
  std::filesystem::create_directories("codegen_runtime_tmp");
  std::string cpp = "codegen_runtime_tmp/model_gen.cpp";
  std::string so = "codegen_runtime_tmp/model_gen.so";
  rasp::Codegen::generate(llir, cpp);
  std::string cmd = "g++ -std=c++17 -O2 -shared -fPIC " + cpp + " -o " + so;
  int ret = std::system(cmd.c_str());
  if (ret != 0) return check("test_codegen_runtime_compile", false);

  rasp::Runtime rt;
  if (!rt.load("./" + so)) {
    std::cerr << rt.last_error() << "\n";
    return check("test_codegen_runtime_load", false);
  }
  std::vector<rasp::Tensor> outputs;
  if (!rt.run(inputs, outputs)) {
    std::cerr << rt.last_error() << "\n";
    return check("test_codegen_runtime_run", false);
  }

  return check("test_codegen_runtime_compare",
               outputs.size() == expected.size() &&
               rasp::allclose(outputs[0], expected[0]));
}

int main() {
  bool ok = true;
  ok &= test_codegen_runtime_compare();
  return ok ? 0 : 1;
}
