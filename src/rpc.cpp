#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "high_level_ir/hl_ir.h"
#include "pass/pass_manager.h"
#include "utils/utils.h"

/* Build HLIR optimization pipeline based on opt_level.
 * TODO chapter 6: add passes (ConstantFolding, OperatorFusion, ...). */
static rasp::PassManager build_hlir_pass_pipeline(int opt_level) {
  rasp::PassManager pm;
  (void)opt_level;
  return pm;
}

/* Build LLIR optimization pipeline based on opt_level.
 * TODO chapter 6: add LLIR passes (LoopTiling, ...). */
[[maybe_unused]] static rasp::PassManager build_llir_pass_pipeline(int opt_level) {
  rasp::PassManager pm;
  (void)opt_level;
  return pm;
}

int main(int argc, char* argv[]) {
  /* parse args */
  if (argc != 2) {
    LOG_E("Usage: rasp_compiler <ir_module.json>");
    return rasp::kRpcError;
  }
  std::string json_path = argv[1];
  std::ifstream input(json_path);
  if (!input) {
    LOG_E(("Failed to open JSON file: " + json_path).c_str());
    return rasp::kRpcError;
  }
  std::ostringstream input_json_str;
  input_json_str << input.rdbuf();

  /* load IRModule from JSON */
  auto mod = rasp::IRModule::from_json(input_json_str.str());

  /* configure PassContext */
  rasp::PassContext ctx;
  ctx.opt_level = 2;
  ctx.dump_ir = false;

  /* run HLIR pass pipeline */
  rasp::PassManager hlir_pm = build_hlir_pass_pipeline(ctx.opt_level);
  auto optimized_mod = hlir_pm.run(mod, ctx);

  /* TODO chapter 7: lower HLIR -> LLIR */
  /* TODO chapter 6: run LLIR pass pipeline via build_llir_pass_pipeline */
  /* TODO chapter 8: codegen from LLIR */

  /* dump debug.json from optimized IRModule */
  std::string output_json_str = optimized_mod->to_json();
  std::ofstream output("debug.json");
  if (!output) {
    LOG_E("Failed to create output JSON file: debug.json");
    return rasp::kRpcError;
  }
  output << output_json_str;
  output << '\n';

  return rasp::kRpcSuccess;
}

