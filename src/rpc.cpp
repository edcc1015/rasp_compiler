#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "high_level_ir/hl_ir.h"
#include "utils/utils.h"

int main(int argc, char* argv[]) {
  /* parser arg */
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


  /* main logic */
  auto mod = rasp::IRModule::from_json(input_json_str.str());
  auto fn  = mod->get_function("main");
  std::string output_json_str = mod->to_json();

  /* dump debug.json from IRModule */
  std::ofstream output("debug.json");
  if (!output) {
    LOG_E("Failed to create output JSON file: debug.json");
    return rasp::kRpcError;
  }
  output << output_json_str;
  output << '\n';

  return rasp::kRpcSuccess;
}
