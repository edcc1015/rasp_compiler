#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "high_level_ir/hl_ir.h"

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <ir_module.json>\n";
    return 1;
  }

  std::string json_path = argv[1];
  std::ifstream input(json_path);
  if (!input) {
    std::cerr << "Failed to open JSON file: " << json_path << "\n";
    return 1;
  }

  std::ostringstream buffer;
  buffer << input.rdbuf();

  auto mod = rasp::IRModule::from_json(buffer.str());
  auto fn  = mod->get_function("main");

  std::string json_str = mod->to_json();
  (void)fn;

  std::ofstream output("debug.json");
  if (!output) {
    std::cerr << "Failed to create output JSON file: debug.json\n";
    return 1;
  }

  output << json_str;
  output << '\n';

  return 0;
}
