#ifndef RASP_INCLUDE_PARSE_CMD_H_
#define RASP_INCLUDE_PARSE_CMD_H_

#include <fstream>
#include <sstream>

#include "utils/utils.h"

namespace rasp {

inline void print_help() {
  std::cout << R"(
  Usage:
  rasp_compiler <json_path> [options]

  Options:
  -d, --dump-ir               Dump IR to JSON files
  -r, --dump-dir <dir>        Directory for IR dumps (default: ./ir_dump)
  -l, --level <n>             Optimisation level 0-2 (default: 2)
  -o, --output <path>         Output path (.cpp or .so, default: model_gen.cpp)
  -s, --input-shape <shape>   Comma-separated concrete input shape, e.g. 1,3,224,224
                              Required when the model has dynamic input dimensions
      --emit-so               Cross-compile generated .cpp to a .so shared library
      --cc <compiler>         Cross-compiler binary for --emit-so
                              (default: aarch64-linux-gnu-g++)
  )";
}

namespace detail {

inline std::vector<int64_t> parse_shape_str(const std::string& s) {
  std::vector<int64_t> shape;
  std::istringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    try {
      int64_t v = std::stoll(tok);
      if (v <= 0)
        throw std::runtime_error("shape values must be positive, got: " + tok);
      shape.push_back(v);
    } catch (const std::invalid_argument&) {
      throw std::runtime_error("invalid shape token: '" + tok + "'");
    }
  }
  if (shape.empty()) throw std::runtime_error("--input-shape is empty");
  return shape;
}

} /* namespace detail */

inline int parse_cmd(int argc, char** argv, InputArgument& args) {
  if (argc < 2 ||
      std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
    print_help();
    return kRpcCmdLineErr;
  }

  args.json_path = argv[1];
  {
    std::ifstream test(args.json_path);
    if (!test) {
      LOG_E(("Failed to open: " + args.json_path).c_str());
      return kRpcCmdLineErr;
    }
  }

  for (int i = 2; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "-d" || arg == "--dump-ir") {
      args.dump_ir = true;
    } else if (arg == "-r" || arg == "--dump-dir") {
      if (i + 1 >= argc) { LOG_E("Expected argument after " + arg); return kRpcCmdLineErr; }
      args.dump_dir = "./" + std::string(argv[++i]);
    } else if (arg == "-l" || arg == "--level") {
      if (i + 1 >= argc) { LOG_E("Expected argument after " + arg); return kRpcCmdLineErr; }
      try {
        args.opt_level = std::stoi(argv[++i]);
      } catch (const std::exception&) {
        LOG_E("Invalid optimization level: " + std::string(argv[i]));
        return kRpcCmdLineErr;
      }
    } else if (arg == "-o" || arg == "--output") {
      if (i + 1 >= argc) { LOG_E("Expected argument after " + arg); return kRpcCmdLineErr; }
      args.output_path = argv[++i];
    } else if (arg == "-s" || arg == "--input-shape") {
      if (i + 1 >= argc) { LOG_E("Expected argument after " + arg); return kRpcCmdLineErr; }
      try {
        args.input_shape = detail::parse_shape_str(argv[++i]);
      } catch (const std::exception& e) {
        LOG_E(("Invalid --input-shape: " + std::string(e.what())).c_str());
        return kRpcCmdLineErr;
      }
    } else if (arg == "--emit-so") {
      args.emit_so = true;
    } else if (arg == "--cc") {
      if (i + 1 >= argc) { LOG_E("Expected argument after " + arg); return kRpcCmdLineErr; }
      args.cc = argv[++i];
    } else {
      LOG_E("Unknown argument: " + arg);
      print_help();
      return kRpcCmdLineErr;
    }
  }

  return kRpcSuccess;
}

} /* namespace rasp */

#endif /* RASP_INCLUDE_PARSE_CMD_H_ */
