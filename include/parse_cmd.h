#ifndef RASP_INCLUDE_PARSE_CMD_H_
#define RASP_INCLUDE_PARSE_CMD_H_

#include <fstream>

#include "utils/utils.h"

namespace rasp {

void print_help() {
  std::cout << R"(
  Usage:
  rasp_compiler <json_path> [options]

  Options:
  -d, --dump-ir
  -r, --dump-dir <dir_name>
  -l, --level <n>
  )";
}

static int parse_cmd(int argc, char **argv, InputArgument& args) {
  if (argc < 2 || (
      std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
    print_help();
    return kRpcCmdLineErr;
  }

  std::ifstream input(argv[1]);
  if (!input) {
    LOG_E(("Failed to open: " + std::string(argv[1])).c_str());
    return kRpcCmdLineErr;
  }
  args.input_ir_str_buf << input.rdbuf();

  for (int i = 2;i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "-d" || arg == "--dump-ir") {
      /* dump ir json */
      args.dump_ir = true;
    } else if (arg == "-r" || arg == "--dump-dir") {
      /* dump ir json dir */
      if (i + 1 >= argc) {
        LOG_E("Expected argument after " + arg);
        return kRpcCmdLineErr;
      }
      args.dump_dir = "./" + std::string(argv[++i]);
    } else if (arg == "-l" || arg == "--level") {
      /* optimization pass level */
      if (i + 1 >= argc) {
        LOG_E("Expected argument after " + arg);
        return kRpcCmdLineErr;
      }
      try {
        args.opt_level = std::stoi(argv[++i]);
      } catch (const std::exception& e) {
        LOG_E("Invalid optimization level: " + std::string(argv[i]));
        return kRpcCmdLineErr;
      }
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