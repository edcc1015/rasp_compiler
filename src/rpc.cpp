#include "driver/compiler_driver.h"
#include "parse_cmd.h"
#include "utils/log.h"

int main(int argc, char** argv) {
  rasp::InputArgument arg;
  RPC_CHECK(rasp::parse_cmd(argc, argv, arg));

  try {
    rasp::CompilerOptions opts;
    opts.opt_level   = arg.opt_level;
    opts.dump_ir     = arg.dump_ir;
    opts.dump_dir    = arg.dump_dir;
    opts.cc          = arg.cc;
    opts.input_shape = arg.input_shape;

    if (arg.emit_so) {
      rasp::CompilerDriver::compile(arg.json_path, arg.output_path, opts);
    } else {
      rasp::CompilerDriver::generate_cpp(arg.json_path, arg.output_path, opts);
    }
  } catch (const std::exception& e) {
    LOG_E(e.what());
    return rasp::kRpcError;
  }

  return rasp::kRpcSuccess;
}
