#include "parse_cmd.h"
#include "pass/pass_pipeline.h"


int main(int argc, char **argv) {
  rasp::InputArgument arg;
  RPC_CHECK(rasp::parse_cmd(argc, argv, arg));

  auto mod = rasp::IRModule::from_json(arg.input_ir_str_buf.str());
  rasp::PassContext ctx;
  ctx.opt_level = arg.opt_level;
  ctx.dump_ir = arg.dump_ir;
  ctx.dump_dir = arg.dump_dir;

  rasp::PassManager hlir_pm = build_hlir_pass_pipeline(ctx.opt_level);
  hlir_pm.print_pipeline();
  auto optimized_mod = hlir_pm.run(mod, ctx);

  /* TODO chapter 7: lower HLIR -> LLIR */
  /* TODO chapter 8: codegen from LLIR */

  return rasp::kRpcSuccess;
}
