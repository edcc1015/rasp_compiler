#include "parse_cmd.h"
#include "lowering/lowering.h"
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

  auto llir_mod = rasp::Lowering::lower(optimized_mod);
  LOG_I(("lowered prim funcs = " + std::to_string(llir_mod->exec_order.size())).c_str());
  rasp::PassManager llir_pm = build_llir_pass_pipeline(ctx.opt_level);
  auto optimized_llir = llir_pm.run_llir(llir_mod, ctx);
  (void)optimized_llir;

  return rasp::kRpcSuccess;
}
