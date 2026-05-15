#include "driver/compiler_driver.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include <sys/wait.h>
#include <unistd.h>

#include "codegen/codegen.h"
#include "high_level_ir/hl_expr_visitor.h"
#include "high_level_ir/ir_module.h"
#include "lowering/lowering.h"
#include "pass/pass_pipeline.h"
#include "utils/log.h"

namespace rasp {

namespace {

/* Visitor that replaces -1 dims in every Var's type_annotation by var name.
 * This is defence-in-depth for models where from_json body Var objects are
 * separate allocations from func->params (pre-fix deserialisers). */
class VarTypeSpecializer : public ExprVisitor {
  const std::unordered_map<std::string, Ref<TensorType>>& new_types_;

 public:
  explicit VarTypeSpecializer(
      const std::unordered_map<std::string, Ref<TensorType>>& m)
      : new_types_(m) {}

  void visit_var(Ref<Var> node) override {
    auto it = new_types_.find(node->name);
    if (it != new_types_.end()) node->type_annotation = it->second;
  }
};

/* Replace dynamic (-1) dimensions in the main function's param types with
 * the concrete values from `shape`, and propagate to all body Var nodes. */
static void specialize_shapes(Ref<IRModule>& mod,
                               const std::vector<int64_t>& shape) {
  if (!mod->has_function("main")) return;
  auto func = mod->get_function("main");

  std::unordered_map<std::string, Ref<TensorType>> new_types;
  for (auto& param : func->params) {
    if (!param->type_annotation ||
        param->type_annotation->node_type() != IRNodeType::kTensorType)
      continue;
    auto ttype = std::static_pointer_cast<TensorType>(param->type_annotation);
    if (static_cast<size_t>(ttype->ndim()) != shape.size()) {
      throw std::runtime_error(
          "--input-shape rank " + std::to_string(shape.size()) +
          " does not match param '" + param->name + "' rank " +
          std::to_string(ttype->ndim()));
    }
    auto new_shape = ttype->shape;
    for (size_t i = 0; i < new_shape.size(); ++i) {
      if (new_shape[i] < 0) new_shape[i] = shape[i];
    }
    auto new_type = TensorType::make(new_shape, ttype->dtype);
    param->type_annotation = new_type;
    new_types[param->name] = new_type;
  }

  VarTypeSpecializer specializer(new_types);
  specializer.visit(func->body);
}

/* Spawn a process via execvp — no shell, no injection risk. */
static int run_subprocess(const std::vector<std::string>& args) {
  std::vector<const char*> argv;
  argv.reserve(args.size() + 1);
  for (auto& a : args) argv.push_back(a.c_str());
  argv.push_back(nullptr);

  pid_t pid = fork();
  if (pid < 0) throw std::runtime_error("CompilerDriver: fork() failed");
  if (pid == 0) {
    execvp(argv[0], const_cast<char* const*>(argv.data()));
    _exit(127);
  }
  int status = 0;
  waitpid(pid, &status, 0);
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

} /* anonymous namespace */

void CompilerDriver::generate_cpp(const std::string& json_path,
                                   const std::string& output_cpp,
                                   const CompilerOptions& opts) {
  std::ifstream ifs(json_path);
  if (!ifs) throw std::runtime_error("CompilerDriver: cannot open " + json_path);
  std::ostringstream buf;
  buf << ifs.rdbuf();

  auto mod = IRModule::from_json(buf.str());

  if (!opts.input_shape.empty()) specialize_shapes(mod, opts.input_shape);

  PassContext ctx;
  ctx.opt_level = opts.opt_level;
  ctx.dump_ir   = opts.dump_ir;
  ctx.dump_dir  = opts.dump_dir;

  PassManager hlir_pm = build_hlir_pass_pipeline(ctx.opt_level);
  hlir_pm.print_pipeline();
  auto optimized_mod = hlir_pm.run(mod, ctx);

  auto llir_mod = Lowering::lower(optimized_mod);
  LOG_I(("lowered prim funcs = " +
         std::to_string(llir_mod->exec_order.size())).c_str());

  PassManager llir_pm = build_llir_pass_pipeline(ctx.opt_level);
  auto optimized_llir = llir_pm.run_llir(llir_mod, ctx);

  Codegen::generate(optimized_llir, output_cpp);
  LOG_I(("generated cpp = " + output_cpp).c_str());
}

void CompilerDriver::compile(const std::string& json_path,
                              const std::string& output_so,
                              const CompilerOptions& opts) {
  std::filesystem::path so_path(output_so);
  std::string cpp_path =
      (so_path.parent_path() / so_path.stem()).string() + ".cpp";

  generate_cpp(json_path, cpp_path, opts);

  std::vector<std::string> args = {
      opts.cc,
      "-O3",
      "-march=armv8-a+simd",
      "-mfpu=neon-fp-armv8",
      "-ftree-vectorize",
      "-shared", "-fPIC",
      "-o", output_so,
      cpp_path,
  };

  std::string cmd_display = opts.cc + " ... -shared -fPIC -o " + output_so;
  LOG_I(("invoking: " + cmd_display).c_str());

  int ret = run_subprocess(args);
  if (ret != 0) {
    throw std::runtime_error(
        "CompilerDriver: cross-compiler exited with code " +
        std::to_string(ret));
  }
  LOG_I(("generated so = " + output_so).c_str());
}

} /* namespace rasp */
