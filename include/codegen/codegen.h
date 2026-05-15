#ifndef RASP_INCLUDE_CODEGEN_CODEGEN_H_
#define RASP_INCLUDE_CODEGEN_CODEGEN_H_

#pragma once

#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "low_level_ir/ll_ir.h"

namespace rasp {

class CodegenContext {
 public:
  std::ostringstream code;
  int indent_level = 0;

  void emit(const std::string& line);
  void indent() { indent_level++; }
  void dedent() { indent_level--; }
  std::string get_indent() const;
};

class Codegen {
 public:
  static void generate(Ref<LLIRModule> mod, const std::string& output_path);

 private:
  CodegenContext ctx_;
  std::unordered_map<std::string, Ref<Buffer>> buffers_;
  std::vector<Ref<Buffer>> input_buffers_;
  std::vector<Ref<Buffer>> output_buffers_;
  std::vector<Ref<Buffer>> const_buffers_;
  std::vector<Ref<Buffer>> local_buffers_;

  void collect_buffers(Ref<LLIRModule> mod);
  void emit_header();
  void emit_constants();
  void emit_prim_func(Ref<PrimFunc> func);
  void emit_stmt(Ref<Stmt> stmt);
  std::string emit_expr(Ref<PrimExpr> expr);
  std::string emit_buffer_param(Ref<Buffer> buf);
  std::string linearize_indices(Ref<Buffer> buf,
                                const std::vector<Ref<PrimExpr>>& indices);
  void emit_model_run(Ref<LLIRModule> mod);
  void emit_meta();
  int64_t buffer_numel(Ref<Buffer> buf) const;
  std::string ctype(DataType dtype) const;
};

} /* namespace rasp */

#endif /* RASP_INCLUDE_CODEGEN_CODEGEN_H_ */
