#ifndef RASP_INCLUDE_DRIVER_COMPILER_DRIVER_H_
#define RASP_INCLUDE_DRIVER_COMPILER_DRIVER_H_

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rasp {

struct CompilerOptions {
  int         opt_level = 2;
  /* Cross-compiler binary used by compile(). */
  std::string cc        = "aarch64-linux-gnu-g++";
  bool        dump_ir   = false;
  std::string dump_dir  = "./ir_dump";
  /* Concrete input dimensions for models with dynamic (-1) shapes. */
  std::vector<int64_t> input_shape;
};

class CompilerDriver {
 public:
  /* HLIR JSON → optimised LLIR → C++ source file. */
  static void generate_cpp(const std::string& json_path,
                            const std::string& output_cpp,
                            const CompilerOptions& opts = {});

  /* Same as generate_cpp, then invokes opts.cc to cross-compile to a .so.
   * The intermediate .cpp is written with the same stem as output_so. */
  static void compile(const std::string& json_path,
                      const std::string& output_so,
                      const CompilerOptions& opts = {});
};

} /* namespace rasp */

#endif /* RASP_INCLUDE_DRIVER_COMPILER_DRIVER_H_ */
