#ifndef RASP_INCLUDE_RUNTIME_RUNTIME_H_
#define RASP_INCLUDE_RUNTIME_RUNTIME_H_

#pragma once

#include <string>
#include <vector>

#include "runtime/tensor.h"

namespace rasp {

class Runtime {
 public:
  Runtime() = default;
  ~Runtime();

  bool load(const std::string& lib_path);
  bool run(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs);
  std::string last_error() const { return last_error_; }
  int num_inputs() const;
  int num_outputs() const;
  int64_t input_numel(int index) const;
  int64_t output_numel(int index) const;

 private:
  using RunFunc = void (*)(const float** inputs, float** outputs);
  using NumFunc = int (*)();
  using NumelFunc = int64_t (*)(int index);

  void* lib_handle_ = nullptr;
  RunFunc run_func_ = nullptr;
  NumFunc num_inputs_func_ = nullptr;
  NumFunc num_outputs_func_ = nullptr;
  NumelFunc input_numel_func_ = nullptr;
  NumelFunc output_numel_func_ = nullptr;
  std::string last_error_;
};

} /* namespace rasp */

#endif /* RASP_INCLUDE_RUNTIME_RUNTIME_H_ */
