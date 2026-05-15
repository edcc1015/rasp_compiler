#include "runtime/runtime.h"

#include <dlfcn.h>

namespace rasp {

Runtime::~Runtime() {
  if (lib_handle_) dlclose(lib_handle_);
}

bool Runtime::load(const std::string& lib_path) {
  lib_handle_ = dlopen(lib_path.c_str(), RTLD_NOW);
  if (!lib_handle_) {
    last_error_ = dlerror();
    return false;
  }
  run_func_ = reinterpret_cast<RunFunc>(dlsym(lib_handle_, "model_run"));
  num_inputs_func_ = reinterpret_cast<NumFunc>(dlsym(lib_handle_, "model_num_inputs"));
  num_outputs_func_ = reinterpret_cast<NumFunc>(dlsym(lib_handle_, "model_num_outputs"));
  input_numel_func_ = reinterpret_cast<NumelFunc>(dlsym(lib_handle_, "model_input_numel"));
  output_numel_func_ = reinterpret_cast<NumelFunc>(dlsym(lib_handle_, "model_output_numel"));
  if (!run_func_ || !num_inputs_func_ || !num_outputs_func_ ||
      !input_numel_func_ || !output_numel_func_) {
    last_error_ = "Runtime: generated library is missing required symbols";
    dlclose(lib_handle_);
    lib_handle_ = nullptr;
    return false;
  }
  return true;
}

bool Runtime::run(const std::vector<Tensor>& inputs,
                  std::vector<Tensor>& outputs) {
  if (!run_func_) {
    last_error_ = "Runtime: model is not loaded";
    return false;
  }
  if (static_cast<int>(inputs.size()) != num_inputs()) {
    last_error_ = "Runtime: wrong input count";
    return false;
  }
  std::vector<const float*> in_ptrs;
  in_ptrs.reserve(inputs.size());
  for (const auto& input : inputs) in_ptrs.push_back(input.data_f32());

  outputs.clear();
  std::vector<float*> out_ptrs;
  int nout = num_outputs();
  for (int i = 0; i < nout; ++i) {
    outputs.push_back(Tensor::allocate({output_numel(i)}, DType::kFloat32));
    out_ptrs.push_back(outputs.back().data_f32());
  }
  run_func_(in_ptrs.data(), out_ptrs.data());
  return true;
}

int Runtime::num_inputs() const {
  return num_inputs_func_ ? num_inputs_func_() : 0;
}

int Runtime::num_outputs() const {
  return num_outputs_func_ ? num_outputs_func_() : 0;
}

int64_t Runtime::input_numel(int index) const {
  return input_numel_func_ ? input_numel_func_(index) : 0;
}

int64_t Runtime::output_numel(int index) const {
  return output_numel_func_ ? output_numel_func_(index) : 0;
}

} /* namespace rasp */
