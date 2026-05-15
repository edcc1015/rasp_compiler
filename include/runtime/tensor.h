#ifndef RASP_INCLUDE_RUNTIME_TENSOR_H_
#define RASP_INCLUDE_RUNTIME_TENSOR_H_

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rasp {

enum class DType {
  kFloat32,
  kInt32,
  kInt64,
};

struct Tensor {
  void* data = nullptr;
  std::vector<int64_t> shape;
  DType dtype = DType::kFloat32;
  bool owned = false;

  Tensor() = default;
  ~Tensor();
  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;
  Tensor(Tensor&& other) noexcept;
  Tensor& operator=(Tensor&& other) noexcept;

  static Tensor allocate(const std::vector<int64_t>& shape, DType dtype);
  static Tensor from_data(void* data,
                          const std::vector<int64_t>& shape,
                          DType dtype);

  int64_t numel() const;
  size_t nbytes() const;
  float* data_f32() const { return static_cast<float*>(data); }
};

} /* namespace rasp */

#endif /* RASP_INCLUDE_RUNTIME_TENSOR_H_ */
