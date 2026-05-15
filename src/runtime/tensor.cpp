#include "runtime/tensor.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace rasp {

namespace {

size_t dtype_size(DType dtype) {
  switch (dtype) {
    case DType::kFloat32: return 4;
    case DType::kInt32: return 4;
    case DType::kInt64: return 8;
  }
  throw std::runtime_error("Tensor: unknown dtype");
}

size_t aligned_size(size_t bytes) {
  return ((bytes + 15) / 16) * 16;
}

} /* anonymous namespace */

Tensor::~Tensor() {
  if (owned && data) std::free(data);
}

Tensor::Tensor(Tensor&& other) noexcept {
  data = other.data;
  shape = std::move(other.shape);
  dtype = other.dtype;
  owned = other.owned;
  other.data = nullptr;
  other.owned = false;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
  if (this != &other) {
    if (owned && data) std::free(data);
    data = other.data;
    shape = std::move(other.shape);
    dtype = other.dtype;
    owned = other.owned;
    other.data = nullptr;
    other.owned = false;
  }
  return *this;
}

Tensor Tensor::allocate(const std::vector<int64_t>& shape, DType dtype) {
  Tensor t;
  t.shape = shape;
  t.dtype = dtype;
  t.owned = true;
  size_t bytes = aligned_size(t.nbytes());
  t.data = std::aligned_alloc(16, bytes);
  if (!t.data) throw std::bad_alloc();
  std::memset(t.data, 0, bytes);
  return t;
}

Tensor Tensor::from_data(void* data,
                         const std::vector<int64_t>& shape,
                         DType dtype) {
  Tensor t;
  t.data = data;
  t.shape = shape;
  t.dtype = dtype;
  t.owned = false;
  return t;
}

int64_t Tensor::numel() const {
  int64_t n = 1;
  for (int64_t dim : shape) n *= dim;
  return n;
}

size_t Tensor::nbytes() const {
  return static_cast<size_t>(numel()) * dtype_size(dtype);
}

} /* namespace rasp */
