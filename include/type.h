#ifndef RASP_INCLUDE_TYPE_H_
#define RASP_INCLUDE_TYPE_H_

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "ir_node.h"

namespace rasp {

/* ──────────────────────────────────────────────────────────────────────────
 * DataType
 * Scalar element type of a tensor.
 * ────────────────────────────────────────────────────────────────────────── */
enum class DataType {
  kFloat32,
  kFloat16,
  kFloat64,
  kInt32,
  kInt64,
  kInt8,
  kUInt8,
  kBool,
};


inline std::string dtype_to_string(DataType dt) {
  switch (dt) {
    case DataType::kFloat32: return "float32";
    case DataType::kFloat16: return "float16";
    case DataType::kFloat64: return "float64";
    case DataType::kInt32:   return "int32";
    case DataType::kInt64:   return "int64";
    case DataType::kInt8:    return "int8";
    case DataType::kUInt8:   return "uint8";
    case DataType::kBool:    return "bool";
  }
  throw std::invalid_argument("Unknown DataType");
}


inline DataType dtype_from_string(const std::string& s) {
  if (s == "float32") return DataType::kFloat32;
  if (s == "float16") return DataType::kFloat16;
  if (s == "float64") return DataType::kFloat64;
  if (s == "int32")   return DataType::kInt32;
  if (s == "int64")   return DataType::kInt64;
  if (s == "int8")    return DataType::kInt8;
  if (s == "uint8")   return DataType::kUInt8;
  if (s == "bool")    return DataType::kBool;
  throw std::invalid_argument("Unknown dtype string: " + s);
}


inline int dtype_bytes(DataType dt) {
  switch (dt) {
    case DataType::kFloat32: return 4;
    case DataType::kFloat16: return 2;
    case DataType::kFloat64: return 8;
    case DataType::kInt32:   return 4;
    case DataType::kInt64:   return 8;
    case DataType::kInt8:    return 1;
    case DataType::kUInt8:   return 1;
    case DataType::kBool:    return 1;
  }
  throw std::invalid_argument("Unknown DataType");
}

/* ──────────────────────────────────────────────────────────────────────────
 * TensorType
 * Describes a statically-shaped tensor.
 * shape[i] == -1 denotes an unknown (dynamic) dimension.
 * ────────────────────────────────────────────────────────────────────────── */
class TensorType : public IRNode {
 public:
  std::vector<int64_t> shape;
  DataType dtype;

  IRNodeType node_type() const override { return IRNodeType::kTensorType; }

  static Ref<TensorType> make(std::vector<int64_t> shape, DataType dtype);

  int64_t ndim() const { return static_cast<int64_t>(shape.size()); }

  /* Returns the total number of elements, or -1 if any dimension is dynamic. */
  int64_t numel() const {
    int64_t n = 1;
    for (int64_t d : shape) {
      if (d < 0) return -1;
      n *= d;
    }
    return n;
  }

private:
  TensorType(std::vector<int64_t> shape, DataType dtype)
    : shape(std::move(shape)), dtype(dtype) {}
};

inline Ref<TensorType> TensorType::make(std::vector<int64_t> shape, DataType dtype) {
  return std::shared_ptr<TensorType>(new TensorType(std::move(shape), dtype));
}

/* ──────────────────────────────────────────────────────────────────────────
 * TupleType
 * Type of a tuple expression; each field is a TensorType or TupleType.
 * Used for multi-output operators.
 * ────────────────────────────────────────────────────────────────────────── */
class TupleType : public IRNode {
 public:
  std::vector<Ref<IRNode>> fields;

  IRNodeType node_type() const override { return IRNodeType::kTupleType; }

  static Ref<TupleType> make(std::vector<Ref<IRNode>> fields);

 private:
  explicit TupleType(std::vector<Ref<IRNode>> fields)
    : fields(std::move(fields)) {}
};

inline Ref<TupleType> TupleType::make(std::vector<Ref<IRNode>> fields) {
  return std::shared_ptr<TupleType>(new TupleType(std::move(fields)));
}

/* ──────────────────────────────────────────────────────────────────────────
 * FuncType
 * Type signature of a Function expression.
 * ────────────────────────────────────────────────────────────────────────── */
class FuncType : public IRNode {
 public:
  std::vector<Ref<IRNode>> arg_types;
  Ref<IRNode> ret_type;

  IRNodeType node_type() const override { return IRNodeType::kFuncType; }

  static Ref<FuncType> make(std::vector<Ref<IRNode>> arg_types, Ref<IRNode> ret_type);

 private:
  FuncType(std::vector<Ref<IRNode>> arg_types, Ref<IRNode> ret_type)
    : arg_types(std::move(arg_types)), ret_type(std::move(ret_type)) {}
};

inline Ref<FuncType> FuncType::make(std::vector<Ref<IRNode>> arg_types, Ref<IRNode> ret_type) {
  return std::shared_ptr<FuncType>(new FuncType(std::move(arg_types), std::move(ret_type)));
}

} /* namespace rasp */

#endif /* RASP_INCLUDE_TYPE_H_ */
