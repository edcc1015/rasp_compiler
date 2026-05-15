#ifndef RASP_INCLUDE_LOW_LEVEL_IR_BUFFER_H_
#define RASP_INCLUDE_LOW_LEVEL_IR_BUFFER_H_

#pragma once

#include <string>
#include <cstdint>
#include <vector>

#include "low_level_ir/prim_expr.h"

namespace rasp {

/* ──────────────────────────────────────────────────────────────────────────
 * BufferScope — indicates where a Buffer lives in the memory hierarchy.
 * ────────────────────────────────────────────────────────────────────────── */
enum class BufferScope {
  kGlobal, /* model inputs / outputs / weights — supplied by the caller */
  kLocal,  /* intra-function temporaries — allocated by an Allocate stmt */
};

/* ──────────────────────────────────────────────────────────────────────────
 * Buffer
 * Describes a multi-dimensional memory region without owning its data.
 * The runtime allocates actual memory; Buffer is a pure descriptor used
 * by BufferLoad, BufferStore, and Allocate nodes.
 *
 * Layout:
 *   strides empty  → row-major (C-order) contiguous storage
 *   strides set    → explicit stride for each dimension (e.g. NCHWc)
 *
 * data_alignment = 16 by default (NEON requires 128-bit / 16-byte alignment).
 * ────────────────────────────────────────────────────────────────────────── */
class Buffer : public IRNode {
 public:
  std::string name;
  DataType dtype;
  std::vector<Ref<PrimExpr>> shape;   /* one PrimExpr per dimension */
  std::vector<Ref<PrimExpr>> strides; /* empty → row-major contiguous */
  int data_alignment;
  BufferScope scope;
  std::vector<uint8_t> data;

  IRNodeType node_type() const override { return IRNodeType::kBuffer; }

  static Ref<Buffer> make(
    std::string name,
    DataType dtype,
    std::vector<Ref<PrimExpr>> shape,
    std::vector<Ref<PrimExpr>> strides = {},
    int data_alignment = 16,
    BufferScope scope = BufferScope::kGlobal);

  /* Compute the flat linear byte-offset for the given multi-dimensional
   * indices.  Row-major strides are derived from shape when strides is empty.
   * Returns IntImm(0) for a zero-dimensional buffer.
   * offset = indices[0]*strides[0] + ... + indices[N-1]*strides[N-1]
   * */
  Ref<PrimExpr> linearize(const std::vector<Ref<PrimExpr>>& indices) const;

 private:
  Buffer(std::string name,
         DataType dtype,
         std::vector<Ref<PrimExpr>> shape,
         std::vector<Ref<PrimExpr>> strides,
         int data_alignment,
         BufferScope scope)
    : name(std::move(name)),
      dtype(dtype),
      shape(std::move(shape)),
      strides(std::move(strides)),
      data_alignment(data_alignment),
      scope(scope) {}
};

inline Ref<Buffer> Buffer::make(std::string name,
                                 DataType dtype,
                                 std::vector<Ref<PrimExpr>> shape,
                                 std::vector<Ref<PrimExpr>> strides,
                                 int data_alignment,
                                 BufferScope scope) {
  return std::shared_ptr<Buffer>(new Buffer(std::move(name), dtype,
                                             std::move(shape),
                                             std::move(strides),
                                             data_alignment, scope));
}

} /* namespace rasp */

#endif /* RASP_INCLUDE_LOW_LEVEL_IR_BUFFER_H_ */
