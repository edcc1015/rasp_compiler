#include "low_level_ir/buffer.h"
#include "low_level_ir/prim_expr.h"

namespace rasp {

/* ──────────────────────────────────────────────────────────────────────────
 * BufferLoad::make
 * Defined here (not inline) because it needs the full Buffer definition to
 * read buf->dtype.
 * ────────────────────────────────────────────────────────────────────────── */
Ref<BufferLoad> BufferLoad::make(Ref<Buffer>                 buf,
                                  std::vector<Ref<PrimExpr>> indices) {
  DataType dtype = buf->dtype;
  return std::shared_ptr<BufferLoad>(
    new BufferLoad(std::move(buf), std::move(indices), dtype));
}

/* ──────────────────────────────────────────────────────────────────────────
 * Buffer::linearize
 * Computes the flat integer offset for a set of multi-dimensional indices.
 *
 * If strides is non-empty:
 *   offset = sum_i( indices[i] * strides[i] )
 *
 * If strides is empty (row-major contiguous):
 *   stride[last]   = 1
 *   stride[i]      = stride[i+1] * shape[i+1]
 *   offset         = sum_i( indices[i] * stride[i] )
 *
 * Returns IntImm(0) for a zero-dimensional (scalar) buffer.
 * ────────────────────────────────────────────────────────────────────────── */
Ref<PrimExpr> Buffer::linearize(
    const std::vector<Ref<PrimExpr>>& indices) const {
  const std::size_t ndim = indices.size();

  if (ndim == 0) {
    return IntImm::make(0, DataType::kInt32);
  }

  /* Helper: multiply two PrimExprs using the Add/Mul node types.
   * We use int32 arithmetic for index computation throughout. */
  auto mul_expr = [](Ref<PrimExpr> a, Ref<PrimExpr> b) -> Ref<PrimExpr> {
    return Mul::make(std::move(a), std::move(b));
  };
  auto add_expr = [](Ref<PrimExpr> a, Ref<PrimExpr> b) -> Ref<PrimExpr> {
    return Add::make(std::move(a), std::move(b));
  };

  auto zero = []() -> Ref<PrimExpr> {
    return IntImm::make(0, DataType::kInt32);
  };
  auto one = []() -> Ref<PrimExpr> {
    return IntImm::make(1, DataType::kInt32);
  };

  if (!strides.empty()) {
    /* Explicit strides provided. */
    Ref<PrimExpr> offset = zero();
    for (std::size_t i = 0; i < ndim; ++i) {
      offset = add_expr(std::move(offset),
                        mul_expr(indices[i], strides[i]));
    }
    return offset;
  }

  /* Row-major: derive strides from shape.
   * stride[ndim-1] = 1
   * stride[i]      = shape[i+1] * stride[i+1]  */
  std::vector<Ref<PrimExpr>> row_strides(ndim);
  row_strides[ndim - 1] = one();
  for (std::size_t i = ndim - 1; i-- > 0; ) {
    row_strides[i] = mul_expr(shape[i + 1], row_strides[i + 1]);
  }

  Ref<PrimExpr> offset = zero();
  for (std::size_t i = 0; i < ndim; ++i) {
    offset = add_expr(std::move(offset),
                      mul_expr(indices[i], row_strides[i]));
  }
  return offset;
}

} /* namespace rasp */
