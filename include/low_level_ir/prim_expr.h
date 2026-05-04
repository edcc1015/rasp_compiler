#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "ir_node.h"
#include "type.h"

namespace rasp {

class Buffer; /* forward-declared to break the Buffer <-> PrimExpr cycle */

/* ──────────────────────────────────────────────────────────────────────────
 * PrimExpr
 * Abstract base class for all LLIR scalar / vector expressions.
 * dtype stores the scalar element type; Ramp and Broadcast carry lanes count.
 * ────────────────────────────────────────────────────────────────────────── */
class PrimExpr : public IRNode {
 public:
  DataType dtype;

 protected:
  explicit PrimExpr(DataType dtype) : dtype(dtype) {}
};

/* ──────────────────────────────────────────────────────────────────────────
 * LLVar
 * A named scalar variable inside a PrimFunc.
 * Represents loop counters, temporary accumulators, and pointer variables.
 * Distinct from HLIR Var (which is an Expr, not a PrimExpr).
 * ────────────────────────────────────────────────────────────────────────── */
class LLVar : public PrimExpr {
 public:
  std::string name;

  IRNodeType node_type() const override { return IRNodeType::kLLVar; }

  static Ref<LLVar> make(std::string name,
                          DataType dtype = DataType::kInt64);

 private:
  LLVar(std::string name, DataType dtype)
    : PrimExpr(dtype), name(std::move(name)) {}
};

inline Ref<LLVar> LLVar::make(std::string name, DataType dtype) {
  return std::shared_ptr<LLVar>(new LLVar(std::move(name), dtype));
}

/* ──────────────────────────────────────────────────────────────────────────
 * IntImm — compile-time integer constant
 * ────────────────────────────────────────────────────────────────────────── */
class IntImm : public PrimExpr {
 public:
  int64_t value;

  IRNodeType node_type() const override { return IRNodeType::kIntImm; }

  static Ref<IntImm> make(int64_t v,
                            DataType dt = DataType::kInt64);

 private:
  IntImm(int64_t v, DataType dt) : PrimExpr(dt), value(v) {}
};

inline Ref<IntImm> IntImm::make(int64_t v, DataType dt) {
  return std::shared_ptr<IntImm>(new IntImm(v, dt));
}

/* ──────────────────────────────────────────────────────────────────────────
 * FloatImm — compile-time floating-point constant
 * ────────────────────────────────────────────────────────────────────────── */
class FloatImm : public PrimExpr {
 public:
  double value;

  IRNodeType node_type() const override { return IRNodeType::kFloatImm; }

  static Ref<FloatImm> make(double v,
                            DataType dt = DataType::kFloat32);

 private:
  FloatImm(double v, DataType dt) : PrimExpr(dt), value(v) {}
};

inline Ref<FloatImm> FloatImm::make(double v, DataType dt) {
  return std::shared_ptr<FloatImm>(new FloatImm(v, dt));
}

/* ──────────────────────────────────────────────────────────────────────────
 * BufferLoad — read a scalar (or vector) value from a Buffer.
 * Buffer is only forward-declared here; include buffer.h for the full type.
 * make() is implemented in buffer.cpp because it reads buf->dtype.
 * ────────────────────────────────────────────────────────────────────────── */
class BufferLoad : public PrimExpr {
 public:
  Ref<Buffer> buffer;
  std::vector<Ref<PrimExpr>> indices; /* one index per buffer dimension */

  IRNodeType node_type() const override { return IRNodeType::kBufferLoad; }

  /* Implemented in buffer.cpp (needs full Buffer definition). */
  static Ref<BufferLoad> make(Ref<Buffer> buf,
                              std::vector<Ref<PrimExpr>> indices);

 private:
  BufferLoad(Ref<Buffer> buf,
             std::vector<Ref<PrimExpr>> indices,
             DataType dt)
    : PrimExpr(dt),
      buffer(std::move(buf)),
      indices(std::move(indices)) {}
};

/* ──────────────────────────────────────────────────────────────────────────
 * BinaryOp — abstract base for two-operand arithmetic / comparison nodes.
 * dtype is taken from operand a; both operands must share the same type.
 * Concrete subclasses: Add Sub Mul Div Mod FloorDiv FloorMod Min Max
 * ────────────────────────────────────────────────────────────────────────── */
class BinaryOp : public PrimExpr {
 public:
  Ref<PrimExpr> a;
  Ref<PrimExpr> b;

 protected:
  BinaryOp(DataType dtype, Ref<PrimExpr> a, Ref<PrimExpr> b)
    : PrimExpr(dtype), a(std::move(a)), b(std::move(b)) {}
};

/* Macro: stamp out nine identical BinaryOp subclasses. */
#define RASP_LLIR_BINARY_OP(Name, Kind)                                  \
  class Name : public BinaryOp {                                         \
   public:                                                               \
    IRNodeType node_type() const override { return IRNodeType::Kind; }   \
    static Ref<Name> make(Ref<PrimExpr> a, Ref<PrimExpr> b) {           \
      DataType dt = a->dtype;                                            \
      return std::shared_ptr<Name>(                                      \
        new Name(dt, std::move(a), std::move(b)));                       \
    }                                                                    \
   private:                                                              \
    Name(DataType dt, Ref<PrimExpr> a, Ref<PrimExpr> b)                 \
      : BinaryOp(dt, std::move(a), std::move(b)) {}                     \
  }

RASP_LLIR_BINARY_OP(Add, kAdd);
RASP_LLIR_BINARY_OP(Sub, kSub);
RASP_LLIR_BINARY_OP(Mul, kMul);
RASP_LLIR_BINARY_OP(Div, kDiv);
RASP_LLIR_BINARY_OP(Mod, kMod);
RASP_LLIR_BINARY_OP(FloorDiv, kFloorDiv);
RASP_LLIR_BINARY_OP(FloorMod, kFloorMod);
RASP_LLIR_BINARY_OP(Min, kMin);
RASP_LLIR_BINARY_OP(Max, kMax);

#undef RASP_LLIR_BINARY_OP

/* ──────────────────────────────────────────────────────────────────────────
 * Select — element-wise ternary: condition ? true_value : false_value
 * ────────────────────────────────────────────────────────────────────────── */
class Select : public PrimExpr {
 public:
  Ref<PrimExpr> condition;
  Ref<PrimExpr> true_value;
  Ref<PrimExpr> false_value;

  IRNodeType node_type() const override { return IRNodeType::kSelect; }

  static Ref<Select> make(Ref<PrimExpr> cond,
                          Ref<PrimExpr> true_val,
                          Ref<PrimExpr> false_val);

 private:
  Select(Ref<PrimExpr> cond,
         Ref<PrimExpr> true_val,
         Ref<PrimExpr> false_val)
    : PrimExpr(true_val->dtype),
      condition(std::move(cond)),
      true_value(std::move(true_val)),
      false_value(std::move(false_val)) {}
};

inline Ref<Select> Select::make(Ref<PrimExpr> cond,
                                Ref<PrimExpr> true_val,
                                Ref<PrimExpr> false_val) {
  return std::shared_ptr<Select>(
    new Select(std::move(cond), std::move(true_val), std::move(false_val)));
}

/* ──────────────────────────────────────────────────────────────────────────
 * Cast — type-conversion; target dtype is stored in PrimExpr::dtype.
 * ────────────────────────────────────────────────────────────────────────── */
class Cast : public PrimExpr {
 public:
  Ref<PrimExpr> value;

  IRNodeType node_type() const override { return IRNodeType::kCast; }

  static Ref<Cast> make(DataType target_dtype, Ref<PrimExpr> value);

 private:
  Cast(DataType target_dtype, Ref<PrimExpr> value)
    : PrimExpr(target_dtype), value(std::move(value)) {}
};

inline Ref<Cast> Cast::make(DataType target_dtype, Ref<PrimExpr> value) {
  return std::shared_ptr<Cast>(new Cast(target_dtype, std::move(value)));
}

/* ──────────────────────────────────────────────────────────────────────────
 * Ramp — vector index: [base, base+stride, …, base+(lanes-1)*stride]
 * dtype holds the scalar element type; lanes is the NEON vector width.
 * Typical use: Ramp(ow*4, 1, 4) for loading four consecutive FP32 values.
 * ────────────────────────────────────────────────────────────────────────── */
class Ramp : public PrimExpr {
 public:
  Ref<PrimExpr> base;
  Ref<PrimExpr> stride;
  int lanes;

  IRNodeType node_type() const override { return IRNodeType::kRamp; }

  static Ref<Ramp> make(Ref<PrimExpr> base,
                        Ref<PrimExpr> stride,
                        int lanes);

 private:
  Ramp(Ref<PrimExpr> base, Ref<PrimExpr> stride, int lanes)
    : PrimExpr(base->dtype),
      base(std::move(base)),
      stride(std::move(stride)),
      lanes(lanes) {}
};

inline Ref<Ramp> Ramp::make(Ref<PrimExpr> base,
                            Ref<PrimExpr> stride,
                            int lanes) {
  return std::shared_ptr<Ramp>(
    new Ramp(std::move(base), std::move(stride), lanes));
}

/* ──────────────────────────────────────────────────────────────────────────
 * Broadcast — replicates a scalar across lanes to form a vector.
 * Used to spread a weight constant across all NEON FP32 lanes.
 * ────────────────────────────────────────────────────────────────────────── */
class Broadcast : public PrimExpr {
 public:
  Ref<PrimExpr> value;
  int lanes;

  IRNodeType node_type() const override { return IRNodeType::kBroadcast; }

  static Ref<Broadcast> make(Ref<PrimExpr> value, int lanes);

 private:
  Broadcast(Ref<PrimExpr> value, int lanes)
    : PrimExpr(value->dtype),
      value(std::move(value)),
      lanes(lanes) {}
};

inline Ref<Broadcast> Broadcast::make(Ref<PrimExpr> value, int lanes) {
  return std::shared_ptr<Broadcast>(new Broadcast(std::move(value), lanes));
}

/* ──────────────────────────────────────────────────────────────────────────
 * CallType — discriminates the target of a PrimCall.
 * ────────────────────────────────────────────────────────────────────────── */
enum class CallType {
  kIntrinsic, /* ARM NEON intrinsic */
  kExtern,    /* external C-linkage function */
  kBuiltin,   /* built-in math function */
};

/* ──────────────────────────────────────────────────────────────────────────
 * PrimCall — calls a built-in, external, or NEON intrinsic function.
 * Named PrimCall to avoid collision with the HLIR Call (which is an Expr).
 * ────────────────────────────────────────────────────────────────────────── */
class PrimCall : public PrimExpr {
 public:
  std::string func_name;
  CallType call_type;
  std::vector<Ref<PrimExpr>> args;

  IRNodeType node_type() const override { return IRNodeType::kLLCall; }

  static Ref<PrimCall> make(DataType dtype,
                             std::string func_name,
                             CallType call_type,
                             std::vector<Ref<PrimExpr>> args);

 private:
  PrimCall(DataType dtype,
           std::string func_name,
           CallType call_type,
           std::vector<Ref<PrimExpr>> args)
    : PrimExpr(dtype),
      func_name(std::move(func_name)),
      call_type(call_type),
      args(std::move(args)) {}
};

inline Ref<PrimCall> PrimCall::make(DataType dtype,
                                     std::string func_name,
                                     CallType call_type,
                                     std::vector<Ref<PrimExpr>> args) {
  return std::shared_ptr<PrimCall>(
    new PrimCall(dtype, std::move(func_name), call_type, std::move(args)));
}

} /* namespace rasp */
