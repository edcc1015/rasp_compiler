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

/* ══════════════════════════════════════════════════════════════════════════
 * Visitor / Mutator patterns for PrimExpr
 * ══════════════════════════════════════════════════════════════════════════ */

/* ──────────────────────────────────────────────────────────────────────────
 * PrimExprVisitor — read-only recursive traversal of a PrimExpr tree.
 * Override specific visit_* methods to inspect nodes of interest.
 * Default implementations recurse into child expressions.
 * ────────────────────────────────────────────────────────────────────────── */
class PrimExprVisitor {
 public:
  virtual ~PrimExprVisitor() = default;

  /* Entry point: dispatch to the appropriate visit_* method. */
  virtual void visit(Ref<PrimExpr> expr);

  /* Leaf nodes — default: no-op. */
  virtual void visit_ll_var(Ref<LLVar>) {}
  virtual void visit_int_imm(Ref<IntImm>) {}
  virtual void visit_float_imm(Ref<FloatImm>) {}

  /* Composite nodes — default: recurse into children. */
  virtual void visit_buffer_load(Ref<BufferLoad> node) {
    for (auto& idx : node->indices) visit(idx);
  }

  virtual void visit_add(Ref<Add> node) { visit(node->a); visit(node->b); }
  virtual void visit_sub(Ref<Sub> node) { visit(node->a); visit(node->b); }
  virtual void visit_mul(Ref<Mul> node) { visit(node->a); visit(node->b); }
  virtual void visit_div(Ref<Div> node) { visit(node->a); visit(node->b); }
  virtual void visit_mod(Ref<Mod> node) { visit(node->a); visit(node->b); }
  virtual void visit_floor_div(Ref<FloorDiv> node) { visit(node->a); visit(node->b); }
  virtual void visit_floor_mod(Ref<FloorMod> node) { visit(node->a); visit(node->b); }
  virtual void visit_min(Ref<Min> node) { visit(node->a); visit(node->b); }
  virtual void visit_max(Ref<Max> node) { visit(node->a); visit(node->b); }

  virtual void visit_select(Ref<Select> node) {
    visit(node->condition);
    visit(node->true_value);
    visit(node->false_value);
  }

  virtual void visit_cast(Ref<Cast> node) { visit(node->value); }

  virtual void visit_ramp(Ref<Ramp> node) {
    visit(node->base);
    visit(node->stride);
  }

  virtual void visit_broadcast(Ref<Broadcast> node) { visit(node->value); }

  virtual void visit_prim_call(Ref<PrimCall> node) {
    for (auto& arg : node->args) visit(arg);
  }
};

/* ──────────────────────────────────────────────────────────────────────────
 * PrimExprMutator — rewriting traversal of a PrimExpr tree.
 * Override specific mutate_* methods to substitute nodes.
 * Default implementations rebuild a node only when a child has changed.
 * ────────────────────────────────────────────────────────────────────────── */
class PrimExprMutator {
 public:
  virtual ~PrimExprMutator() = default;

  /* Entry point: dispatch to the appropriate mutate_* method. */
  virtual Ref<PrimExpr> mutate(Ref<PrimExpr> expr);

  /* Leaf nodes — default: return unchanged. */
  virtual Ref<PrimExpr> mutate_ll_var(Ref<LLVar> node)       { return node; }
  virtual Ref<PrimExpr> mutate_int_imm(Ref<IntImm> node)     { return node; }
  virtual Ref<PrimExpr> mutate_float_imm(Ref<FloatImm> node) { return node; }

  virtual Ref<PrimExpr> mutate_buffer_load(Ref<BufferLoad> node) {
    std::vector<Ref<PrimExpr>> new_idx;
    new_idx.reserve(node->indices.size());
    bool changed = false;
    for (auto& idx : node->indices) {
      auto ni = mutate(idx);
      if (ni != idx) changed = true;
      new_idx.push_back(std::move(ni));
    }
    if (!changed) return node;
    return BufferLoad::make(node->buffer, std::move(new_idx));
  }

  /* Helper: rebuild a BinaryOp node only when a child changed. */
  template <typename T>
  Ref<PrimExpr> mutate_binary(Ref<T> node) {
    auto na = mutate(node->a);
    auto nb = mutate(node->b);
    if (na == node->a && nb == node->b) return node;
    return T::make(std::move(na), std::move(nb));
  }

  virtual Ref<PrimExpr> mutate_add(Ref<Add> node) { return mutate_binary(node); }
  virtual Ref<PrimExpr> mutate_sub(Ref<Sub> node) { return mutate_binary(node); }
  virtual Ref<PrimExpr> mutate_mul(Ref<Mul> node) { return mutate_binary(node); }
  virtual Ref<PrimExpr> mutate_div(Ref<Div> node) { return mutate_binary(node); }
  virtual Ref<PrimExpr> mutate_mod(Ref<Mod> node) { return mutate_binary(node); }
  virtual Ref<PrimExpr> mutate_floor_div(Ref<FloorDiv> node){ return mutate_binary(node); }
  virtual Ref<PrimExpr> mutate_floor_mod(Ref<FloorMod> node){ return mutate_binary(node); }
  virtual Ref<PrimExpr> mutate_min(Ref<Min> node) { return mutate_binary(node); }
  virtual Ref<PrimExpr> mutate_max(Ref<Max> node) { return mutate_binary(node); }

  virtual Ref<PrimExpr> mutate_select(Ref<Select> node) {
    auto nc = mutate(node->condition);
    auto nt = mutate(node->true_value);
    auto nf = mutate(node->false_value);
    if (nc == node->condition &&
        nt == node->true_value &&
        nf == node->false_value) return node;
    return Select::make(std::move(nc), std::move(nt), std::move(nf));
  }

  virtual Ref<PrimExpr> mutate_cast(Ref<Cast> node) {
    auto nv = mutate(node->value);
    if (nv == node->value) return node;
    return Cast::make(node->dtype, std::move(nv));
  }

  virtual Ref<PrimExpr> mutate_ramp(Ref<Ramp> node) {
    auto nb = mutate(node->base);
    auto ns = mutate(node->stride);
    if (nb == node->base && ns == node->stride) return node;
    return Ramp::make(std::move(nb), std::move(ns), node->lanes);
  }

  virtual Ref<PrimExpr> mutate_broadcast(Ref<Broadcast> node) {
    auto nv = mutate(node->value);
    if (nv == node->value) return node;
    return Broadcast::make(std::move(nv), node->lanes);
  }

  virtual Ref<PrimExpr> mutate_prim_call(Ref<PrimCall> node) {
    std::vector<Ref<PrimExpr>> new_args;
    new_args.reserve(node->args.size());
    bool changed = false;
    for (auto& arg : node->args) {
      auto na = mutate(arg);
      if (na != arg) changed = true;
      new_args.push_back(std::move(na));
    }
    if (!changed) return node;
    return PrimCall::make(node->dtype, node->func_name,
                          node->call_type, std::move(new_args));
  }
};

/* ──────────────────────────────────────────────────────────────────────────
 * PrimExprVisitor::visit  — type-dispatch (defined inline after all classes)
 * PrimExprMutator::mutate — type-dispatch (defined inline after all classes)
 * ────────────────────────────────────────────────────────────────────────── */
inline void PrimExprVisitor::visit(Ref<PrimExpr> expr) {
  if (!expr) return;
  switch (expr->node_type()) {
    case IRNodeType::kLLVar:
      return visit_ll_var(std::static_pointer_cast<LLVar>(expr));
    case IRNodeType::kIntImm:
      return visit_int_imm(std::static_pointer_cast<IntImm>(expr));
    case IRNodeType::kFloatImm:
      return visit_float_imm(std::static_pointer_cast<FloatImm>(expr));
    case IRNodeType::kBufferLoad:
      return visit_buffer_load(std::static_pointer_cast<BufferLoad>(expr));
    case IRNodeType::kAdd:
      return visit_add(std::static_pointer_cast<Add>(expr));
    case IRNodeType::kSub:
      return visit_sub(std::static_pointer_cast<Sub>(expr));
    case IRNodeType::kMul:
      return visit_mul(std::static_pointer_cast<Mul>(expr));
    case IRNodeType::kDiv:
      return visit_div(std::static_pointer_cast<Div>(expr));
    case IRNodeType::kMod:
      return visit_mod(std::static_pointer_cast<Mod>(expr));
    case IRNodeType::kFloorDiv:
      return visit_floor_div(std::static_pointer_cast<FloorDiv>(expr));
    case IRNodeType::kFloorMod:
      return visit_floor_mod(std::static_pointer_cast<FloorMod>(expr));
    case IRNodeType::kMin:
      return visit_min(std::static_pointer_cast<Min>(expr));
    case IRNodeType::kMax:
      return visit_max(std::static_pointer_cast<Max>(expr));
    case IRNodeType::kSelect:
      return visit_select(std::static_pointer_cast<Select>(expr));
    case IRNodeType::kCast:
      return visit_cast(std::static_pointer_cast<Cast>(expr));
    case IRNodeType::kRamp:
      return visit_ramp(std::static_pointer_cast<Ramp>(expr));
    case IRNodeType::kBroadcast:
      return visit_broadcast(std::static_pointer_cast<Broadcast>(expr));
    case IRNodeType::kLLCall:
      return visit_prim_call(std::static_pointer_cast<PrimCall>(expr));
    default:
      throw std::runtime_error("PrimExprVisitor: unknown PrimExpr node type");
  }
}

inline Ref<PrimExpr> PrimExprMutator::mutate(Ref<PrimExpr> expr) {
  if (!expr) return nullptr;
  switch (expr->node_type()) {
    case IRNodeType::kLLVar:
      return mutate_ll_var(std::static_pointer_cast<LLVar>(expr));
    case IRNodeType::kIntImm:
      return mutate_int_imm(std::static_pointer_cast<IntImm>(expr));
    case IRNodeType::kFloatImm:
      return mutate_float_imm(std::static_pointer_cast<FloatImm>(expr));
    case IRNodeType::kBufferLoad:
      return mutate_buffer_load(std::static_pointer_cast<BufferLoad>(expr));
    case IRNodeType::kAdd:
      return mutate_add(std::static_pointer_cast<Add>(expr));
    case IRNodeType::kSub:
      return mutate_sub(std::static_pointer_cast<Sub>(expr));
    case IRNodeType::kMul:
      return mutate_mul(std::static_pointer_cast<Mul>(expr));
    case IRNodeType::kDiv:
      return mutate_div(std::static_pointer_cast<Div>(expr));
    case IRNodeType::kMod:
      return mutate_mod(std::static_pointer_cast<Mod>(expr));
    case IRNodeType::kFloorDiv:
      return mutate_floor_div(std::static_pointer_cast<FloorDiv>(expr));
    case IRNodeType::kFloorMod:
      return mutate_floor_mod(std::static_pointer_cast<FloorMod>(expr));
    case IRNodeType::kMin:
      return mutate_min(std::static_pointer_cast<Min>(expr));
    case IRNodeType::kMax:
      return mutate_max(std::static_pointer_cast<Max>(expr));
    case IRNodeType::kSelect:
      return mutate_select(std::static_pointer_cast<Select>(expr));
    case IRNodeType::kCast:
      return mutate_cast(std::static_pointer_cast<Cast>(expr));
    case IRNodeType::kRamp:
      return mutate_ramp(std::static_pointer_cast<Ramp>(expr));
    case IRNodeType::kBroadcast:
      return mutate_broadcast(std::static_pointer_cast<Broadcast>(expr));
    case IRNodeType::kLLCall:
      return mutate_prim_call(std::static_pointer_cast<PrimCall>(expr));
    default:
      throw std::runtime_error("PrimExprMutator: unknown PrimExpr node type");
  }
}

} /* namespace rasp */
