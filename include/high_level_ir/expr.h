#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "ir_node.h"
#include "type.h"
#include "op.h"

namespace rasp {

/* ──────────────────────────────────────────────────────────────────────────
 * Expr
 * Base class for all expression nodes.
 * checked_type is filled by the type-inference pass; nullptr until then.
 * ────────────────────────────────────────────────────────────────────────── */
class Expr : public IRNode {
 public:
  mutable Ref<IRNode> checked_type;
};

/* ──────────────────────────────────────────────────────────────────────────
 * Var
 * A named variable reference.
 * Used as function parameters and Let-binding variables.
 * ────────────────────────────────────────────────────────────────────────── */
class Var : public Expr {
 public:
  std::string name;
  /* Optional explicit type annotation set by the frontend. */
  Ref<IRNode> type_annotation;

  IRNodeType node_type() const override { return IRNodeType::kVar; }

  static Ref<Var> make(std::string name, Ref<IRNode> type_annotation = nullptr);

 private:
  explicit Var(std::string name, Ref<IRNode> type_annotation)
    : name(std::move(name)), type_annotation(std::move(type_annotation)) {}
};

inline Ref<Var> Var::make(std::string name, Ref<IRNode> type_annotation) {
  return std::shared_ptr<Var>(new Var(std::move(name), std::move(type_annotation)));
}

/* ──────────────────────────────────────────────────────────────────────────
 * Constant
 * A constant tensor node.
 * data is stored row-major as float32 raw bytes for simplicity;
 * the actual dtype is recorded in tensor_type.
 * ────────────────────────────────────────────────────────────────────────── */
class Constant : public Expr {
 public:
  std::vector<float> data;
  Ref<TensorType> tensor_type;

  IRNodeType node_type() const override { return IRNodeType::kConstant; }

  static Ref<Constant> make(std::vector<float> data, Ref<TensorType> ttype);

 private:
  Constant(std::vector<float> data, Ref<TensorType> ttype)
    : data(std::move(data)), tensor_type(std::move(ttype)) {}
};

inline Ref<Constant> Constant::make(std::vector<float> data, Ref<TensorType> ttype) {
  return std::shared_ptr<Constant>(new Constant(std::move(data), std::move(ttype)));
}

/* ──────────────────────────────────────────────────────────────────────────
 * Attrs
 * Generic key-value attribute bag attached to a Call node.
 * Values may be int64_t, double, string, vector<int64_t>, or vector<double>.
 * ────────────────────────────────────────────────────────────────────────── */
using AttrValue = std::variant<
  int64_t,
  double,
  std::string,
  std::vector<int64_t>,
  std::vector<double>
>;

class Attrs {
 public:
  std::unordered_map<std::string, AttrValue> values;

  /* Retrieve attribute, throwing if absent or wrong type. */
  template <typename T>
  T get(const std::string& key) const {
    auto it = values.find(key);
    if (it == values.end()) {
      throw std::out_of_range("Attr not found: " + key);
    }
    return std::get<T>(it->second);
  }

  /* Retrieve attribute with a default if absent. */
  template <typename T>
  T get_or(const std::string& key, T default_val) const {
    auto it = values.find(key);
    if (it == values.end()) return default_val;
    return std::get<T>(it->second);
  }

  bool has(const std::string& key) const {
    return values.count(key) > 0;
  }
};

/* ──────────────────────────────────────────────────────────────────────────
 * Call
 * The central expression node representing one operator invocation.
 * op points to an Op descriptor (or a Function for inlined fused ops).
 * ────────────────────────────────────────────────────────────────────────── */
class Call : public Expr {
 public:
  Ref<IRNode> op;
  std::vector<Ref<Expr>> args;
  Attrs attrs;

  IRNodeType node_type() const override { return IRNodeType::kCall; }

  static Ref<Call> make(Ref<IRNode> op,
                         std::vector<Ref<Expr>> args,
                         Attrs attrs = {});

 private:
  Call(Ref<IRNode> op, std::vector<Ref<Expr>> args, Attrs attrs)
    : op(std::move(op)), args(std::move(args)), attrs(std::move(attrs)) {}
};

inline Ref<Call> Call::make(Ref<IRNode> op, std::vector<Ref<Expr>> args, Attrs attrs) {
  return std::shared_ptr<Call>(new Call(std::move(op), std::move(args), std::move(attrs)));
}

/* ──────────────────────────────────────────────────────────────────────────
 * Let
 * Let-binding: introduces a named variable bound to a value expression,
 * scoped over a body expression.  Used by CSE to avoid duplicate subtrees.
 *
 *   let var = value in body
 * ────────────────────────────────────────────────────────────────────────── */
class Let : public Expr {
 public:
  Ref<Var> var;
  Ref<Expr> value;
  Ref<Expr> body;

  IRNodeType node_type() const override { return IRNodeType::kLet; }

  static Ref<Let> make(Ref<Var> var, Ref<Expr> value, Ref<Expr> body);

 private:
  Let(Ref<Var> var, Ref<Expr> value, Ref<Expr> body)
    : var(std::move(var)),
      value(std::move(value)),
      body(std::move(body)) {}
};

inline Ref<Let> Let::make(Ref<Var> var, Ref<Expr> value, Ref<Expr> body) {
  return std::shared_ptr<Let>(new Let(std::move(var), std::move(value), std::move(body)));
}

/* ──────────────────────────────────────────────────────────────────────────
 * Tuple
 * Constructs a tuple from a list of expressions.
 * Typically used for the return value of multi-output operators.
 * ────────────────────────────────────────────────────────────────────────── */
class Tuple : public Expr {
 public:
  std::vector<Ref<Expr>> fields;

  IRNodeType node_type() const override { return IRNodeType::kTuple; }

  static Ref<Tuple> make(std::vector<Ref<Expr>> fields);

 private:
  explicit Tuple(std::vector<Ref<Expr>> fields)
    : fields(std::move(fields)) {}
};

inline Ref<Tuple> Tuple::make(std::vector<Ref<Expr>> fields) {
  return std::shared_ptr<Tuple>(new Tuple(std::move(fields)));
}

/* ──────────────────────────────────────────────────────────────────────────
 * TupleGetItem
 * Extracts the element at a given index from a Tuple expression.
 * ────────────────────────────────────────────────────────────────────────── */
class TupleGetItem : public Expr {
 public:
  Ref<Expr> tuple;
  int index;

  IRNodeType node_type() const override { return IRNodeType::kTupleGetItem; }

  static Ref<TupleGetItem> make(Ref<Expr> tuple, int index);

 private:
  TupleGetItem(Ref<Expr> tuple, int index)
    : tuple(std::move(tuple)), index(index) {}
};

inline Ref<TupleGetItem> TupleGetItem::make(Ref<Expr> tuple, int index) {
  return std::shared_ptr<TupleGetItem>(new TupleGetItem(std::move(tuple), index));
}

/* ──────────────────────────────────────────────────────────────────────────
 * Function
 * A first-class function expression.
 * When attrs["Primitive"] == "1" it represents a fused operator kernel.
 * ────────────────────────────────────────────────────────────────────────── */
class Function : public Expr {
 public:
  std::vector<Ref<Var>> params;
  Ref<Expr> body;
  Ref<IRNode> ret_type;
  std::unordered_map<std::string, std::string> attrs;

  IRNodeType node_type() const override { return IRNodeType::kFunction; }

  static Ref<Function> make(
    std::vector<Ref<Var>> params,
    Ref<Expr> body,
    Ref<IRNode> ret_type = nullptr,
    std::unordered_map<std::string, std::string> attrs = {});

  bool is_primitive() const {
    auto it = attrs.find("Primitive");
    return it != attrs.end() && it->second == "1";
  }

 private:
  Function(std::vector<Ref<Var>> params,
           Ref<Expr> body,
           Ref<IRNode> ret_type,
           std::unordered_map<std::string, std::string> attrs)
    : params(std::move(params)),
      body(std::move(body)),
      ret_type(std::move(ret_type)),
      attrs(std::move(attrs)) {}
};

inline Ref<Function> Function::make(
    std::vector<Ref<Var>> params,
    Ref<Expr> body,
    Ref<IRNode> ret_type,
    std::unordered_map<std::string, std::string> attrs) {
  return std::shared_ptr<Function>(
    new Function(std::move(params), std::move(body),
                 std::move(ret_type), std::move(attrs)));
}

} /* namespace rasp */
