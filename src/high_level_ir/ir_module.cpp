/*
 * ir_module.cpp — IRModule serialization / deserialization.
 *
 * to_json  : IR expression tree  →  compact JSON string
 * from_json: JSON string          →  IR expression tree
 *
 * Uses nlohmann/json for all JSON construction and parsing.
 * Base64 helpers are kept for Constant::data (raw float bytes in JSON strings).
 * Compile with -I include -I third_party.
 */

#include "high_level_ir/hl_ir.h"
#include "nlohmann/json.hpp"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace rasp {

using json = nlohmann::json;

/* ══════════════════════════════════════════════════════════════════════════
 * Base64 — encode/decode raw bytes to/from a JSON-safe string
 * ══════════════════════════════════════════════════════════════════════════ */

static const char B64_TABLE[] =
  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static std::string base64_encode(const void* src, size_t len) {
  const uint8_t* in = reinterpret_cast<const uint8_t*>(src);
  std::string out;
  out.reserve(((len + 2) / 3) * 4);
  for (size_t i = 0; i < len; i += 3) {
    uint32_t b = static_cast<uint32_t>(in[i]) << 16;
    if (i + 1 < len) b |= static_cast<uint32_t>(in[i + 1]) << 8;
    if (i + 2 < len) b |= in[i + 2];
    out += B64_TABLE[(b >> 18) & 0x3F];
    out += B64_TABLE[(b >> 12) & 0x3F];
    out += (i + 1 < len) ? B64_TABLE[(b >> 6) & 0x3F] : '=';
    out += (i + 2 < len) ? B64_TABLE[b & 0x3F]        : '=';
  }
  return out;
}

static std::vector<uint8_t> base64_decode(const std::string& s) {
  uint8_t rev[256];
  memset(rev, 0xFF, sizeof(rev));
  for (int i = 0; i < 64; ++i) rev[static_cast<uint8_t>(B64_TABLE[i])] = i;

  std::vector<uint8_t> out;
  out.reserve((s.size() / 4) * 3);
  for (size_t i = 0; i + 3 < s.size(); i += 4) {
    uint8_t a = rev[static_cast<uint8_t>(s[i])];
    uint8_t b = rev[static_cast<uint8_t>(s[i + 1])];
    uint8_t c = rev[static_cast<uint8_t>(s[i + 2])];
    uint8_t d = rev[static_cast<uint8_t>(s[i + 3])];
    uint32_t tri = (uint32_t(a) << 18) | (uint32_t(b) << 12)
                 | (uint32_t(c) << 6)  |  uint32_t(d);
    out.push_back((tri >> 16) & 0xFF);
    if (s[i + 2] != '=') out.push_back((tri >> 8) & 0xFF);
    if (s[i + 3] != '=') out.push_back(tri & 0xFF);
  }
  return out;
}

/* ══════════════════════════════════════════════════════════════════════════
 * to_json helpers — IR node → nlohmann::json object
 * ══════════════════════════════════════════════════════════════════════════ */

static json type_to_json(const Ref<IRNode>& t);
static json expr_to_json(const Ref<Expr>& e);

static json attrs_to_json(const Attrs& attrs) {
  json j = json::object();
  for (auto& kv : attrs.values) {
    std::visit([&](auto&& v) {
      using T = std::decay_t<decltype(v)>;
      if constexpr (std::is_same_v<T, int64_t>) {
        j[kv.first] = v;
      } else if constexpr (std::is_same_v<T, double>) {
        j[kv.first] = v;
      } else if constexpr (std::is_same_v<T, std::string>) {
        j[kv.first] = v;
      } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
        j[kv.first] = v;
      } else if constexpr (std::is_same_v<T, std::vector<double>>) {
        j[kv.first] = v;
      }
    }, kv.second);
  }
  return j;
}

static json type_to_json(const Ref<IRNode>& t) {
  if (!t) return nullptr;
  switch (t->node_type()) {
    case IRNodeType::kTensorType: {
      auto* tt = static_cast<TensorType*>(t.get());
      return {
        {"kind",  "TensorType"},
        {"dtype", dtype_to_string(tt->dtype)},
        {"shape", tt->shape},
      };
    }
    case IRNodeType::kTupleType: {
      auto* tt = static_cast<TupleType*>(t.get());
      json fields = json::array();
      for (auto& f : tt->fields) fields.push_back(type_to_json(f));
      return {{"kind", "TupleType"}, {"fields", std::move(fields)}};
    }
    case IRNodeType::kFuncType: {
      auto* ft = static_cast<FuncType*>(t.get());
      json arg_types = json::array();
      for (auto& a : ft->arg_types) arg_types.push_back(type_to_json(a));
      return {
        {"kind",       "FuncType"},
        {"arg_types",  std::move(arg_types)},
        {"ret_type",   type_to_json(ft->ret_type)},
      };
    }
    default:
      return nullptr;
  }
}

static json expr_to_json(const Ref<Expr>& e) {
  if (!e) return nullptr;
  switch (e->node_type()) {

    case IRNodeType::kVar: {
      auto* v = static_cast<Var*>(e.get());
      json j = {{"kind", "Var"}, {"name", v->name}};
      if (v->type_annotation) j["type"] = type_to_json(v->type_annotation);
      return j;
    }

    case IRNodeType::kConstant: {
      auto* c = static_cast<Constant*>(e.get());
      json j;
      j["kind"] = "Constant";
      if (c->tensor_type) {
        j["dtype"] = dtype_to_string(c->tensor_type->dtype);
        j["shape"] = c->tensor_type->shape;
      } else {
        j["dtype"] = "float32";
      }
      j["data"] = base64_encode(c->data.data(), c->data.size());
      return j;
    }

    case IRNodeType::kCall: {
      auto* call = static_cast<Call*>(e.get());
      json j;
      j["kind"] = "Call";
      if (call->op && call->op->node_type() == IRNodeType::kOp) {
        auto* op = static_cast<Op*>(call->op.get());
        j["op"] = {{"kind", "Op"}, {"name", op->name}};
      } else if (call->op) {
        j["op"] = expr_to_json(std::static_pointer_cast<Expr>(call->op));
      } else {
        j["op"] = nullptr;
      }
      json args = json::array();
      for (auto& a : call->args) args.push_back(expr_to_json(a));
      j["args"] = std::move(args);
      if (!call->attrs.values.empty())
        j["attrs"] = attrs_to_json(call->attrs);
      return j;
    }

    case IRNodeType::kLet: {
      auto* let = static_cast<Let*>(e.get());
      return {
        {"kind",  "Let"},
        {"var",   expr_to_json(let->var)},
        {"value", expr_to_json(let->value)},
        {"body",  expr_to_json(let->body)},
      };
    }

    case IRNodeType::kTuple: {
      auto* tup = static_cast<Tuple*>(e.get());
      json fields = json::array();
      for (auto& f : tup->fields) fields.push_back(expr_to_json(f));
      return {{"kind", "Tuple"}, {"fields", std::move(fields)}};
    }

    case IRNodeType::kTupleGetItem: {
      auto* tgi = static_cast<TupleGetItem*>(e.get());
      return {
        {"kind",  "TupleGetItem"},
        {"tuple", expr_to_json(tgi->tuple)},
        {"index", tgi->index},
      };
    }

    case IRNodeType::kFunction: {
      auto* fn = static_cast<Function*>(e.get());
      json params = json::array();
      for (auto& p : fn->params) params.push_back(expr_to_json(p));
      json j = {
        {"kind",   "Function"},
        {"params", std::move(params)},
        {"body",   expr_to_json(fn->body)},
      };
      if (fn->ret_type) j["ret_type"] = type_to_json(fn->ret_type);
      if (!fn->attrs.empty()) j["attrs"] = fn->attrs;
      return j;
    }

    default:
      return nullptr;
  }
}

/* ──────────────────────────────────────────────────────────────────────────
 * IRModule::to_json
 * ────────────────────────────────────────────────────────────────────────── */
std::string IRModule::to_json() const {
  json j;
  j["kind"] = "IRModule";
  j["functions"] = json::object();
  for (auto& kv : functions) {
    j["functions"][kv.first] = expr_to_json(kv.second);
  }
  return j.dump();
}

/* ══════════════════════════════════════════════════════════════════════════
 * from_json helpers — nlohmann::json object → IR node
 * ══════════════════════════════════════════════════════════════════════════ */

static Ref<IRNode> json_to_type(const json& j);
static Ref<Expr>   json_to_expr(const json& j);

static Attrs json_to_attrs(const json& j) {
  Attrs attrs;
  for (auto& [key, val] : j.items()) {
    if (val.is_string()) {
      attrs.values[key] = val.get<std::string>();
    } else if (val.is_boolean()) {
      /* JSON bool → int64_t (0/1) to fit into AttrValue variant. */
      attrs.values[key] = val.get<bool>() ? int64_t(1) : int64_t(0);
    } else if (val.is_number_integer()) {
      attrs.values[key] = val.get<int64_t>();
    } else if (val.is_number_float()) {
      attrs.values[key] = val.get<double>();
    } else if (val.is_array() && !val.empty()) {
      if (val[0].is_number_integer()) {
        attrs.values[key] = val.get<std::vector<int64_t>>();
      } else {
        attrs.values[key] = val.get<std::vector<double>>();
      }
    }
  }
  return attrs;
}

static Ref<IRNode> json_to_type(const json& j) {
  if (j.is_null()) return nullptr;
  const std::string& kind = j.at("kind").get<std::string>();

  if (kind == "TensorType") {
    DataType dt = dtype_from_string(j.at("dtype").get<std::string>());
    return TensorType::make(j.at("shape").get<std::vector<int64_t>>(), dt);
  }
  if (kind == "TupleType") {
    std::vector<Ref<IRNode>> fields;
    for (auto& f : j.at("fields")) fields.push_back(json_to_type(f));
    return TupleType::make(std::move(fields));
  }
  if (kind == "FuncType") {
    std::vector<Ref<IRNode>> arg_types;
    for (auto& a : j.at("arg_types")) arg_types.push_back(json_to_type(a));
    return FuncType::make(std::move(arg_types), json_to_type(j.at("ret_type")));
  }
  throw std::runtime_error("from_json: unknown type kind '" + kind + "'");
}

static Ref<Expr> json_to_expr(const json& j) {
  if (j.is_null()) return nullptr;
  const std::string& kind = j.at("kind").get<std::string>();

  if (kind == "Var") {
    Ref<IRNode> ta;
    if (j.contains("type") && !j["type"].is_null())
      ta = json_to_type(j["type"]);
    return Var::make(j.at("name").get<std::string>(), ta);
  }

  if (kind == "Constant") {
    DataType dt = dtype_from_string(j.at("dtype").get<std::string>());
    auto shape = j.at("shape").get<std::vector<int64_t>>();
    auto raw = base64_decode(j.at("data").get<std::string>());
    return Constant::make(std::move(raw), TensorType::make(shape, dt));
  }

  if (kind == "Call") {
    Ref<IRNode> op_node;
    const json& op_j = j.at("op");
    if (!op_j.is_null()) {
      const std::string& op_kind = op_j.at("kind").get<std::string>();
      if (op_kind == "Op") {
        std::string name = op_j.at("name").get<std::string>();
        if (OpRegistry::has(name)) {
          op_node = OpRegistry::get(name);
        } else {
          std::cerr << "[from_json] WARNING: Op '" << name
                    << "' not found in OpRegistry — op will be null. "
                    << "Make sure op_registry_init.cpp is linked.\n";
          op_node = nullptr;
        }
      } else {
        op_node = json_to_expr(op_j);
      }
    }
    std::vector<Ref<Expr>> args;
    for (auto& a : j.at("args")) args.push_back(json_to_expr(a));
    Attrs attrs;
    if (j.contains("attrs")) attrs = json_to_attrs(j["attrs"]);
    return Call::make(op_node, std::move(args), std::move(attrs));
  }

  if (kind == "Let") {
    auto var = std::dynamic_pointer_cast<Var>(json_to_expr(j.at("var")));
    return Let::make(var, json_to_expr(j.at("value")), json_to_expr(j.at("body")));
  }

  if (kind == "Tuple") {
    std::vector<Ref<Expr>> fields;
    for (auto& f : j.at("fields")) fields.push_back(json_to_expr(f));
    return Tuple::make(std::move(fields));
  }

  if (kind == "TupleGetItem") {
    return TupleGetItem::make(
      json_to_expr(j.at("tuple")),
      j.at("index").get<int>());
  }

  if (kind == "Function") {
    std::vector<Ref<Var>> params;
    for (auto& p : j.at("params"))
      params.push_back(std::dynamic_pointer_cast<Var>(json_to_expr(p)));
    Ref<IRNode> ret_type;
    if (j.contains("ret_type") && !j["ret_type"].is_null())
      ret_type = json_to_type(j["ret_type"]);
    std::unordered_map<std::string, std::string> fn_attrs;
    if (j.contains("attrs"))
      fn_attrs = j["attrs"].get<std::unordered_map<std::string, std::string>>();
    return Function::make(std::move(params), json_to_expr(j.at("body")),
                          ret_type, std::move(fn_attrs));
  }

  throw std::runtime_error("from_json: unknown expr kind '" + kind + "'");
}

/* ──────────────────────────────────────────────────────────────────────────
 * IRModule::from_json
 * ────────────────────────────────────────────────────────────────────────── */
Ref<IRModule> IRModule::from_json(const std::string& json_str) {
  json j = json::parse(json_str);
  if (j.at("kind").get<std::string>() != "IRModule")
    throw std::runtime_error("from_json: top-level kind is not 'IRModule'");

  auto mod = IRModule::make();
  for (auto& [name, fn_j] : j.at("functions").items()) {
    auto fn = std::dynamic_pointer_cast<Function>(json_to_expr(fn_j));
    if (!fn)
      throw std::runtime_error("from_json: '" + name + "' is not a Function");
    mod->add_function(name, fn);
  }
  return mod;
}

} /* namespace rasp */
