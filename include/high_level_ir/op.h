#ifndef RASP_INCLUDE_HIGH_LEVEL_IR_OP_H_
#define RASP_INCLUDE_HIGH_LEVEL_IR_OP_H_

#pragma once

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "ir_node.h"

namespace rasp {

/* ──────────────────────────────────────────────────────────────────────────
 * AttrSchema
 * Describes one attribute slot of an Op.
 * ────────────────────────────────────────────────────────────────────────── */
struct AttrSchema {
  std::string name;
  /*
   * Allowed type strings: "int", "float", "ints", "floats", "string".
   * These mirror the ONNX attribute type vocabulary used by the frontend.
   * Allowed attributes, their types, required or optional, and default values.
   */
  std::string type;
  bool required;
  std::string default_value;
};

/* ──────────────────────────────────────────────────────────────────────────
 * Op
 * Lightweight operator descriptor stored in the global OpRegistry.
 * Nodes of this type are shared; they are never mutated after registration.
 * ────────────────────────────────────────────────────────────────────────── */
class Op : public IRNode {
 public:
  std::string name;
  std::string description;
  std::vector<AttrSchema> attr_schema;

  IRNodeType node_type() const override { return IRNodeType::kOp; }

 private:
  Op(std::string name, std::vector<AttrSchema> schema, std::string desc)
    : name(std::move(name)),
      description(std::move(desc)),
      attr_schema(std::move(schema)) {}

  friend class OpRegistry;
};

/* ──────────────────────────────────────────────────────────────────────────
 * OpRegistry
 * Global singleton that maps op names to Op descriptors.
 *
 * Usage:
 *   OpRegistry::register_op("nn.relu", {}, "ReLU activation");
 *   Ref<Op> op = OpRegistry::get("nn.relu");
 * ────────────────────────────────────────────────────────────────────────── */
class OpRegistry {
 public:
  /* Register a new op.  Throws std::runtime_error if already registered. */
  static void register_op(const std::string& name,
                          std::vector<AttrSchema> schema,
                          const std::string& desc = "");

  /* Retrieve a registered op.  Throws std::runtime_error if not found. */
  static Ref<Op> get(const std::string& name);

  /* Returns true iff the op has been registered. */
  static bool has(const std::string& name);

  /* Returns the names of all registered ops in unspecified order. */
  static std::vector<std::string> list_all();

 private:
  static std::unordered_map<std::string, Ref<Op>>& registry();
};

} /* namespace rasp */

#endif /* RASP_INCLUDE_HIGH_LEVEL_IR_OP_H_ */
