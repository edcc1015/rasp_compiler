#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "low_level_ir/buffer.h"
#include "low_level_ir/stmt.h"

namespace rasp {

/* ──────────────────────────────────────────────────────────────────────────
 * PrimFunc
 * The unit of compilation in the LLIR: corresponds to one operator (or fused
 * operator group) from the HLIR.  Lowering converts each HLIR Call node into
 * a PrimFunc with explicit Buffer parameters and a nested-loop Stmt body.
 *
 * attrs carries optional metadata consumed by Codegen:
 *   attrs["target"] = "arm_neon"
 *   attrs["layout"] = "NCHWc"
 * ────────────────────────────────────────────────────────────────────────── */
class PrimFunc : public IRNode {
 public:
  std::string name;
  std::vector<Ref<Buffer>> params; /* input and output Buffers in order  */
  Ref<Stmt> body;   /* Stmt tree describing the computation */
  std::unordered_map<std::string, std::string> attrs;

  IRNodeType node_type() const override { return IRNodeType::kPrimFunc; }

  static Ref<PrimFunc> make(
    std::string name,
    std::vector<Ref<Buffer>> params,
    Ref<Stmt> body,
    std::unordered_map<std::string, std::string> attrs = {});

 private:
  PrimFunc(std::string name,
           std::vector<Ref<Buffer>> params,
           Ref<Stmt> body,
           std::unordered_map<std::string, std::string> attrs)
    : name(std::move(name)),
      params(std::move(params)),
      body(std::move(body)),
      attrs(std::move(attrs)) {}
};

inline Ref<PrimFunc> PrimFunc::make(
    std::string name,
    std::vector<Ref<Buffer>> params,
    Ref<Stmt> body,
    std::unordered_map<std::string, std::string> attrs) {
  return std::shared_ptr<PrimFunc>(
    new PrimFunc(std::move(name), std::move(params),
                 std::move(body), std::move(attrs)));
}

} /* namespace rasp */
