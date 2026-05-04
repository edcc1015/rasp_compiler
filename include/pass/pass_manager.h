#pragma once

#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "pass.h"

namespace rasp {

/* ──────────────────────────────────────────────────────────────────────────
 * PassManager
 * Manages and executes a pipeline of Passes in registration order.
 * ────────────────────────────────────────────────────────────────────────── */
class PassManager {
 public:
  void add_pass(std::shared_ptr<Pass> pass);
  void add_passes(std::initializer_list<std::shared_ptr<Pass>> passes);

  /* Execute all registered HLIR passes; returns the final transformed module. */
  Ref<IRModule> run(Ref<IRModule> mod, const PassContext& ctx) const;

  /* Execute all registered LLIR passes; returns the final transformed module. */
  Ref<LLIRModule> run_llir(Ref<LLIRModule> mod, const PassContext& ctx) const;

  void clear();
  void print_pipeline() const;

 private:
  std::vector<std::shared_ptr<Pass>> passes_;

  /* Returns true if the pass should be skipped given the context. */
  bool should_skip(const Pass& pass, const PassContext& ctx) const;

  /* Writes mod as JSON to dump_dir/<idx>_<pass_name>.json when dump_ir is set. */
  void dump_ir(Ref<IRModule> mod, int idx,
               const std::string& pass_name,
               const PassContext& ctx) const;

  /* Validates that all declared dependencies appear earlier in passes_. */
  void verify_dependencies() const;
};

/* ──────────────────────────────────────────────────────────────────────────
 * SequentialPass
 * Wraps a list of Passes as a single named ModulePass (nested PassManager).
 * ────────────────────────────────────────────────────────────────────────── */
class SequentialPass : public ModulePass {
 public:
  SequentialPass(std::vector<std::shared_ptr<Pass>> passes, std::string name)
    : passes_(std::move(passes)), name_(std::move(name)) {}

  std::string name() const override { return name_; }

  Ref<IRModule> transform_module(Ref<IRModule> mod,
                                  const PassContext& ctx) override {
    PassManager inner;
    for (auto& p : passes_) inner.add_pass(p);
    return inner.run(mod, ctx);
  }

 private:
  std::vector<std::shared_ptr<Pass>> passes_;
  std::string name_;
};

/* ──────────────────────────────────────────────────────────────────────────
 * PassRegistry
 * Global factory registry for dynamic pipeline construction.
 * ────────────────────────────────────────────────────────────────────────── */
class PassRegistry {
 public:
  using PassFactory = std::function<std::shared_ptr<Pass>()>;

  static void register_pass(const std::string& name, PassFactory factory);
  static std::shared_ptr<Pass> create(const std::string& name);
  static bool has(const std::string& name);
  static std::vector<std::string> list_all();

 private:
  static std::unordered_map<std::string, PassFactory>& registry();
};

/* Place at the end of each Pass .cpp file to auto-register the pass. */
#define REGISTER_PASS(cls) \
  static bool _##cls##_registered = []() { \
    ::rasp::PassRegistry::register_pass(#cls, []() { \
      return std::make_shared<cls>(); \
    }); \
    return true; \
  }()

} /* namespace rasp */
