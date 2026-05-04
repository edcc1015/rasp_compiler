#include "pass/pass_manager.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include "utils/log.h"

namespace rasp {

/* ── FunctionPass ─────────────────────────────────────────────────────────── */

Ref<IRModule> FunctionPass::transform_hlir(Ref<IRModule> mod,
                                            const PassContext& ctx) {
  auto new_mod = IRModule::make();
  for (auto& [name, func] : mod->functions) {
    auto new_func = transform_function(func, mod, ctx);
    new_mod->add_function(name, new_func);
  }
  return new_mod;
}

/* ── PassManager ──────────────────────────────────────────────────────────── */

void PassManager::add_pass(std::shared_ptr<Pass> pass) {
  passes_.push_back(std::move(pass));
}

void PassManager::add_passes(std::initializer_list<std::shared_ptr<Pass>> passes) {
  for (auto& p : passes) passes_.push_back(p);
}

void PassManager::clear() {
  passes_.clear();
}

void PassManager::print_pipeline() const {
  LOG_I("[PassManager] Pipeline:");
  for (size_t i = 0; i < passes_.size(); ++i) {
    std::ostringstream msg;
    msg << "  [" << i << "] " << passes_[i]->name()
        << " (min_opt_level=" << passes_[i]->min_opt_level() << ")";
    LOG_I(msg.str().c_str());
  }
}

bool PassManager::should_skip(const Pass& pass, const PassContext& ctx) const {
  /* Required passes always run regardless of opt_level. */
  for (const auto& req : ctx.required_passes) {
    if (req == pass.name()) return false;
  }
  if (ctx.is_disabled(pass.name())) return true;
  if (ctx.opt_level < pass.min_opt_level()) return true;
  return false;
}

void PassManager::dump_ir(Ref<IRModule> mod, int idx,
                           const std::string& pass_name,
                           const PassContext& ctx) const {
  std::filesystem::create_directories(ctx.dump_dir);
  std::ostringstream filename;
  filename << ctx.dump_dir << "/";
  if (idx < 10) filename << "0";
  filename << idx << "_" << pass_name << ".json";
  std::ofstream out(filename.str());
  if (!out) {
    LOG_E(("Failed to dump IR to: " + filename.str()).c_str());
    return;
  }
  out << mod->to_json() << "\n";
}

void PassManager::verify_dependencies() const {
  std::unordered_set<std::string> seen;
  for (auto& pass : passes_) {
    for (const auto& dep : pass->dependencies()) {
      if (seen.find(dep) == seen.end()) {
        throw std::runtime_error(
          "Pass '" + pass->name() +
          "' depends on '" + dep + "' which hasn't been added before it."
        );
      }
    }
    seen.insert(pass->name());
  }
}

Ref<IRModule> PassManager::run(Ref<IRModule> mod, const PassContext& ctx) const {
  verify_dependencies();
  if (ctx.dump_ir) {
    dump_ir(mod, 0, "input", ctx);
  }
  int dump_idx = 1;
  for (auto& pass : passes_) {
    if (should_skip(*pass, ctx)) {
      std::ostringstream msg;
      if (ctx.is_disabled(pass->name())) {
        msg << "[PassManager] Skipping (disabled): " << pass->name();
      } else {
        msg << "[PassManager] Skipping (opt_level too low): " << pass->name();
      }
      LOG_I(msg.str().c_str());
      continue;
    }
    auto t0 = std::chrono::steady_clock::now();
    LOG_I(("[PassManager] Running: " + pass->name()).c_str());
    mod = pass->transform_hlir(mod, ctx);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::ostringstream time_msg;
    time_msg << "[PassManager]   Time: " << ms << " ms";
    LOG_I(time_msg.str().c_str());
    if (ctx.dump_ir) {
      dump_ir(mod, dump_idx++, pass->name(), ctx);
    }
  }
  return mod;
}

Ref<LLIRModule> PassManager::run_llir(Ref<LLIRModule> mod,
                                       const PassContext& ctx) const {
  verify_dependencies();
  for (auto& pass : passes_) {
    if (should_skip(*pass, ctx)) {
      std::ostringstream msg;
      if (ctx.is_disabled(pass->name())) {
        msg << "[PassManager] Skipping (disabled): " << pass->name();
      } else {
        msg << "[PassManager] Skipping (opt_level too low): " << pass->name();
      }
      LOG_I(msg.str().c_str());
      continue;
    }
    auto t0 = std::chrono::steady_clock::now();
    LOG_I(("[PassManager] Running: " + pass->name()).c_str());
    mod = pass->transform_llir(mod, ctx);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::ostringstream time_msg;
    time_msg << "[PassManager]   Time: " << ms << " ms";
    LOG_I(time_msg.str().c_str());
  }
  return mod;
}

/* ── PassRegistry ─────────────────────────────────────────────────────────── */

std::unordered_map<std::string, PassRegistry::PassFactory>&
PassRegistry::registry() {
  static std::unordered_map<std::string, PassFactory> reg;
  return reg;
}

void PassRegistry::register_pass(const std::string& name, PassFactory factory) {
  auto& reg = registry();
  if (reg.count(name)) {
    throw std::runtime_error("PassRegistry: pass already registered: " + name);
  }
  reg[name] = std::move(factory);
}

std::shared_ptr<Pass> PassRegistry::create(const std::string& name) {
  auto& reg = registry();
  auto it = reg.find(name);
  if (it == reg.end()) {
    throw std::runtime_error("PassRegistry: pass not found: " + name);
  }
  return it->second();
}

bool PassRegistry::has(const std::string& name) {
  return registry().count(name) > 0;
}

std::vector<std::string> PassRegistry::list_all() {
  std::vector<std::string> names;
  for (auto& kv : registry()) names.push_back(kv.first);
  return names;
}

} /* namespace rasp */
