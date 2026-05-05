#ifndef RASP_INCLUDE_PASS_PASS_CONTEXT_H_
#define RASP_INCLUDE_PASS_PASS_CONTEXT_H_

#pragma once

#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace rasp {

/* ──────────────────────────────────────────────────────────────────────────
 * PassContext
 * Carries shared configuration and state for a Pass execution run.
 * ────────────────────────────────────────────────────────────────────────── */
struct PassContext {
  int opt_level = 2;
  std::vector<std::string> required_passes;
  std::vector<std::string> disabled_passes;
  bool dump_ir = false;
  std::string dump_dir = "/tmp/rasp_ir_dump";
  std::unordered_map<std::string, std::string> config;

  bool is_disabled(const std::string& pass_name) const {
    for (const auto& name : disabled_passes) {
      if (name == pass_name) return true;
    }
    return false;
  }

  template <typename T>
  T get_config(const std::string& key, T default_val) const {
    auto it = config.find(key);
    if (it == config.end()) return default_val;
    std::istringstream ss(it->second);
    T val;
    ss >> val;
    if (ss.fail()) return default_val;
    return val;
  }
};

template <>
inline std::string PassContext::get_config<std::string>(
    const std::string& key, std::string default_val) const {
  auto it = config.find(key);
  if (it == config.end()) return default_val;
  return it->second;
}

} /* namespace rasp */

#endif /* RASP_INCLUDE_PASS_PASS_CONTEXT_H_ */
