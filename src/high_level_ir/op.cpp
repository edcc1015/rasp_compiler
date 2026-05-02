#include "high_level_ir/op.h"

namespace rasp {

std::unordered_map<std::string, Ref<Op>>& OpRegistry::registry() {
  /* Meyer's singleton — initialized once, destroyed at program exit. */
  static std::unordered_map<std::string, Ref<Op>> instance;
  return instance;
}

void OpRegistry::register_op(const std::string& name,
                              std::vector<AttrSchema> schema,
                              const std::string& desc) {
  auto& reg = registry();
  if (reg.count(name)) {
    throw std::runtime_error("Op already registered: " + name);
  }
  /* Op constructor is private; use the friend access via a local subclass. */
  struct Enabler : Op {
    Enabler(const std::string& n, std::vector<AttrSchema> s, const std::string& d)
      : Op(n, std::move(s), d) {}
  };
  reg[name] = std::make_shared<Enabler>(name, std::move(schema), desc);
}

Ref<Op> OpRegistry::get(const std::string& name) {
  auto& reg = registry();
  auto  it  = reg.find(name);
  if (it == reg.end()) {
    throw std::runtime_error("Op not found: " + name);
  }
  return it->second;
}

bool OpRegistry::has(const std::string& name) {
  return registry().count(name) > 0;
}

std::vector<std::string> OpRegistry::list_all() {
  std::vector<std::string> names;
  names.reserve(registry().size());
  for (auto& kv : registry()) names.push_back(kv.first);
  return names;
}

} /* namespace rasp */
