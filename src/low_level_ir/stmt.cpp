#include "low_level_ir/stmt.h"

namespace rasp {

/* ──────────────────────────────────────────────────────────────────────────
 * SeqStmt::flatten
 * Recursively unwraps nested SeqStmt nodes into a single flat list, then:
 *   - returns the single element directly if the list has only one entry;
 *   - wraps in a new SeqStmt otherwise.
 * ────────────────────────────────────────────────────────────────────────── */
static void collect_stmts(Ref<Stmt> s, std::vector<Ref<Stmt>>& out) {
  if (!s) return;
  if (s->node_type() == IRNodeType::kSeqStmt) {
    auto seq = std::static_pointer_cast<SeqStmt>(s);
    for (auto& child : seq->seq) {
      collect_stmts(child, out);
    }
  } else {
    out.push_back(std::move(s));
  }
}

Ref<Stmt> SeqStmt::flatten(std::vector<Ref<Stmt>> stmts) {
  std::vector<Ref<Stmt>> flat;
  for (auto& s : stmts) {
    collect_stmts(s, flat);
  }
  if (flat.empty()) return nullptr;
  if (flat.size() == 1) return flat[0];
  return SeqStmt::make(std::move(flat));
}

} /* namespace rasp */
