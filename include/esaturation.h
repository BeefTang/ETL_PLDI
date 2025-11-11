#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <functional>
#include <optional>
#include <iostream>
#include <cassert>

// =========================
// Basic AST for expressions
// =========================

struct Expr {
    std::string op;
    std::vector<std::shared_ptr<Expr>> children;

    Expr(std::string op, std::vector<std::shared_ptr<Expr>> children = {})
        : op(std::move(op)), children(std::move(children)) {}

    static std::shared_ptr<Expr> make(
        const std::string &op, std::vector<std::shared_ptr<Expr>> children = {})
    {
        return std::make_shared<Expr>(op, std::move(children));
    }
};

using EClassId = std::size_t;

// =========================
// Union-Find
// =========================

struct UnionFind {
    std::vector<EClassId> parent;
    std::vector<unsigned> rank;

    EClassId make() {
        EClassId id = parent.size();
        parent.push_back(id);
        rank.push_back(0);
        return id;
    }

    EClassId find(EClassId x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }

    bool unite(EClassId a, EClassId b, EClassId &outRoot) {
        a = find(a);
        b = find(b);
        if (a == b) { outRoot = a; return false; }
        if (rank[a] < rank[b]) std::swap(a, b);
        parent[b] = a;
        if (rank[a] == rank[b]) rank[a]++;
        outRoot = a;
        return true;
    }
};

// =========================
// E-node and E-class
// =========================

struct ENode {
    std::string op;
    std::vector<EClassId> children; // children are e-class IDs

    bool operator==(const ENode &other) const {
        return op == other.op && children == other.children;
    }
};

struct ENodeHash {
    std::size_t operator()(const ENode &n) const noexcept {
        std::size_t h = std::hash<std::string>{}(n.op);
        for (auto c : n.children) {
            h ^= std::hash<EClassId>{}(c) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        }
        return h;
    }
};

struct EClass {
    EClassId id{};
    std::vector<std::size_t> nodes; // indices into EGraph::nodes
};

// =========================
// E-graph
// =========================

class EGraph {
public:
    std::vector<ENode> nodes;           // all nodes
    std::vector<EClass> classes;        // indexed by EClassId
    UnionFind uf;
    std::unordered_map<ENode, EClassId, ENodeHash> enode_table;

    // Optional: name -> leaf eclass (for variables / constants)
    std::unordered_map<std::string, EClassId> leaf_env;

    EGraph() = default;

    EClassId make_eclass() {
        EClassId id = uf.make();
        if (id >= classes.size()) {
            classes.push_back(EClass{ id, {} });
        }
        return id;
    }

    EClassId find(EClassId id) const {
        return const_cast<UnionFind&>(uf).find(id);
    }

    // Add an ENode; canonicalize children first; hash-cons it into an e-class.
    EClassId add_enode(const ENode &n_raw) {
        ENode n = n_raw;
        for (auto &c : n.children) {
            c = uf.find(c);
        }

        auto it = enode_table.find(n);
        if (it != enode_table.end()) {
            return uf.find(it->second);
        }

        // New e-node -> new e-class
        EClassId cid = make_eclass();
        std::size_t nid = nodes.size();
        nodes.push_back(n);
        classes[cid].nodes.push_back(nid);
        enode_table.emplace(n, cid);
        return cid;
    }

    // Build from Expr tree
    EClassId add_expr(const std::shared_ptr<Expr> &e) {
        if (!e) throw std::runtime_error("null Expr");

        // Leaf
        if (e->children.empty()) {
            auto it = leaf_env.find(e->op);
            if (it != leaf_env.end()) return it->second;
            EClassId cid = make_eclass();
            // Represent leaves as op with no children
            ENode n{ e->op, {} };
            std::size_t nid = nodes.size();
            nodes.push_back(n);
            classes[cid].nodes.push_back(nid);
            enode_table.emplace(n, cid);
            leaf_env[e->op] = cid;
            return cid;
        }

        std::vector<EClassId> child_ids;
        child_ids.reserve(e->children.size());
        for (auto &ch : e->children) {
            child_ids.push_back(add_expr(ch));
        }
        ENode n{ e->op, child_ids };
        return add_enode(n);
    }

    // Merge two e-classes (returns representative)
    EClassId merge(EClassId a, EClassId b) {
        EClassId root;
        if (!uf.unite(a, b, root)) {
            return root; // already same
        }
        // We'll repair classes + enode_table in rebuild()
        return root;
    }

    // Rebuild e-graph after merges (restores congruence invariants)
    void rebuild() {
        // 1. Canonicalize children of all nodes
        for (auto &n : nodes) {
            for (auto &c : n.children) {
                c = uf.find(c);
            }
        }

        // 2. Re-collect nodes into their canonical e-classes
        std::vector<std::vector<std::size_t>> tmp_nodes(classes.size());
        for (EClassId cid = 0; cid < classes.size(); ++cid) {
            EClassId root = uf.find(cid);
            if (root >= tmp_nodes.size()) tmp_nodes.resize(root + 1);
            for (auto ni : classes[cid].nodes) {
                tmp_nodes[root].push_back(ni);
            }
        }

        for (EClassId cid = 0; cid < classes.size(); ++cid) {
            classes[cid].id = cid;
            classes[cid].nodes = std::move(tmp_nodes[cid]);
        }

        // 3. Rebuild enode_table
        enode_table.clear();
        for (EClassId cid = 0; cid < classes.size(); ++cid) {
            if (classes[cid].nodes.empty()) continue;
            for (auto ni : classes[cid].nodes) {
                enode_table[nodes[ni]] = cid;
            }
        }
    }

    void dump() const {
        std::cout << "EGraph dump:\n";
        for (EClassId cid = 0; cid < classes.size(); ++cid) {
            if (classes[cid].nodes.empty()) continue;
            std::cout << "  EClass " << cid << " [root=" << uf.parent[cid] << "]:\n";
            for (auto ni : classes[cid].nodes) {
                auto &n = nodes[ni];
                std::cout << "    " << n.op << "(";
                for (std::size_t i = 0; i < n.children.size(); ++i) {
                    std::cout << n.children[i];
                    if (i + 1 < n.children.size()) std::cout << ", ";
                }
                std::cout << ")\n";
            }
        }
    }
};

// =========================
// Pattern language
// =========================

struct Pattern {
    bool is_var = false;        // ?x style
    std::string op;             // operator name or var name
    std::vector<Pattern> children;

    static Pattern Var(const std::string &name) {
        Pattern p;
        p.is_var = true;
        p.op = name;
        return p;
    }

    static Pattern Node(const std::string &op, std::vector<Pattern> children = {}) {
        Pattern p;
        p.is_var = false;
        p.op = op;
        p.children = std::move(children);
        return p;
    }
};

struct MatchContext {
    std::unordered_map<std::string, EClassId> bindings;
};

// Recursive matching: pattern vs e-class
bool match_pattern_eclass(const Pattern &p,
                          const EGraph &eg,
                          EClassId cid,
                          MatchContext &ctx);

// Try matching pattern against one e-node in a given e-class
bool match_pattern_enode(const Pattern &p,
                         const EGraph &eg,
                         EClassId cid,
                         const ENode &node,
                         MatchContext &ctx)
{
    if (p.is_var) {
        auto it = ctx.bindings.find(p.op);
        if (it == ctx.bindings.end()) {
            ctx.bindings[p.op] = cid;
            return true;
        } else {
            return it->second == cid;
        }
    }

    if (p.op != node.op) return false;
    if (p.children.size() != node.children.size()) return false;

    // Need backtracking for child matches
    MatchContext saved = ctx;
    for (std::size_t i = 0; i < p.children.size(); ++i) {
        EClassId child_cid = eg.find(node.children[i]);
        if (!match_pattern_eclass(p.children[i], eg, child_cid, ctx)) {
            ctx = saved; // rollback
            return false;
        }
    }
    return true;
}

bool match_pattern_eclass(const Pattern &p,
                          const EGraph &eg,
                          EClassId cid,
                          MatchContext &ctx)
{
    cid = eg.find(cid);
    if (cid >= eg.classes.size()) return false;
    auto &ec = eg.classes[cid];
    if (ec.nodes.empty()) return false;

    // If var, we don't need to inspect nodes at all
    if (p.is_var) {
        auto it = ctx.bindings.find(p.op);
        if (it == ctx.bindings.end()) {
            ctx.bindings[p.op] = cid;
            return true;
        } else {
            return it->second == cid;
        }
    }

    // Try every enode in this eclass
    for (auto ni : ec.nodes) {
        const ENode &n = eg.nodes[ni];
        MatchContext tmp = ctx;
        if (match_pattern_enode(p, eg, cid, n, tmp)) {
            ctx = std::move(tmp);
            return true;
        }
    }
    return false;
}

// Find all matches: (root eclass, MatchContext)
struct Match {
    EClassId root;
    MatchContext ctx;
};

std::vector<Match> find_matches(const Pattern &pat, const EGraph &eg) {
    std::vector<Match> out;
    for (EClassId cid = 0; cid < eg.classes.size(); ++cid) {
        if (eg.classes[cid].nodes.empty()) continue;
        MatchContext ctx;
        if (match_pattern_eclass(pat, eg, cid, ctx)) {
            out.push_back(Match{ cid, std::move(ctx) });
        }
    }
    return out;
}

// Build an e-class from a RHS pattern using bindings
EClassId build_from_pattern(const Pattern &p,
                            const MatchContext &ctx,
                            EGraph &eg)
{
    if (p.is_var) {
        auto it = ctx.bindings.find(p.op);
        if (it == ctx.bindings.end())
            throw std::runtime_error("Unbound pattern var: " + p.op);
        return eg.find(it->second);
    }

    std::vector<EClassId> child_ids;
    child_ids.reserve(p.children.size());
    for (auto &ch : p.children) {
        child_ids.push_back(build_from_pattern(ch, ctx, eg));
    }
    ENode n{ p.op, child_ids };
    return eg.add_enode(n);
}

// =========================
// Rewrite rule and saturation
// =========================

struct RewriteRule {
    std::string name;
    Pattern lhs;
    Pattern rhs;

    RewriteRule(std::string name, Pattern lhs, Pattern rhs)
        : name(std::move(name)), lhs(std::move(lhs)), rhs(std::move(rhs)) {}
};

// One saturation pass: apply all rules until no changes or limit
bool run_one_saturation_round(EGraph &eg,
                              const std::vector<RewriteRule> &rules,
                              std::size_t &applied,
                              std::size_t max_applications_per_round = 1000)
{
    bool changed = false;
    for (const auto &rule : rules) {
        auto matches = find_matches(rule.lhs, eg);
        for (auto &m : matches) {
            if (applied >= max_applications_per_round) break;

            EClassId new_id = build_from_pattern(rule.rhs, m.ctx, eg);
            eg.merge(m.root, new_id);
            applied++;
            changed = true;
            // In a more serious implementation, you'd avoid re-matching
            // immediately after each merge for performance.
        }
    }
    if (changed) eg.rebuild();
    return changed;
}

// High-level saturation driver
void saturate(EGraph &eg,
              const std::vector<RewriteRule> &rules,
              std::size_t iter_limit = 10,
              std::size_t app_limit_per_round = 1000)
{
    for (std::size_t iter = 0; iter < iter_limit; ++iter) {
        std::size_t applied = 0;
        bool changed = run_one_saturation_round(eg, rules, applied, app_limit_per_round);
        std::cout << "[saturate] iter " << iter
                  << ", rewrites applied: " << applied << "\n";
        if (!changed) break;
    }
}


