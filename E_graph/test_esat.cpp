#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <functional>
#include <optional>

// ======================
//   Basic ETL types
// ======================

using ModeType = int;
using Modes = std::vector<ModeType>;

enum class OpKind {
    Input,
    Perm,
    Gemm
};

std::ostream& operator<<(std::ostream& os, OpKind k) {
    switch (k) {
        case OpKind::Input: os << "Input"; break;
        case OpKind::Perm:  os << "Perm";  break;
        case OpKind::Gemm:  os << "Gemm";  break;
    }
    return os;
}

// ======================
//   ENode
// ======================

struct ENode {
    OpKind op;
    std::vector<int> children; // E-class IDs (canonicalized via find())

    // --- Modes ---
    // Input:
    Modes input_modes; // used when op == Input

    // Perm:
    Modes perm_in_modes;   // modes before perm
    Modes perm_out_modes;  // modes after perm

    // Gemm:
    Modes gemm_l_in_modes;
    Modes gemm_r_in_modes;
    Modes gemm_out_modes;  // output tensor modes

    Modes M, N, K, C;      // GEMM layout decomposition (you decide semantics)

    // Constructors for each op
    static ENode make_input(const Modes& in) {
        ENode n;
        n.op = OpKind::Input;
        n.input_modes = in;
        return n;
    }

    static ENode make_perm(int child_expr,
                           const Modes& in_modes,
                           const Modes& out_modes) {
        ENode n;
        n.op = OpKind::Perm;
        n.children = {child_expr};
        n.perm_in_modes  = in_modes;
        n.perm_out_modes = out_modes;
        return n;
    }

    static ENode make_gemm(int left_expr,
                           int right_expr,
                           const Modes& l_in,
                           const Modes& r_in,
                           const Modes& out,
                           const Modes& M,
                           const Modes& N,
                           const Modes& K,
                           const Modes& C) {
        ENode n;
        n.op = OpKind::Gemm;
        n.children = {left_expr, right_expr};
        n.gemm_l_in_modes  = l_in;
        n.gemm_r_in_modes  = r_in;
        n.gemm_out_modes   = out;
        n.M = M; n.N = N; n.K = K; n.C = C;
        return n;
    }

    bool operator==(const ENode& o) const {
        return op == o.op &&
               children == o.children &&
               input_modes == o.input_modes &&
               perm_in_modes == o.perm_in_modes &&
               perm_out_modes == o.perm_out_modes &&
               gemm_l_in_modes == o.gemm_l_in_modes &&
               gemm_r_in_modes == o.gemm_r_in_modes &&
               gemm_out_modes == o.gemm_out_modes &&
               M == o.M && N == o.N && K == o.K && C == o.C;
    }
};

struct ENodeHash {
    std::size_t operator()(const ENode& n) const {
        auto h = std::hash<int>{}(static_cast<int>(n.op));
        auto mix = [&](std::size_t v) {
            h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        };
        for (int c : n.children) mix(std::hash<int>{}(c));
        auto hash_modes = [&](const Modes& m) {
            std::size_t hh = 0;
            for (auto x : m) {
                hh ^= std::hash<int>{}(x) + 0x9e3779b97f4a7c15ULL + (hh << 6) + (hh >> 2);
            }
            return hh;
        };
        mix(hash_modes(n.input_modes));
        mix(hash_modes(n.perm_in_modes));
        mix(hash_modes(n.perm_out_modes));
        mix(hash_modes(n.gemm_l_in_modes));
        mix(hash_modes(n.gemm_r_in_modes));
        mix(hash_modes(n.gemm_out_modes));
        mix(hash_modes(n.M));
        mix(hash_modes(n.N));
        mix(hash_modes(n.K));
        mix(hash_modes(n.C));
        return h;
    }
};

// ======================
//   EClass
// ======================

struct EClass {
    int id;
    std::vector<int> nodes; // indices into EGraph::enodes
};

// ======================
//   EGraph with union–find
// ======================

class EGraph {
public:
    // Storage
    std::vector<ENode> enodes;
    std::vector<EClass> classes;

    // union–find parent over eclasses
    std::vector<int> parent;

    // memo: canonical ENode -> eclass id
    std::unordered_map<ENode, int, ENodeHash> memo;

    // Find representative of eclass
    int find(int x) {
        if (parent[x] != x)
            parent[x] = find(parent[x]);
        return parent[x];
    }

    // Merge two eclasses
    void merge(int a, int b) {
        a = find(a);
        b = find(b);
        if (a == b) return;
        // very simple union; could do union-by-rank, but not necessary for demo
        parent[b] = a;

        // We don't do a full rebuild here; for a serious implementation,
        // you’d want to rebuild memo + compress.
        classes[a].nodes.insert(classes[a].nodes.end(),
                                classes[b].nodes.begin(),
                                classes[b].nodes.end());
        classes[b].nodes.clear();
    }

    // Add a node; returns the eclass id for the resulting class
    int add(const ENode& raw_node) {
        // Canonicalize children by union–find
        ENode node = raw_node;
        for (int& c : node.children)
            c = find(c);

        auto it = memo.find(node);
        if (it != memo.end()) {
            // Node already exists in this eclass
            int cid = it->second;
            int nid = (int)enodes.size();
            enodes.push_back(node);
            classes[cid].nodes.push_back(nid);
            return cid;
        }

        // Create new eclass
        int cid = (int)classes.size();
        int nid = (int)enodes.size();
        enodes.push_back(node);
        classes.push_back(EClass{cid, {nid}});
        parent.push_back(cid);
        memo.emplace(node, cid);
        return cid;
    }

    void print() {
        std::cout << "EGraph:\n";
        for (auto& ec : classes) {
            if (ec.nodes.empty()) continue;
            int rep = find(ec.id);
            std::cout << "  EClass #" << ec.id;
            if (rep != ec.id) std::cout << " (-> " << rep << ")";
            std::cout << ":\n";
            for (int nid : ec.nodes) {
                const auto& n = enodes[nid];
                std::cout << "    " << n.op << " children=[";
                for (auto c : n.children) std::cout << find(c) << " ";
                std::cout << "]";
                if (n.op == OpKind::Input) {
                    std::cout << " input_modes=[";
                    for (auto m : n.input_modes) std::cout << m << " ";
                    std::cout << "]";
                } else if (n.op == OpKind::Perm) {
                    std::cout << " in=[";
                    for (auto m : n.perm_in_modes) std::cout << m << " ";
                    std::cout << "] out=[";
                    for (auto m : n.perm_out_modes) std::cout << m << " ";
                    std::cout << "]";
                } else if (n.op == OpKind::Gemm) {
                    std::cout << " l_in=[";
                    for (auto m : n.gemm_l_in_modes) std::cout << m << " ";
                    std::cout << "] r_in=[";
                    for (auto m : n.gemm_r_in_modes) std::cout << m << " ";
                    std::cout << "] out=[";
                    for (auto m : n.gemm_out_modes) std::cout << m << " ";
                    std::cout << "]";
                }
                std::cout << "\n";
            }
        }
    }
};

// ======================
//   Rewrite rules
// ======================

struct RewriteRule {
    std::string name;
    // Return true if changed something
    std::function<bool(EGraph&)> apply;
};

// ======================
//   Helper: apply saturation
// ======================

void run_saturation(EGraph& eg,
                    std::vector<RewriteRule>& rules,
                    int max_iter = 5) {
    for (int iter = 0; iter < max_iter; ++iter) {
        bool changed = false;
        for (auto& r : rules) {
            if (r.apply(eg)) changed = true;
        }
        if (!changed) {
            std::cout << "Saturation converged at iter " << iter << "\n";
            return;
        }
    }
    std::cout << "Saturation stopped at max_iter=" << max_iter << "\n";
}

// ======================
//   Concrete pattern: 
//   perm ∘ gemm ∘ perm/perm
// ======================

// Look for a node that matches:
//
//   Perm(I, O, Gemm( O, l, r,
//                    Perm(l, any_modes, exp1),
//                    Perm(r, any_modes, exp2)))
//
// In our representation:
//   - root: Perm (call it P_top)
//     P_top.children[0] -> eclass with at least one Gemm node G
//   - G.children[0/1] -> eclasses with Perm nodes P_l, P_r
//
// This function tries to find one such pattern and, if found,
// adds an alternative with transformed modes, then merges.
bool apply_perm_gemm_layout_rule(EGraph& eg) {
    bool changed = false;

    const int num_classes = (int)eg.classes.size();
    for (int cid = 0; cid < num_classes; ++cid) {
        int rep_cid = eg.find(cid);
        if (rep_cid != cid) continue; // only look at canonical classes

        auto& ec = eg.classes[cid];
        for (int nid : ec.nodes) {
            const ENode& top = eg.enodes[nid];
            if (top.op != OpKind::Perm) continue;

            if (top.children.size() != 1) continue;
            int gemm_cid = eg.find(top.children[0]);
            auto& gemm_ec = eg.classes[gemm_cid];

            // Find a Gemm node inside that eclass
            int gemm_nid = -1;
            for (int g_nid : gemm_ec.nodes) {
                if (eg.enodes[g_nid].op == OpKind::Gemm) {
                    gemm_nid = g_nid;
                    break;
                }
            }
            if (gemm_nid == -1) continue;

            const ENode& gemm = eg.enodes[gemm_nid];
            if (gemm.children.size() != 2) continue;

            int left_perm_cid  = eg.find(gemm.children[0]);
            int right_perm_cid = eg.find(gemm.children[1]);

            auto& left_ec  = eg.classes[left_perm_cid];
            auto& right_ec = eg.classes[right_perm_cid];

            // Find Perm nodes on both sides
            int left_perm_nid  = -1;
            int right_perm_nid = -1;

            for (int pnid : left_ec.nodes) {
                if (eg.enodes[pnid].op == OpKind::Perm) {
                    left_perm_nid = pnid;
                    break;
                }
            }
            if (left_perm_nid == -1) continue;

            for (int pnid : right_ec.nodes) {
                if (eg.enodes[pnid].op == OpKind::Perm) {
                    right_perm_nid = pnid;
                    break;
                }
            }
            if (right_perm_nid == -1) continue;

            const ENode& P_top   = top;
            const ENode& P_left  = eg.enodes[left_perm_nid];
            const ENode& P_right = eg.enodes[right_perm_nid];

            // Here you can check mode constraints, e.g.:
            // - P_top.perm_in_modes == P_left.perm_in_modes ⊗ P_right.perm_in_modes ?
            // - P_top.perm_out_modes == gemm.gemm_out_modes ?
            // For demo, we just accept any.

            // -------------------------------------------
            // Compute new modes O', l', r'
            // -------------------------------------------
            Modes O_prime  = gemm.gemm_out_modes;
            Modes l_prime  = gemm.gemm_l_in_modes;
            Modes r_prime  = gemm.gemm_r_in_modes;

            // TODO: ***Your GEMM layout algebra goes here***
            // e.g. rewrite O,l,r according to some layout rules.
            // For now, just copy (no-op).

            // Build new Gemm node with updated modes (could change M,N,K,C too)
            ENode new_gemm = ENode::make_gemm(
                left_perm_cid,
                right_perm_cid,
                l_prime,
                r_prime,
                O_prime,
                gemm.M,
                gemm.N,
                gemm.K,
                gemm.C
            );
            int new_gemm_cid = eg.add(new_gemm);

            // Build new top perm with new out_modes O'
            ENode new_top_perm = ENode::make_perm(
                new_gemm_cid,
                P_top.perm_in_modes,  // still I
                O_prime               // O'
            );
            int new_top_cid = eg.add(new_top_perm);

            // Merge old top perm's eclass with new one
            eg.merge(cid, new_top_cid);

            std::cout << "[rule] perm-gemm layout variant added: merged EClass "
                      << cid << " with " << new_top_cid << "\n";

            changed = true;
            // We only apply once per saturation step for demo;
            // remove 'return changed;' if you want multiple per pass.
            return changed;
        }
    }
    return changed;
}

// ======================
//   Demo main
// ======================

int main() {
    EGraph eg;

    // Build a tiny ETL expression:
    //
    //   P_top = Perm(I, O,  Gemm( O, l, r,
    //                             Perm(l, any1, exp1),
    //                             Perm(r, any2, exp2) ))
    //
    // For now, exp1, exp2 are just Inputs.

    Modes I  = {0, 1};    // input modes
    Modes O  = {1, 0};    // output modes (some perm)
    Modes l  = {0};       // left GEMM modes
    Modes r  = {1};       // right GEMM modes
    Modes any1 = {0};     // dummy
    Modes any2 = {1};     // dummy

    // exp1, exp2
    int exp1_cid = eg.add(ENode::make_input({0}));
    int exp2_cid = eg.add(ENode::make_input({1}));

    // Perm(l, any1, exp1)
    int P_l_cid = eg.add(ENode::make_perm(exp1_cid, l, any1));

    // Perm(r, any2, exp2)
    int P_r_cid = eg.add(ENode::make_perm(exp2_cid, r, any2));

    // Gemm(O, l, r, P_l, P_r)
    Modes M = {0};
    Modes N = {1};
    Modes K = {2};
    Modes C = {}; // whatever you want
    int G_cid = eg.add(ENode::make_gemm(P_l_cid, P_r_cid, l, r, O, M, N, K, C));

    // Top perm: Perm(I, O, Gemm(...))
    int P_top_cid = eg.add(ENode::make_perm(G_cid, I, O));

    std::cout << "Before saturation:\n";
    eg.print();

    // Add the perm-gemm layout rule
    std::vector<RewriteRule> rules;
    rules.push_back({"perm_gemm_layout",
                     [](EGraph& eg) { return apply_perm_gemm_layout_rule(eg); }});

    // Run e-saturation
    run_saturation(eg, rules, /*max_iter=*/5);

    std::cout << "\nAfter saturation:\n";
    eg.print();

    return 0;
}

