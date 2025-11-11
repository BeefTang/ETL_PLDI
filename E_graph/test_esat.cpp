#pragma once
#include <vector>
#include <unordered_map>
#include <string>
#include <iostream>
#include <functional>

/**************************************
 *  Basic ETL Mode Definitions
 **************************************/
using ModeType = int;
using Modes = std::vector<ModeType>;

/**************************************
 *  OpKind
 **************************************/
enum class OpKind { Input, Perm, Gemm };

inline std::ostream &operator<<(std::ostream &os, OpKind k) {
    switch (k) {
        case OpKind::Input: os << "Input"; break;
        case OpKind::Perm:  os << "Perm";  break;
        case OpKind::Gemm:  os << "Gemm";  break;
    }
    return os;
}

/**************************************
 *  ENode
 **************************************/
struct ENode {
    OpKind op;
    std::vector<int> children; // eclass ids

    // Input
    Modes input_modes;

    // Perm
    Modes perm_in_modes;
    Modes perm_out_modes;

    // Gemm
    Modes gemm_l_in_modes;
    Modes gemm_r_in_modes;
    Modes gemm_out_modes;
    Modes M, N, K, C; // GEMM decomposition

    /******** Constructors ********/
    static ENode make_input(const Modes &in) {
        ENode n;
        n.op = OpKind::Input;
        n.input_modes = in;
        return n;
    }

    static ENode make_perm(int child, const Modes &in, const Modes &out) {
        ENode n;
        n.op = OpKind::Perm;
        n.children = {child};
        n.perm_in_modes = in;
        n.perm_out_modes = out;
        return n;
    }

    static ENode make_gemm(int lhs, int rhs,
                           const Modes &l_in,
                           const Modes &r_in,
                           const Modes &out,
                           const Modes &M,
                           const Modes &N,
                           const Modes &K,
                           const Modes &C) {
        ENode n;
        n.op = OpKind::Gemm;
        n.children = {lhs, rhs};
        n.gemm_l_in_modes = l_in;
        n.gemm_r_in_modes = r_in;
        n.gemm_out_modes  = out;
        n.M = M; n.N = N; n.K = K; n.C = C;
        return n;
    }

    /******** Equality ********/
    bool operator==(const ENode &o) const {
        if (op != o.op || children != o.children)
            return false;
        switch (op) {
            case OpKind::Input:
                return input_modes == o.input_modes;
            case OpKind::Perm:
                return perm_in_modes == o.perm_in_modes &&
                       perm_out_modes == o.perm_out_modes;
            case OpKind::Gemm:
                return gemm_l_in_modes == o.gemm_l_in_modes &&
                       gemm_r_in_modes == o.gemm_r_in_modes &&
                       gemm_out_modes == o.gemm_out_modes &&
                       M == o.M && N == o.N && K == o.K && C == o.C;
        }
        return false;
    }
};

/**************************************
 *  ENode Hash
 **************************************/
struct ENodeHash {
    std::size_t operator()(const ENode &n) const {
        std::size_t h = std::hash<int>{}(static_cast<int>(n.op));
        auto mix = [&](std::size_t v) {
            h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        };
        for (int c : n.children)
            mix(std::hash<int>{}(c));

        auto hash_modes = [&](const Modes &m) {
            std::size_t hh = 0;
            for (auto x : m)
                hh ^= std::hash<int>{}(x) + 0x9e3779b97f4a7c15ULL +
                      (hh << 6) + (hh >> 2);
            return hh;
        };

        switch (n.op) {
            case OpKind::Input:
                mix(hash_modes(n.input_modes));
                break;
            case OpKind::Perm:
                mix(hash_modes(n.perm_in_modes));
                mix(hash_modes(n.perm_out_modes));
                break;
            case OpKind::Gemm:
                mix(hash_modes(n.gemm_l_in_modes));
                mix(hash_modes(n.gemm_r_in_modes));
                mix(hash_modes(n.gemm_out_modes));
                mix(hash_modes(n.M));
                mix(hash_modes(n.N));
                mix(hash_modes(n.K));
                mix(hash_modes(n.C));
                break;
        }
        return h;
    }
};

/**************************************
 *  EClass
 **************************************/
struct EClass {
    int id;
    std::vector<int> nodes; // indices of enodes
};

/**************************************
 *  GEMM Permutation Closure Utilities
 **************************************/
static const std::vector<std::vector<int>> gemm_perm_group = {
    {0,1,2,3},   // identity
    {1,0,2,3},   // swap M,N
    {0,1,3,2},   // swap K,C
    {1,0,3,2}    // swap both
};

inline Modes permute_modes(const Modes &src, const std::vector<int> &perm) {
    Modes dst;
    dst.reserve(src.size());
    for (int i : perm)
        if (i < (int)src.size())
            dst.push_back(src[i]);
    return dst;
}

/**************************************
 *  EGraph (union–find + memo + closure)
 **************************************/
class EGraph {
public:
    std::vector<ENode> enodes;
    std::vector<EClass> classes;
    std::vector<int> parent;
    std::unordered_map<ENode, int, ENodeHash> memo;

    /******** Union–find ********/
    int find(int x) {
        if (parent[x] != x)
            parent[x] = find(parent[x]);
        return parent[x];
    }

    void merge(int a, int b) {
        a = find(a);
        b = find(b);
        if (a == b) return;
        parent[b] = a;
        classes[a].nodes.insert(classes[a].nodes.end(),
                                classes[b].nodes.begin(),
                                classes[b].nodes.end());
        classes[b].nodes.clear();
    }

    /******** Add Node (with GEMM closure) ********/
    int add(const ENode &raw_node) {
        // canonicalize children
        ENode node = raw_node;
        for (int &c : node.children)
            c = find(c);

        // normal memo check
        auto it = memo.find(node);
        if (it != memo.end())
            return it->second;

        int cid = (int)classes.size();
        int nid = (int)enodes.size();
        enodes.push_back(node);
        classes.push_back(EClass{cid, {nid}});
        parent.push_back(cid);
        memo.emplace(node, cid);

        /***** GEMM permutation closure *****/
        if (node.op == OpKind::Gemm) {
            for (auto &perm : gemm_perm_group) {
                ENode variant = node;
                variant.M = permute_modes(node.M, perm);
                variant.N = permute_modes(node.N, perm);
                variant.K = permute_modes(node.K, perm);
                variant.C = permute_modes(node.C, perm);

                // Optionally permute out_modes if it's tied to M,N,K,C
                variant.gemm_out_modes = permute_modes(node.gemm_out_modes, perm);

                auto jt = memo.find(variant);
                if (jt == memo.end()) {
                    int new_cid = add(variant); // recursive add (canonicalized)
                    merge(cid, new_cid);
                } else {
                    merge(cid, jt->second);
                }
            }
        }

        return cid;
    }

    /******** Debug Print ********/
    void print() const {
        std::cout << "EGraph:\n";
        for (auto &ec : classes) {
            if (ec.nodes.empty()) continue;
            int rep = const_cast<EGraph*>(this)->find(ec.id);
            std::cout << "EClass #" << ec.id;
            if (rep != ec.id) std::cout << " (->" << rep << ")";
            std::cout << ":\n";
            for (int nid : ec.nodes) {
                const auto &n = enodes[nid];
                std::cout << "  " << n.op << " children=[";
                for (auto c : n.children) std::cout << const_cast<EGraph*>(this)->find(c) << " ";
                std::cout << "]";
                if (n.op == OpKind::Input) {
                    std::cout << " modes=";
                    for (auto m : n.input_modes) std::cout << m << " ";
                } else if (n.op == OpKind::Perm) {
                    std::cout << " in=";
                    for (auto m : n.perm_in_modes) std::cout << m << " ";
                    std::cout << " out=";
                    for (auto m : n.perm_out_modes) std::cout << m << " ";
                } else if (n.op == OpKind::Gemm) {
                    std::cout << " M=";
                    for (auto m : n.M) std::cout << m << " ";
                    std::cout << " N=";
                    for (auto m : n.N) std::cout << m << " ";
                    std::cout << " K=";
                    for (auto m : n.K) std::cout << m << " ";
                    std::cout << " C=";
                    for (auto m : n.C) std::cout << m << " ";
                }
                std::cout << "\n";
            }
        }
    }
};

/**************************************
 *  Example Usage
 **************************************/
#ifdef EGRAPH_MAIN
int main() {
    EGraph eg;

    // Simple Inputs
    int A = eg.add(ENode::make_input({0, 1}));
    int B = eg.add(ENode::make_input({2, 3}));

    // GEMM node (will auto-generate permuted equivalents)
    Modes l = {0}, r = {1}, out = {2};
    Modes M = {0}, N = {1}, K = {2}, C = {3};
    int G = eg.add(ENode::make_gemm(A, B, l, r, out, M, N, K, C));

    // Perm node
    int P = eg.add(ENode::make_perm(G, {0, 1}, {1, 0}));

    eg.print();
}
#endif

