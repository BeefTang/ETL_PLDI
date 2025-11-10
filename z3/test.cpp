#include <z3++.h>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>

using namespace z3;

// --- helper functions ---
expr z3_max(const std::vector<expr> &args) {
    assert(!args.empty());
    expr m = args[0];
    for (size_t i = 1; i < args.size(); ++i)
        m = ite(args[i] > m, args[i], m);
    return m;
}

expr z3_min(const std::vector<expr> &args) {
    assert(!args.empty());
    expr m = args[0];
    for (size_t i = 1; i < args.size(); ++i)
        m = ite(args[i] < m, args[i], m);
    return m;
}

std::vector<std::vector<std::string>> permutation_with_rules(
    const std::vector<std::string> &elements,
    const std::vector<std::pair<std::vector<std::vector<std::string>>,
                                std::vector<std::string>>> &rules)
{
    context c;
    solver s(c);
    int n = elements.size();

    // build position variables
    std::map<std::string, expr> pos;
    for (auto &x : elements)
        pos.emplace(x, c.int_const(("pos_" + x).c_str()));

    // each pos in [0, n-1] and all distinct
    for (auto &x : elements)
        s.add(pos.at(x) >= 0 && pos.at(x) < n);

    expr_vector distincts(c);
    for (auto &x : elements)
        distincts.push_back(pos.at(x));
    s.add(distinct(distincts));

    // apply rules
    for (auto &[blocks, rightmost] : rules) {
        for (auto &block : blocks) {
            std::vector<expr> b_pos;
            for (auto &x : block) b_pos.push_back(pos.at(x));
            s.add(z3_max(b_pos) - z3_min(b_pos) + 1 == (int)block.size());
        }

        if (!rightmost.empty()) {
            std::vector<expr> R;
            for (auto &x : rightmost) R.push_back(pos.at(x));
            expr min_R = z3_min(R);
            for (auto &x : elements) {
                if (std::find(rightmost.begin(), rightmost.end(), x) == rightmost.end())
                    s.add(pos.at(x) < min_R);
            }
        }

        for (size_t i = 0; i < blocks.size(); ++i) {
            for (size_t j = i + 1; j < blocks.size(); ++j) {
                std::vector<expr> bi_pos, bj_pos;
                for (auto &x : blocks[i]) bi_pos.push_back(pos.at(x));
                for (auto &x : blocks[j]) bj_pos.push_back(pos.at(x));
                expr ai_min = z3_min(bi_pos), ai_max = z3_max(bi_pos);
                expr aj_min = z3_min(bj_pos), aj_max = z3_max(bj_pos);
                s.add((ai_max < aj_min) || (aj_max < ai_min));
            }
        }
    }

    std::vector<std::vector<std::string>> results;
    while (s.check() == sat) {
        model m = s.get_model();
        std::vector<std::pair<int, std::string>> sorted;
        for (auto &x : elements)
            sorted.emplace_back(m.eval(pos.at(x)).get_numeral_int(), x);
        std::sort(sorted.begin(), sorted.end());

        std::vector<std::string> perm;
        for (auto &[_, x] : sorted)
            perm.push_back(x);

        results.push_back(perm);

        expr_vector diff(c);
        for (auto &x : elements)
            diff.push_back(pos.at(x) != m.eval(pos.at(x)));
        s.add(mk_or(diff));
    }

    return results;
}

int main() {
    std::vector<std::string> S = {"i", "j", "c", "a", "e"};
    std::vector<std::pair<std::vector<std::vector<std::string>>, std::vector<std::string>>> rules = {
        {{{"a"}, {"i","j","c"}}, {"e"}},
        {{{"a","i"}, {"j","c"}}, {"e"}}
    };

    auto perms = permutation_with_rules(S, rules);
    std::cout << "Number of valid permutations: " << perms.size() << "\n";
    for (auto &p : perms) {
        for (auto &x : p) std::cout << x << " ";
        std::cout << "\n";
    }
}

