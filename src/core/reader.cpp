
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>  // for std::remove
#include "reader.h"

std::vector<ein_exp> parseFile(const std::string& filename) {
    std::vector<ein_exp> results;
    std::ifstream infile(filename);
    std::string line;

    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string expr_str, vec_str, pair_str;
        if (!(iss >> expr_str >> vec_str >> pair_str)) continue;

        // Remove surrounding quotes
        auto strip_quotes = [](std::string& s) {
            if (!s.empty() && s.front() == '"' && s.back() == '"') {
                s = s.substr(1, s.size() - 2);
            }
        };
        strip_quotes(expr_str);
        strip_quotes(vec_str);
        strip_quotes(pair_str);

        ein_exp exp;
        exp.expression = expr_str;

        // Parse extents
        std::istringstream vec_stream(vec_str);
        std::string val;
        while (std::getline(vec_stream, val, ',')) {
            exp.extents.push_back(std::stoi(val));
        }

        // Parse path (supports multiple pairs like "(0,1)(2,3)")
        for (size_t i = 0; i < pair_str.size();) {
            if (pair_str[i] == '(') {
                int a, b;
                ++i;
                size_t comma = pair_str.find(',', i);
                size_t end = pair_str.find(')', comma);
                a = std::stoi(pair_str.substr(i, comma - i));
                b = std::stoi(pair_str.substr(comma + 1, end - comma - 1));
                exp.path.emplace_back(a, b);
                i = end + 1;
            } else {
                ++i;
            }
        }

        results.push_back(std::move(exp));
    }

    return results;
}
