#pragma once
#include <string>
#include <vector>
#include <utility>

struct ein_exp {
    std::string expression;
    std::vector<int64_t> extents;
    std::vector<std::pair<int, int>> path;
};
std::vector<ein_exp> parseFile(const std::string& filename);