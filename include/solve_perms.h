#ifndef SLOVE_Z3_H
#define SLOVE_Z3_H

namespace ETL{

using UnorderedModes=std::unordered_set<ModeType>;
std::vector<Modes> permutation_with_rules( const Modes &elements, const std::vector<std::pair<std::vector<UnorderedModes>, UnorderedModes>> &rules);//one rule is for up, one is for below
};
#endif

