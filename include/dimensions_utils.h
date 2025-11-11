#ifndef DIMENSIONS_UTILS_H
#define DIMENSIONS_UTILS_H

#include <unordered_map>
#include "context.h"
enum DimType { K, M, N, C };

//TODO: remove modetype
using modetype = ETL::ModeType;
using ModesType = ETL::Modes;

std::unordered_map<modetype, DimType> categorize_dims(
    const ModesType& I1,
    const ModesType& I2,
    const ModesType& O);
bool is_already_ordered(
    const ModesType& I1,
    const ModesType& I2,
    const ModesType& O);
void reorder_all(
    ModesType& I1,
    ModesType& I2,
    ModesType& O);
std::vector<ModesType> get_M_N_K_C(
    const ModesType &I1,
    const ModesType &I2,
    const ModesType &O);

double spearman_footrule_normalized(const std::vector<modetype>& a, const std::vector<modetype>& b);
#endif
