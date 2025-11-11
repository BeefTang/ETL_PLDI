#include <vector>
#include <iostream>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include "dimensions_utils.h"

// --- Step 1: Classify dimension types ---


std::unordered_map<modetype, DimType> categorize_dims(
    const ModesType& I1,
    const ModesType& I2,
    const ModesType& O) {

    std::unordered_set<modetype> sI1(I1.begin(), I1.end());
    std::unordered_set<modetype> sI2(I2.begin(), I2.end());
    std::unordered_set<modetype> sO(O.begin(), O.end());

    std::unordered_map<modetype, DimType> dim_type;

    ModesType all_dims = I1;
    all_dims.insert(all_dims.end(), I2.begin(), I2.end());
    all_dims.insert(all_dims.end(), O.begin(), O.end());

    std::unordered_set<modetype> seen;

    for (const modetype& dim : all_dims) {
        if (seen.count(dim)) continue;
        seen.insert(dim);

        bool in_I1 = sI1.count(dim);
        bool in_I2 = sI2.count(dim);
        bool in_O  = sO.count(dim);

        if (in_I1 && in_I2 && !in_O) dim_type[dim] = K;
        else if (in_I1 && !in_I2 && in_O) dim_type[dim] = M;
        else if (!in_I1 && in_I2 && in_O) dim_type[dim] = N;
        else if (in_I1 && in_I2 && in_O) dim_type[dim] = C;
    }

    return dim_type;
}

// --- Step 2: Get global order per category ---
std::unordered_map<DimType, ModesType> get_global_order(
    const ModesType& I1,
    const ModesType& I2,
    const ModesType& O,
    const std::unordered_map<modetype, DimType>& dim_type) {

    std::unordered_map<DimType, ModesType> global_order;
    std::unordered_set<modetype> seen;

    auto scan = [&](const ModesType& v) {
        for (const auto& d : v) {
            if (seen.count(d)) continue;
            seen.insert(d);
            auto type_it = dim_type.find(d);
            if (type_it != dim_type.end()) {
                global_order[type_it->second].push_back(d);
            }
        }
    };

    scan(I1);
    scan(I2);
    scan(O);

    return global_order;
}

// --- Step 3: Reorder individual inputs according to MKC, NKC, MNC ---
ModesType reorder_by_type(
    const ModesType& input,
    const std::unordered_map<modetype, DimType>& dim_type,
    const std::unordered_map<DimType, ModesType>& global_order,
    const std::vector<DimType>& type_order) {

    std::unordered_set<modetype> input_set(input.begin(), input.end());
    ModesType result;

    for (DimType dtype : type_order) {
        if(global_order.find(dtype) != global_order.end()){
            for (const modetype& d : global_order.at(dtype)) {
                if (input_set.count(d)) {
                    result.push_back(d);
                }
            }
        }
    }

    return result;
}

// --- Step 4: Wrapper function to reorder I1, I2, and O ---
void reorder_all(
    ModesType& I1,
    ModesType& I2,
    ModesType& O) {

    auto dim_type = categorize_dims(I1, I2, O);
    auto global_order = get_global_order(I1, I2, O, dim_type);

    I1 = reorder_by_type(I1, dim_type, global_order, {M, K, C});
    I2 = reorder_by_type(I2, dim_type, global_order, {N, K, C});
    O  = reorder_by_type(O,  dim_type, global_order, {M, N, C});  // or {N, M, C}
}


bool check_global_order(
    const ModesType& vec,
    const std::unordered_map<modetype, DimType>& dim_type,
    const std::unordered_map<DimType, ModesType>& global_order,
    const std::vector<DimType>& expected_type_order) {

    // Build expected ordered list for this vector
    ModesType expected;
    std::unordered_set<modetype> vec_set(vec.begin(), vec.end());

    for (DimType dtype : expected_type_order) { //somttimes there might be dimensions that do not appear here
        if(global_order.find(dtype) != global_order.end()){
            for (const auto& dim : global_order.at(dtype)) {
                if (vec_set.count(dim)) {
                    expected.push_back(dim);
                }
            }

        }
    }

    return vec == expected;
}

bool is_already_ordered(
    const ModesType& I1,
    const ModesType& I2,
    const ModesType& O) {
    //for(auto i : I1){
    //    std::cout<<i<<",";
    //}
    //std::cout<<std::endl;
    //for(auto i : I2){
    //    std::cout<<i<<",";
    //}
    //std::cout<<std::endl;
    //for(auto i : O){
    //    std::cout<<i<<",";
    //}
    //std::cout<<std::endl;

    auto dim_type = categorize_dims(I1, I2, O);
    auto global_order = get_global_order(I1, I2, O, dim_type);
    //for (const auto& pair : global_order) {
    //    std::cout<<pair.first<<", ";
    //    for(auto i : pair.second){
    //        std::cout<<i<<",";
    //    }
    //    std::cout<<std::endl;
    //}

    bool I1_ok = check_global_order(I1, dim_type, global_order, {M, K, C});
    bool I2_ok = check_global_order(I2, dim_type, global_order, {N, K, C});
    bool O_ok1 = check_global_order(O,  dim_type, global_order, {M, N, C});
    bool O_ok2 = check_global_order(O,  dim_type, global_order, {N, M, C});

    return I1_ok && I2_ok && (O_ok1 || O_ok2);
}

std::vector<ModesType> get_M_N_K_C(
    const ModesType &I1,
    const ModesType &I2,
    const ModesType &O) {
        std::unordered_set<modetype> sI1(I1.begin(), I1.end());
        std::unordered_set<modetype> sI2(I2.begin(), I2.end());
        std::unordered_set<modetype> sO(O.begin(), O.end());


        ModesType all_dims = I1;
        all_dims.insert(all_dims.end(), I2.begin(), I2.end());
        all_dims.insert(all_dims.end(), O.begin(), O.end());

        std::unordered_set<modetype> seen;

        ModesType M, N, K, C;

        for (const modetype& dim : all_dims) {
            if (seen.count(dim)) continue;
            seen.insert(dim);

            bool in_I1 = sI1.count(dim);
            bool in_I2 = sI2.count(dim);
            bool in_O  = sO.count(dim);

            if (in_I1 && in_I2 && !in_O) K.push_back(dim);
            else if (in_I1 && !in_I2 && in_O) M.push_back(dim);
            else if (!in_I1 && in_I2 && in_O) N.push_back(dim);
            else if (in_I1 && in_I2 && in_O) C.push_back(dim);
        }

        return {M,N,K,C};
}

// --- check spearman_footrule---


using namespace std;
long long spearman_footrule(const vector<modetype>& a, const vector<modetype>& b) {
    if (a.size() != b.size())
        throw invalid_argument("Spearman footrule: vectors must have the same size.");

    const size_t n = a.size();
    unordered_map<modetype, int> rankA;
    rankA.reserve(n * 2);

    // Map each item -> its rank/position in a
    for (size_t i = 0; i < n; ++i) {
        auto [it, inserted] = rankA.emplace(a[i], (int)i);
        if (!inserted) throw invalid_argument("Duplicate element found in 'a'.");
    }

    long long F = 0;
    for (size_t j = 0; j < n; ++j) {
        auto it = rankA.find(b[j]);
        if (it == rankA.end())
            throw invalid_argument("Element in 'b' not found in 'a'.");
        F += llabs((long long)it->second - (long long)j);
    }
    return F;
}

// Maximum possible footrule distance for permutations of size n: floor(n^2 / 2)
static inline long long spearman_footrule_max(size_t n) {
    long long nn = (long long)n;
    return (nn * nn) / 2; // integer division floors
}

// Normalized footrule in [0,1]
double spearman_footrule_normalized(const vector<modetype>& a, const vector<modetype>& b) {
    // Build maps for membership
    unordered_set<modetype> inA(a.begin(), a.end());
    unordered_set<modetype> inB(b.begin(), b.end());

    // Shared in order of each list (stable order)
    vector<modetype> sharedA; sharedA.reserve(min(a.size(), b.size()));
    for (int x : a) if (inB.count(x)) sharedA.push_back(x);

    vector<modetype> sharedB; sharedB.reserve(min(a.size(), b.size()));
    for (int x : b) if (inA.count(x)) sharedB.push_back(x);


    long long F = spearman_footrule(sharedA, sharedB);
    long long Fmax = spearman_footrule_max(sharedA.size());
    if (Fmax == 0) return 0.0;
    return (double)F / (double)Fmax;
}


