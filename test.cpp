#include "esaturation.h"
#include "ETL.h"
#include <iostream>
#include <memory>

using namespace ETL;

int main() {
    // Step 1: Create a Context (mode mapping etc.)
    Context ctx;
    Modes M = {0}, N = {1}, K = {2}; // example modes

    // Step 2: Build a simple ETL expression: Perm(Gemm(Input A, Input B))
    auto A = std::make_shared<Input>(M, ctx, nullptr, "A");
    auto B = std::make_shared<Input>(N, ctx, nullptr, "B");
    auto gemm = std::make_shared<Gemm>(K, ctx, A, B, "Gemm");
    auto perm = std::make_shared<Perm>(M, ctx, gemm, "Perm");

    // Step 3: Initialize EGraph
    EGraph egraph;

    // Add nodes to EGraph
    auto id_A = egraph.add(ENode("Input", {"A"}));
    auto id_B = egraph.add(ENode("Input", {"B"}));
    auto id_gemm = egraph.add(ENode("Gemm", {id_A, id_B}));
    auto id_perm = egraph.add(ENode("Perm", {id_gemm}));

    // Step 4: Define rewrite rules
    std::vector<RewriteRule> rules;

    // Example rule: remove redundant permutation
    // Perm(Gemm(A, B)) -> Gemm(A, B)
    rules.push_back({
        "perm_gemm_simplify",
        Pattern("Perm", {Pattern("Gemm", {Pattern("Input"), Pattern("Input")})}),
        Pattern("Gemm", {Pattern("Input"), Pattern("Input")})
    });

    // Step 5: Run E-saturation
    egraph.run_saturation(rules, /*max_iterations=*/5);

    // Step 6: Extract best expression from e-graph
    auto extractor = Extractor(egraph);
    auto best = extractor.extract(id_perm);

    std::cout << "Best extracted expression: " << best.to_string() << std::endl;

    return 0;
}

