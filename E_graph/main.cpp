#include "egraph.hpp"

int main() {
    EGraph eg;

    // Build expression: relu(add(matmul(A,B), bias))
    auto A    = Expr::make("A");
    auto B    = Expr::make("B");
    auto bias = Expr::make("bias");

    auto mm   = Expr::make("matmul", { A, B });
    auto add  = Expr::make("add", { mm, bias });
    auto relu = Expr::make("relu", { add });

    EClassId root = eg.add_expr(relu);
    std::cout << "Initial e-graph:\n";
    eg.dump();

    // Define fusion rewrite:
    // relu(add(matmul(?a, ?b), ?bias)) -> fused_matmul_add_relu(?a, ?b, ?bias)
    RewriteRule fuse_relu_add_matmul(
        "fuse-matmul-add-relu",
        Pattern::Node("relu", {
            Pattern::Node("add", {
                Pattern::Node("matmul", {
                    Pattern::Var("?a"),
                    Pattern::Var("?b")
                }),
                Pattern::Var("?bias")
            })
        }),
        Pattern::Node("fused_matmul_add_relu", {
            Pattern::Var("?a"),
            Pattern::Var("?b"),
            Pattern::Var("?bias")
        })
    );

    std::vector<RewriteRule> rules = { fuse_relu_add_matmul };

    // Run saturation
    saturate(eg, rules, /*iter_limit=*/5);

    std::cout << "\nAfter saturation:\n";
    eg.dump();

    return 0;
}

