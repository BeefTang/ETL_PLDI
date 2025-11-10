Distinct + Range ⇒ Permutation
When you combine:

s.add(Distinct(x0, x1, x2, x3))
s.add(And(xi >= 1, xi <= 4) for each xi)


you’re telling Z3:

“Each xi is one of 1..4, and they’re all different.”

Z3 deduces from this that:

(x\_0,x\_1,x\_2,x\_3) belonging to permutations of [1,2,3,4]

It doesn’t “see” the permutations explicitly, but this effectively defines the feasible domain.
