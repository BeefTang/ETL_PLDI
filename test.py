from z3 import Solver, Int, Distinct, Or, If, sat

# --- helper functions ---
def z3_max(*args):
    assert len(args) >= 1
    m = args[0]
    for a in args[1:]:
        m = If(a > m, a, m)
    return m

def z3_min(*args):
    assert len(args) >= 1
    m = args[0]
    for a in args[1:]:
        m = If(a < m, a, m)
    return m
# -------------------------

def permutation_with_rules(elements, rules):
    n = len(elements)
    pos = {x: Int(f"pos_{x}") for x in elements}
    solver = Solver()

    # each pos in [0, n-1] and all distinct
    for x in elements:
        solver.add(pos[x] >= 0, pos[x] < n)
    solver.add(Distinct([pos[x] for x in elements]))

    for (blocks, rightmost) in rules:
        # contiguity for each block
        for block in blocks:
            b_pos = [pos[x] for x in block]
            solver.add(z3_max(*b_pos) - z3_min(*b_pos) + 1 == len(block))

        # rightmost constraint
        if rightmost:
            rightmost = list(rightmost)
            R = [pos[x] for x in rightmost]
            min_R = z3_min(*R)
            for x in elements:
                if x not in rightmost:
                    solver.add(pos[x] < min_R)

        # non-interleaving between different blocks
        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                bi_pos = [pos[x] for x in blocks[i]]
                bj_pos = [pos[x] for x in blocks[j]]
                ai_min, ai_max = z3_min(*bi_pos), z3_max(*bi_pos)
                aj_min, aj_max = z3_min(*bj_pos), z3_max(*bj_pos)
                solver.add(Or(ai_max < aj_min, aj_max < ai_min))

    # enumerate all valid permutations
    results = []
    while solver.check() == sat:
        m = solver.model()
        perm = sorted(elements, key=lambda x: m[pos[x]].as_long())
        results.append(perm)
        # block current solution
        solver.add(Or([pos[x] != m[pos[x]] for x in elements]))

    return results


if __name__ == "__main__":
    S = ['i', 'j', 'c', 'a', 'e']
    rule1 = ([['a'], ['i', 'j', 'c']], ['e'])
    rule2 = ([['a','i'], ['j','c']], ['e'])

    perms = permutation_with_rules(S, [rule1, rule2])
    print("Number of valid permutations:", len(perms))
    for p in perms:
        print(p)

