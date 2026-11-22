import itertools

import numpy as np

from herdsight.count.clustering import (
    _weights,
    cluster,
    greedy_join,
    score,
    solve_exact,
    solve_heuristic,
)


def _brute_force(nodes, w):
    """Best grouping of a tiny node set by trying every partition."""

    def partitions(items):
        if not items:
            yield []
            return
        first, rest = items[0], items[1:]
        for p in partitions(rest):
            for k in range(len(p)):
                yield p[:k] + [[first, *p[k]]] + p[k + 1 :]
            yield [[first], *p]

    best_s = -np.inf
    for p in partitions(list(nodes)):
        if any(len({n[0] for n in g}) < len(g) for g in p):
            continue
        s = score(p, w)
        if s > best_s:
            best_s = s
    return best_s


def test_one_animal_seen_in_three_stills():
    pairs = [((0, 0), (1, 0), 3.0), ((1, 0), (2, 0), 3.0), ((0, 0), (2, 0), 2.0)]
    r = cluster([1, 1, 1], pairs)
    assert r.groups == [[(0, 0), (1, 0), (2, 0)]]


def test_never_two_detections_of_one_still_together():
    # (0,0) and (0,1) both like (1,0); only one can join it
    pairs = [((0, 0), (1, 0), 2.0), ((0, 1), (1, 0), 1.5)]
    for groups in (cluster([2, 1], pairs).groups, greedy_join([2, 1], pairs)):
        for g in groups:
            assert len({n[0] for n in g}) == len(g)
        assert len(groups) == 2


def test_global_fixes_a_chain_that_greedy_gets_wrong():
    # greedy joins the strongest pair a0-b0 first, then cannot add c0 whose evidence ties it to
    # a0 and b1 together; global grouping weighs the whole picture
    a0, b0, b1, c0 = (0, 0), (1, 0), (1, 1), (2, 0)
    pairs = [(a0, b0, 2.0), (a0, c0, 1.5), (b1, c0, 1.5), (a0, b1, 1.4), (b0, c0, -4.0)]
    w = _weights(pairs)
    g = greedy_join([1, 2, 1], pairs)
    c = cluster([1, 2, 1], pairs).groups
    assert score(c, w) >= score(g, w)
    assert score(c, w) == _brute_force([a0, b0, b1, c0], w)


def test_exact_matches_brute_force_on_random_small_graphs():
    rng = np.random.default_rng(3)
    for _ in range(25):
        counts = list(rng.integers(1, 3, size=3))
        nodes = [(i, k) for i, n in enumerate(counts) for k in range(n)]
        pairs = [
            (u, v, float(rng.normal()))
            for u, v in itertools.combinations(nodes, 2)
            if u[0] != v[0] and rng.random() < 0.8
        ]
        w = _weights(pairs)
        exact = solve_exact(nodes, w, 10.0)
        assert exact is not None
        assert abs(score(exact, w) - _brute_force(nodes, w)) < 1e-9
        heur = solve_heuristic(nodes, w)
        assert score(heur, w) <= score(exact, w) + 1e-9
        assert sorted(n for g in heur for n in g) == sorted(nodes)


def test_cluster_reports_gap_and_part_counts():
    rng = np.random.default_rng(5)
    counts = [3, 3, 3, 3]
    nodes = [(i, k) for i, n in enumerate(counts) for k in range(n)]
    pairs = [
        (u, v, float(rng.normal(0.3))) for u, v in itertools.combinations(nodes, 2) if u[0] != v[0]
    ]
    r = cluster(counts, pairs, max_exact=60)
    assert r.exact_parts >= 1 and r.heuristic_gap >= -1e-9
    assert sorted(n for g in r.groups for n in g) == nodes
    r2 = cluster(counts, pairs, max_exact=1)  # forced heuristic
    assert r2.exact_parts == 0 and r2.heuristic_parts >= 1


def test_negative_weights_keep_detections_apart():
    r = cluster([1, 1], [((0, 0), (1, 0), -1.0)])
    assert len(r.groups) == 2
    assert len(greedy_join([1, 1], [((0, 0), (1, 0), -1.0)])) == 2
