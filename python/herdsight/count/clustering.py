"""Group the detections of a flight into animals at once: correlation clustering (D-9).

Nodes are detections, written (still index, detection index). Each candidate pair carries a
weight, the log odds that the two are one animal (herdsight.count.evidence); pairs that are not
candidates weigh 0. A grouping is scored by the total weight of the pairs it puts together, and
two detections from one still are never put together (a still cannot show an animal twice). The
best grouping maximises the score; the flight count is the number of groups.

Two solvers:

* `solve_exact`: the integer program with one 0/1 variable per pair of nodes and the triangle
  inequalities (x_uv + x_vw - x_uw <= 1), solved by HiGHS through scipy.optimize.milp. Only for
  small parts (configs/phase3.yaml, exact_solver); it also measures the heuristic's gap.
* `solve_heuristic`: greedy additive edge contraction (join the two groups with the largest
  positive total weight between them, while allowed), then local moves (move one detection to
  the group, or to a group of its own, that raises the score most) until nothing improves. This
  is the method ported to C++.

`cluster` splits the graph into parts joined by positive weights (an optimal grouping never
joins parts with no positive weight between them), solves each part exactly when it is small
enough and the solver proves optimality in time, and heuristically otherwise. `greedy_join` is
the greedy baseline with the same weights: pairs joined in order of decreasing weight while
positive and allowed (the Phase 1 joining rule, D-30, with the new evidence; H23).
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from herdsight.count.identities import constrained_identities

Node = tuple[int, int]


@dataclass
class Result:
    groups: list[list[Node]]
    exact_parts: int = 0
    heuristic_parts: int = 0
    nodes_in_exact_parts: int = 0
    heuristic_gap: float = 0.0  # summed over exactly solved parts: optimum - heuristic score


def score(groups: list[list[Node]], w: dict[frozenset, float]) -> float:
    return float(
        sum(w.get(frozenset((u, v)), 0.0) for g in groups for u, v in itertools.combinations(g, 2))
    )


def _weights(pairs: list[tuple[Node, Node, float]]) -> dict[frozenset, float]:
    w: dict[frozenset, float] = {}
    for u, v, x in pairs:
        if u[0] != v[0]:
            k = frozenset((u, v))
            w[k] = w.get(k, 0.0) + float(x)
    return w


def solve_heuristic(nodes: list[Node], w: dict[frozenset, float]) -> list[list[Node]]:
    # greedy additive edge contraction
    gid = {n: k for k, n in enumerate(nodes)}
    members: dict[int, list[Node]] = {k: [n] for k, n in enumerate(nodes)}
    stills: dict[int, set[int]] = {k: {n[0]} for k, n in enumerate(nodes)}
    between: dict[int, dict[int, float]] = defaultdict(dict)
    for k, x in w.items():
        u, v = tuple(k)
        if u in gid and v in gid:
            a, b = gid[u], gid[v]
            between[a][b] = between[a].get(b, 0.0) + x
            between[b][a] = between[b].get(a, 0.0) + x
    while True:
        best, pair = 0.0, None
        for a, row in between.items():
            for b, x in row.items():
                if a < b and x > best and not (stills[a] & stills[b]):
                    best, pair = x, (a, b)
        if pair is None:
            break
        a, b = pair
        if len(members[a]) < len(members[b]):
            a, b = b, a
        members[a] += members.pop(b)
        stills[a] |= stills.pop(b)
        for c, x in between.pop(b).items():
            between[c].pop(b, None)
            if c != a:
                between[a][c] = between[a].get(c, 0.0) + x
                between[c][a] = between[a][c]
        between[a].pop(a, None)
    groups = list(members.values())
    return _local_moves(groups, w)


def _local_moves(groups: list[list[Node]], w: dict[frozenset, float]) -> list[list[Node]]:
    where = {n: k for k, g in enumerate(groups) for n in g}
    groups = [list(g) for g in groups]
    nbr: dict[Node, list[tuple[Node, float]]] = defaultdict(list)
    for k, x in w.items():
        u, v = tuple(k)
        nbr[u].append((v, x))
        nbr[v].append((u, x))
    improved = True
    rounds = 0
    while improved and rounds < 100:
        improved, rounds = False, rounds + 1
        for n in list(where):
            here = where[n]
            gain_to: dict[int, float] = defaultdict(float)
            for m, x in nbr[n]:
                gain_to[where[m]] += x
            stay = gain_to.get(here, 0.0)
            best_g, best_gain = None, 0.0
            if stay < -1e-9:  # alone is better than here
                best_g, best_gain = -1, -stay
            for g, x in gain_to.items():
                if g == here or any(m[0] == n[0] for m in groups[g]):
                    continue
                if x - stay > best_gain + 1e-9:
                    best_g, best_gain = g, x - stay
            if best_g is None:
                continue
            groups[here].remove(n)
            if best_g == -1:
                groups.append([n])
                where[n] = len(groups) - 1
            else:
                groups[best_g].append(n)
                where[n] = best_g
            improved = True
    return [sorted(g) for g in groups if g]


def solve_exact(
    nodes: list[Node], w: dict[frozenset, float], time_limit_s: float
) -> list[list[Node]] | None:
    """Optimal grouping of a small part, or None if HiGHS does not prove optimality in time."""
    n = len(nodes)
    if n == 1:
        return [list(nodes)]
    idx = {}
    pairs = list(itertools.combinations(range(n), 2))
    for k, (a, b) in enumerate(pairs):
        idx[(a, b)] = k
    c = np.array([-w.get(frozenset((nodes[a], nodes[b])), 0.0) for a, b in pairs])
    upper = np.array([0.0 if nodes[a][0] == nodes[b][0] else 1.0 for a, b in pairs])
    rows, cols, vals = [], [], []
    r = 0
    for a, b, d in itertools.combinations(range(n), 3):
        ab, bd, ad = idx[(a, b)], idx[(b, d)], idx[(a, d)]
        for p, q, s in ((ab, bd, ad), (ab, ad, bd), (ad, bd, ab)):
            rows += [r, r, r]
            cols += [p, q, s]
            vals += [1.0, 1.0, -1.0]
            r += 1
    cons = []
    if r:
        a_mat = coo_matrix((vals, (rows, cols)), shape=(r, len(pairs))).tocsr()
        cons.append(LinearConstraint(a_mat, -np.inf, 1.0))
    res = milp(
        c,
        constraints=cons,
        integrality=np.ones(len(pairs)),
        bounds=Bounds(np.zeros(len(pairs)), upper),
        options={"time_limit": time_limit_s},
    )
    if res.status != 0 or res.x is None:
        return None
    x = np.round(res.x).astype(int)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for k, (a, b) in enumerate(pairs):
        if x[k]:
            parent[find(a)] = find(b)
    groups: dict[int, list[Node]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(nodes[i])
    return [sorted(g) for g in groups.values()]


def _parts(nodes: list[Node], w: dict[frozenset, float]) -> list[list[Node]]:
    parent = {u: u for u in nodes}

    def find(u: Node) -> Node:
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    for k, x in w.items():
        if x > 0:
            u, v = tuple(k)
            parent[find(u)] = find(v)
    out: dict[Node, list[Node]] = defaultdict(list)
    for u in nodes:
        out[find(u)].append(u)
    return list(out.values())


def cluster(
    counts: list[int],
    pairs: list[tuple[Node, Node, float]],
    max_exact: int = 60,
    time_limit_s: float = 60.0,
) -> Result:
    """Group all detections of a flight. `counts[i]` detections in still i; `pairs` are
    (node, node, log odds). Parts of at most `max_exact` detections are solved exactly."""
    nodes = [(i, k) for i, n in enumerate(counts) for k in range(n)]
    w = _weights(pairs)
    res = Result(groups=[])
    parts = _parts(nodes, w)
    part_of = {u: p for p, part in enumerate(parts) for u in part}
    by_part: dict[int, dict[frozenset, float]] = defaultdict(dict)
    for k, x in w.items():
        u, v = tuple(k)
        if part_of[u] == part_of[v]:
            by_part[part_of[u]][k] = x
    for p, part in enumerate(parts):
        pw = by_part.get(p, {})
        heur = solve_heuristic(part, pw)
        exact = solve_exact(part, pw, time_limit_s) if 1 < len(part) <= max_exact else None
        if exact is not None:
            res.groups += exact
            res.exact_parts += 1
            res.nodes_in_exact_parts += len(part)
            res.heuristic_gap += score(exact, pw) - score(heur, pw)
        else:
            res.groups += heur
            if len(part) > 1:
                res.heuristic_parts += 1
    res.groups.sort()
    return res


def greedy_join(counts: list[int], pairs: list[tuple[Node, Node, float]]) -> list[list[Node]]:
    """Greedy baseline with the same weights: join pairs of positive weight, strongest first,
    never two detections of one still in a group (constrained_identities)."""
    w = _weights(pairs)
    links = [(*tuple(k), -x) for k, x in w.items() if x > 0]
    return constrained_identities(counts, links)
