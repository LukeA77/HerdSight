"""Group pairwise animal links into identities without chain errors (D-9).

Pairwise links between detections in overlapping images can chain: if animal A in image 1 is
linked to B in image 2, and B (wrongly) to C in image 3, plain connected components put A and C
together even when they are two animals standing side by side, and a few wrong links merge a
whole herd into one "animal". A single image can never show the same animal twice, so:

`constrained_identities` joins links shortest first (Kruskal order) and refuses any join that
would place two detections from the same image in one identity. Every identity therefore has at
most one member per image, and the number of identities is never below the largest single-image
count. This is the greedy stage; D-9's global assignment replaces it in Phase 3, and the
greedy result is kept as its baseline.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

Member = tuple[int, int]  # (image index, detection index)


def constrained_identities(
    counts: Sequence[int], links: Iterable[tuple[Member, Member, float]]
) -> list[list[Member]]:
    """Identities (lists of members) from per-image detection counts and weighted links.

    `links` are (member_a, member_b, distance); smaller distances are joined first."""
    parent: dict[Member, Member] = {}
    images: dict[Member, set[int]] = {}
    members: dict[Member, list[Member]] = {}
    for i, n in enumerate(counts):
        for k in range(n):
            m = (i, k)
            parent[m], images[m], members[m] = m, {i}, [m]

    def find(x: Member) -> Member:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, _ in sorted(links, key=lambda t: t[2]):
        ra, rb = find(tuple(a)), find(tuple(b))  # type: ignore[arg-type]
        if ra == rb or images[ra] & images[rb]:
            continue  # already together, or the join would put one image in twice
        if len(members[ra]) < len(members[rb]):
            ra, rb = rb, ra
        parent[rb] = ra
        images[ra] |= images.pop(rb)
        members[ra] += members.pop(rb)
    return sorted((sorted(v) for v in members.values()), key=lambda v: v[0])
