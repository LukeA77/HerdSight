from herdsight.count.identities import constrained_identities
from herdsight.count.link import match_points


def test_match_points_is_one_to_one_and_gated():
    a = [[0, 0], [10, 0], [50, 50]]
    b = [[0.5, 0], [10.4, 0], [200, 200]]
    pairs = match_points(a, b, gate=1.0)
    assert [(i, j) for i, j, _ in pairs] == [(0, 0), (1, 1)]
    assert [round(d, 6) for _, _, d in pairs] == [0.5, 0.4]
    assert match_points([], b, 1.0) == []


def test_plain_chain_of_true_links_forms_one_identity():
    ids = constrained_identities([1, 1, 1], [((0, 0), (1, 0), 0.3), ((1, 0), (2, 0), 0.4)])
    assert ids == [[(0, 0), (1, 0), (2, 0)]]


def test_a_wrong_link_cannot_merge_two_animals_of_the_same_image():
    # Image 0 shows two animals; image 1 shows one of them. A wrong, longer link would join
    # image-0 animal 1 to the same image-1 detection, chaining both image-0 animals together.
    links = [((0, 0), (1, 0), 0.2), ((0, 1), (1, 0), 0.9)]
    ids = constrained_identities([2, 1], links)
    assert sorted(ids) == [[(0, 0), (1, 0)], [(0, 1)]]


def test_identity_count_never_below_the_largest_image_count():
    # Every detection linked to every other: only the per-image rule stops one giant identity.
    counts = [3, 3, 3]
    links = [
        ((i, a), (j, b), 1.0 + 0.01 * (a + b))
        for i in range(3)
        for j in range(i + 1, 3)
        for a in range(3)
        for b in range(3)
    ]
    ids = constrained_identities(counts, links)
    assert len(ids) >= max(counts)
    assert all(len({i for i, _ in v}) == len(v) for v in ids)


def test_unlinked_detections_are_their_own_identities():
    assert constrained_identities([2, 0, 1], []) == [[(0, 0)], [(0, 1)], [(2, 0)]]
