import numpy as np

from herdsight.evaluate.herds import groups, score_image


def box(x: float, y: float, half: float = 5.0) -> list[float]:
    return [x - half, y - half, x + half, y + half]


def test_groups_use_single_linkage_and_a_minimum_size():
    c = [[0, 0], [8, 0], [16, 0], [100, 0], [108, 0]]  # a chain of 3, and a pair
    g = groups(c, link_m=10, min_size=3)
    assert [sorted(x.tolist()) for x in g] == [[0, 1, 2]]
    assert groups([[0, 0]], 10, 3) == []


def test_one_detection_finds_a_herd_but_does_not_group_it():
    truth = [box(0, 0), box(8, 0), box(16, 0)]
    r = score_image(truth, [box(8, 0)], gsd_m=1.0, link_m=10, min_animals=3)
    assert (r.true_herds, r.found, r.grouped, r.false_triggers) == (1, 1, 0, 0)


def test_detected_herd_is_grouped_and_a_far_group_is_a_false_trigger():
    truth = [box(0, 0), box(8, 0), box(16, 0)]
    preds = truth + [box(500, 0), box(505, 0), box(510, 0)]
    r = score_image(truth, preds, gsd_m=1.0, link_m=10, min_animals=3)
    assert (r.found, r.grouped, r.predicted_herds, r.false_triggers) == (1, 1, 2, 1)


def test_ground_sample_distance_scales_the_link_distance():
    truth = [box(0, 0), box(8, 0), box(16, 0)]  # 8 px apart
    assert score_image(truth, [], gsd_m=1.0, link_m=10, min_animals=3).true_herds == 1
    assert score_image(truth, [], gsd_m=2.0, link_m=10, min_animals=3).true_herds == 0


def test_empty_inputs():
    r = score_image(np.zeros((0, 4)), np.zeros((0, 4)), 1.0, 10, 3)
    assert (r.true_herds, r.predicted_herds) == (0, 0)
