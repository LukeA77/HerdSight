"""Gazebo world from a simulated world (python/herdsight/sim/gazebo.py)."""

import xml.etree.ElementTree as ET

import numpy as np

from herdsight.sim.flight import build_world
from herdsight.sim.gazebo import world_sdf
from herdsight.sim.herds import Camp


def test_world_has_every_animal_labelled_and_the_fence():
    camp = Camp.rectangle(200, 150, "River camp")
    world = build_world(camp, 25, np.random.default_rng(5), calf_share=0.3, trees_per_ha=1.0)
    root = ET.fromstring(world_sdf(world))
    models = root.find("world").findall("model")
    animals = [m for m in models if m.get("name").startswith("animal_")]
    assert len(animals) == world.herd.n
    labels = [int(m.find("plugin/label").text) for m in animals]
    assert labels == list(range(1, world.herd.n + 1))  # one label per animal
    assert sum(m.get("name").startswith("fence_") for m in models) == 4
    assert sum(m.get("name").startswith("tree_") for m in models) == len(world.trees)
    for m, (x, y) in zip(animals, world.herd.pos, strict=True):
        px, py = (float(v) for v in m.find("pose").text.split()[:2])
        assert abs(px - x) < 1e-3 and abs(py - y) < 1e-3
