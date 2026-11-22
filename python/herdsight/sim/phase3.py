"""Phase 3 counting on simulated flights: the same pair evidence and grouping as on real flights.

`phase3_counts` takes a simulated flight (herdsight.sim.flight) and counts it five ways from one
set of registration draws, so the methods differ only in how detections are joined:

  phase1           the Phase 1 method (one-to-one pairing within the gate, greedy joining), as
                   count_flight;
  greedy_fixed / global_fixed        pair evidence with distance scaled by the Phase 1 gate;
  greedy_movement / global_movement  the same gate widened by the grazing allowance.

Simulated detections have no box size or detector score, so the pair model used here is the
position-only one (distance, competition, geotag fallback), fitted on all 20 real verified
flights; simulated flights are separate data, so nothing is fitted on what is scored.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from herdsight.count.clustering import cluster, greedy_join
from herdsight.count.evidence import Candidates, Logistic, MotionModel, features
from herdsight.count.identities import constrained_identities
from herdsight.count.link import match_points
from herdsight.sim.errors import RegistrationModel
from herdsight.sim.flight import GEOTAG_GATE_M, REGISTRATION_GATE_SHARE, Flight, _in_footprint
from herdsight.utils.config import load_config

POSITION_COLUMNS = [0, 1, 2, 5]  # dist2, margin, mutual, geotag (herdsight.count.evidence)


@dataclass
class PositionModels:
    motion: MotionModel
    fixed: Logistic
    movement: Logistic


def phase3_counts(
    flight: Flight,
    median_animal_m: float,
    rng: np.random.Generator,
    models: PositionModels,
    gap_s: float,
    max_exact: int = 60,
    time_limit_s: float = 60.0,
    wide_gate_m: float = 4.0,
    wide_geotag_gate_m: float = 10.0,
) -> dict:
    reg = RegistrationModel.load()
    window = float(load_config("mission").require("herd_mode_window_s"))
    st = flight.stills
    points = [s.points.copy() for s in st]
    for h in (s for s in st if s.kind == "herd"):  # herd-mode merge rule, as count_flight
        for k, s in enumerate(st):
            if s.kind != "survey" or abs(s.t - h.t) > window or not len(points[k]):
                continue
            est = points[k] + s.geotag_error
            inside, _ = _in_footprint(est, h.centre + h.geotag_error, flight.heading, h.footprint)
            points[k] = points[k][~inside]
    counts = [len(p) for p in points]
    c, s_ = np.cos(flight.heading), np.sin(flight.heading)
    rot = np.array([[c, -s_], [s_, c]])
    p1_links = []
    cols: dict[str, list] = {k: [] for k in ("i", "a", "j", "b", "d", "dt", "geo", "fix")}
    for i in range(len(st)):
        for j in range(i + 1, len(st)):
            d = (st[j].centre - st[i].centre) @ rot
            fa = (st[i].footprint[1] + st[j].footprint[1]) / 2
            fc = (st[i].footprint[0] + st[j].footprint[0]) / 2
            if abs(d[0]) >= fa or abs(d[1]) >= fc or not counts[i] or not counts[j]:
                continue
            pi = points[i] + st[i].geotag_error
            residual = reg.outcome(rng)
            if residual is not None:
                pj = points[j] + st[i].geotag_error + rng.normal(0, residual * st[i].gsd, 2)
                gate, wide, geo = REGISTRATION_GATE_SHARE * median_animal_m, wide_gate_m, False
            else:
                pj = points[j] + st[j].geotag_error
                gate, wide, geo = GEOTAG_GATE_M, wide_geotag_gate_m, True
            for a, b, dist in match_points(pi, pj, gate):
                p1_links.append(((i, a), (j, b), dist))
            dm = np.linalg.norm(pi[:, None] - pj[None], axis=2)
            a, b = np.nonzero(dm <= wide)
            cols["i"].append(np.full(len(a), i))
            cols["a"].append(a)
            cols["j"].append(np.full(len(a), j))
            cols["b"].append(b)
            cols["d"].append(dm[a, b])
            cols["dt"].append(np.full(len(a), abs(st[j].t - st[i].t)))
            cols["geo"].append(np.full(len(a), geo))
            cols["fix"].append(np.full(len(a), gate))
    cat = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in cols.items()}
    n = len(cat["i"])
    cand = Candidates(
        i=cat["i"].astype(int),
        a=cat["a"].astype(int),
        j=cat["j"].astype(int),
        b=cat["b"].astype(int),
        d_m=cat["d"].astype(float),
        dt_s=cat["dt"].astype(float),
        geotag=cat["geo"].astype(bool),
        fixed_gate_m=cat["fix"].astype(float),
        size=np.zeros(n),
        low_score=np.zeros(n),
    )
    out = {
        "phase1": len(constrained_identities(counts, p1_links)),
        "detections": int(sum(counts)),
        "separate_passes": bool((cand.dt_s > gap_s).any()) if n else False,
    }
    for kind, model in (("fixed", models.fixed), ("movement", models.movement)):
        if n:
            lo = model.log_odds(features(cand, models.motion, kind)[:, POSITION_COLUMNS])
            pairs = [
                ((int(cand.i[k]), int(cand.a[k])), (int(cand.j[k]), int(cand.b[k])), float(lo[k]))
                for k in range(n)
            ]
        else:
            pairs = []
        out[f"greedy_{kind}"] = len(greedy_join(counts, pairs))
        out[f"global_{kind}"] = len(cluster(counts, pairs, max_exact, time_limit_s).groups)
    return out
