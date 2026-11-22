"""Cattle herds in a simulated camp: placement and movement (Phase 5 plan, step 1; D-11).

Positions are in metres in a local east-north frame. A camp is a fence polygon. Animals form
groups; each group is grazing, walking or resting, and switches state after a random bout:
  grazing  each animal drifts by a 2-D random walk with the measured diffusion coefficient
           (configs/sim_worlds.yaml, grazing_diffusion_m2_per_s);
  walking  the group moves at walking speed towards a target point in the camp, keeping its shape;
  resting  no movement.
Animals further from their group's centre than the group's placed spread (at least the cohesion
radius) drift back; animals closer
than the minimum spacing are pushed apart; nobody crosses the fence. When a drone is closer than
the reaction radius and lower than the reaction height, animals under it walk away from it
(swept from no reaction to reaction at survey height).

Placement reproduces the measured nearest-neighbour distances (results/data/herd_stats.json):
each new animal is placed next to a random member of its group at a distance drawn from the
measured distribution. All randomness comes from one numpy Generator, so a seed reproduces a run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from herdsight.utils.config import REPO_ROOT, load_config

STATES = ("graze", "walk", "rest")


@dataclass(frozen=True)
class Camp:
    """A fenced camp: polygon vertices (x, y) in metres, in order."""

    polygon: NDArray[np.float64]
    name: str = "camp"

    @staticmethod
    def rectangle(width: float, height: float, name: str = "camp") -> Camp:
        return Camp(np.array([[0, 0], [width, 0], [width, height], [0, height]], float), name)

    def contains(self, pts: ArrayLike) -> NDArray[np.bool_]:
        """Even-odd ray test for each point."""
        p = np.atleast_2d(np.asarray(pts, float))
        x, y = p[:, 0], p[:, 1]
        poly = self.polygon
        inside = np.zeros(len(p), bool)
        for (x0, y0), (x1, y1) in zip(poly, np.roll(poly, -1, axis=0), strict=True):
            crosses = (y0 > y) != (y1 > y)
            with np.errstate(divide="ignore", invalid="ignore"):
                xc = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            inside ^= crosses & (x < xc)
        return inside

    def area_m2(self) -> float:
        x, y = self.polygon[:, 0], self.polygon[:, 1]
        return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    def random_point(self, rng: np.random.Generator, margin: float = 0.0) -> NDArray[np.float64]:
        lo, hi = self.polygon.min(0) + margin, self.polygon.max(0) - margin
        for _ in range(10000):
            p = rng.uniform(lo, hi)
            if self.contains(p)[0]:
                return p
        raise ValueError(f"no point found inside {self.name}")


@dataclass(frozen=True)
class HerdSettings:
    """One draw of the herd assumptions (ranges in configs/sim_worlds.yaml)."""

    grazing_diffusion: float
    walking_speed: float
    grazing_share: float
    walking_share: float
    bout_s: float
    groups: int
    cohesion_radius: float
    min_spacing: float
    reaction_height: float
    reaction_radius: float
    reaction_speed: float
    nn_quantiles: tuple[tuple[float, float], ...]  # (probability, metres)


def settings_from_config(rng: np.random.Generator) -> HerdSettings:
    """Draw herd settings: swept ranges uniformly, fixed values as they are."""
    cfg = load_config("sim_worlds")

    def u(key: str) -> float:
        lo, hi = cfg.require(key)
        return float(rng.uniform(lo, hi))

    stats = json.loads((REPO_ROOT / cfg.require("nearest_neighbour_source")).read_text())
    nn = tuple(sorted((float(k) / 100, float(v)) for k, v in stats["nearest_neighbour_m"].items()))
    lo_g, hi_g = cfg.require("groups_per_camp")
    return HerdSettings(
        grazing_diffusion=float(cfg.require("grazing_diffusion_m2_per_s")),
        walking_speed=u("walking_speed_m_s"),
        grazing_share=u("grazing_share"),
        walking_share=u("walking_share"),
        bout_s=60 * u("state_bout_minutes"),
        groups=int(rng.integers(lo_g, hi_g + 1)),
        cohesion_radius=float(cfg.require("cohesion_radius_m")),
        min_spacing=float(cfg.require("min_spacing_m")),
        reaction_height=u("drone_reaction_height_m"),
        reaction_radius=float(cfg.require("drone_reaction_radius_m")),
        reaction_speed=float(cfg.require("drone_reaction_speed_m_s")),
        nn_quantiles=nn,
    )


def draw_nn_distance(q: tuple[tuple[float, float], ...], rng: np.random.Generator) -> float:
    """A nearest-neighbour distance from the measured quantiles (linear between them, with the
    tails extended linearly to probability 0 and 1)."""
    p = np.array([x for x, _ in q])
    v = np.array([y for _, y in q])
    lo = max(v[0] - (v[1] - v[0]) * p[0] / (p[1] - p[0]), 0.0)
    hi = v[-1] + (v[-1] - v[-2]) * (1 - p[-1]) / (p[-1] - p[-2])
    return float(np.interp(rng.random(), np.r_[0.0, p, 1.0], np.r_[lo, v, hi]))


@dataclass
class Herd:
    camp: Camp
    settings: HerdSettings
    pos: NDArray[np.float64]  # (N, 2)
    group: NDArray[np.int64]  # (N,)
    state: list[str]  # per group
    target: NDArray[np.float64]  # (groups, 2) walking targets
    t: float = 0.0
    bout_left: list[float] = field(default_factory=list)
    home_radius: list[float] = field(default_factory=list)  # per group: cohesion radius

    @property
    def n(self) -> int:
        return len(self.pos)

    def centres(self) -> NDArray[np.float64]:
        return np.array([self.pos[self.group == g].mean(0) for g in range(len(self.state))])

    def step(
        self, dt: float, rng: np.random.Generator, drone: ArrayLike | None = None
    ) -> NDArray[np.bool_]:
        """Advance by dt seconds. `drone` is (x, y, height above ground) or None.
        Returns which animals reacted to the drone in this step."""
        s = self.settings
        old = self.pos.copy()
        move = np.zeros_like(self.pos)
        centres = self.centres()
        for g, st in enumerate(self.state):
            m = self.group == g
            self.bout_left[g] -= dt
            if self.bout_left[g] <= 0:
                self.state[g] = _draw_state(s, rng)
                self.bout_left[g] = rng.exponential(s.bout_s)
                self.target[g] = self.camp.random_point(rng, margin=5.0)
                st = self.state[g]
            if st == "graze":
                move[m] += rng.normal(0, np.sqrt(2 * s.grazing_diffusion * dt), (m.sum(), 2))
            elif st == "walk":
                d = self.target[g] - centres[g]
                dist = np.linalg.norm(d)
                if dist < 2.0:
                    self.state[g], self.bout_left[g] = "graze", rng.exponential(s.bout_s)
                else:
                    move[m] += d / dist * min(s.walking_speed * dt, dist)
            off = self.pos[m] - centres[g]
            r = np.linalg.norm(off, axis=1)
            far = r > (self.home_radius[g] if self.home_radius else s.cohesion_radius)
            if far.any():  # drift back at grazing pace (0.1 m/s)
                idx = np.flatnonzero(m)[far]
                move[idx] -= off[far] / r[far, None] * min(0.1 * dt, 1.0)
        reacted = np.zeros(self.n, bool)
        if drone is not None:
            dx, dy, dh = (float(v) for v in drone)
            if dh < s.reaction_height:
                away = self.pos - np.array([dx, dy])
                r = np.linalg.norm(away, axis=1)
                reacted = r < s.reaction_radius
                if reacted.any():
                    move[reacted] += (
                        away[reacted] / np.maximum(r[reacted, None], 0.1) * (s.reaction_speed * dt)
                    )
        self.pos = self.pos + move
        self._separate()
        outside = ~self.camp.contains(self.pos)
        self.pos[outside] = old[outside]  # the fence holds
        self.t += dt
        return reacted

    def _separate(self, iterations: int = 3) -> None:
        """Push apart animals closer than the minimum spacing (within each group)."""
        d0 = self.settings.min_spacing
        for _ in range(iterations):
            for g in range(len(self.state)):
                idx = np.flatnonzero(self.group == g)
                if len(idx) < 2:
                    continue
                p = self.pos[idx]
                diff = p[:, None, :] - p[None, :, :]
                dist = np.linalg.norm(diff, axis=2)
                np.fill_diagonal(dist, np.inf)
                close = dist < d0
                if not close.any():
                    continue
                push = np.where(close[..., None], diff / np.maximum(dist, 1e-6)[..., None], 0.0)
                amount = np.where(close, (d0 - dist) / 2, 0.0)[..., None]
                self.pos[idx] += (push * amount).sum(1)


def _draw_state(s: HerdSettings, rng: np.random.Generator) -> str:
    p_rest = max(0.0, 1.0 - s.grazing_share - s.walking_share)
    return str(rng.choice(STATES, p=[s.grazing_share, s.walking_share, p_rest]))


SPACING_SCALES = tuple(np.round(np.arange(1.0, 5.01, 0.25), 2))


def _place_groups(
    camp: Camp, sizes: NDArray[np.int64], s: HerdSettings, scale: float, seed: int
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Grow each group from its centre: every new animal goes next to a random member at a drawn
    nearest-neighbour distance times `scale`, and is kept only if that member is (about) its
    nearest neighbour and it is inside the camp."""
    rng = np.random.default_rng(seed)
    pos, group = [], []
    for g, size in enumerate(sizes):
        members = [camp.random_point(rng, margin=10.0)]
        tries = 0
        while len(members) < size:
            tries += 1
            anchor = members[int(rng.integers(len(members)))]
            r = scale * draw_nn_distance(s.nn_quantiles, rng)
            a = rng.uniform(0, 2 * np.pi)
            p = anchor + r * np.array([np.cos(a), np.sin(a)])
            near = np.min(np.linalg.norm(np.array(members) - p, axis=1))
            ok = near >= max(s.min_spacing, 0.95 * r) and camp.contains(p)[0]
            if ok or tries > 200 * size:
                members.append(p)
        pos += members
        group += [g] * int(size)
    return np.array(pos, float), np.array(group, int)


def _refine(
    camp: Camp,
    pos: NDArray[np.float64],
    group: NDArray[np.int64],
    s: HerdSettings,
    rng: np.random.Generator,
    proposals: int = 1500,
) -> NDArray[np.float64]:
    """Move single animals next to another member of their group at a fresh drawn distance,
    keeping a move only if the spacing gets closer to the measured quantiles."""
    pos = pos.copy()
    best = nn_mismatch_m(pos, s.nn_quantiles)
    for _ in range(proposals):
        i = int(rng.integers(len(pos)))
        mates = np.flatnonzero((group == group[i]) & (np.arange(len(pos)) != i))
        if not len(mates):
            continue
        j = int(rng.choice(mates))
        a = rng.uniform(0, 2 * np.pi)
        new = pos[j] + draw_nn_distance(s.nn_quantiles, rng) * np.array([np.cos(a), np.sin(a)])
        others = np.delete(pos, i, axis=0)
        if np.min(np.linalg.norm(others - new, axis=1)) < s.min_spacing:
            continue
        if not camp.contains(new)[0]:
            continue
        old = pos[i].copy()
        pos[i] = new
        score = nn_mismatch_m(pos, s.nn_quantiles)
        if score < best:
            best = score
        else:
            pos[i] = old
    return pos


def place_herd(camp: Camp, n: int, s: HerdSettings, rng: np.random.Generator) -> Herd:
    """Place n animals in `groups` groups inside the camp, matching the measured spacing.

    Animals placed later sit closer to earlier ones than the distance they were placed at, so
    the drawn distances are scaled up: the scale (from SPACING_SCALES) whose result best matches
    the measured nearest-neighbour quantiles is kept (every trial uses the same random stream),
    then single animals are moved while that brings the spacing closer to the measurement."""
    sizes = rng.multinomial(n, rng.dirichlet(np.full(s.groups, 2.0)))
    sizes = sizes[sizes > 0]
    seed = int(rng.integers(2**63))
    trials = [_place_groups(camp, sizes, s, k, seed) for k in SPACING_SCALES]
    pos, group = min(
        trials, key=lambda tr: nn_mismatch_m(tr[0], s.nn_quantiles) if len(tr[0]) > 1 else 0.0
    )
    if len(pos) > 2:
        pos = _refine(camp, pos, group, s, np.random.default_rng(seed + 1))
    k = len(sizes)
    # a group keeps the spread it was placed with: members beyond its 95th-percentile distance
    # from the centre (at least the configured cohesion radius) drift back
    home = []
    for g in range(k):
        m = pos[group == g]
        r = np.linalg.norm(m - m.mean(0), axis=1)
        home.append(max(s.cohesion_radius, float(np.quantile(r, 0.95))))
    return Herd(
        camp=camp,
        settings=s,
        pos=pos,
        group=group,
        state=[_draw_state(s, rng) for _ in range(k)],
        target=np.array([camp.random_point(rng, margin=5.0) for _ in range(k)]),
        bout_left=[float(rng.exponential(s.bout_s)) for _ in range(k)],
        home_radius=home,
    )


def nearest_neighbour_m(pos: NDArray[np.float64]) -> NDArray[np.float64]:
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return d.min(1)


def nn_mismatch_m(pos: NDArray[np.float64], q: tuple[tuple[float, float], ...]) -> float:
    """Mean absolute difference between simulated and measured nearest-neighbour quantiles."""
    nn = nearest_neighbour_m(pos)
    return float(np.mean([abs(np.quantile(nn, p) - v) for p, v in q]))
