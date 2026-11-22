"""Error models for simulated flights, fitted to real flights (Phase 5 plan, steps 3-4; D-11).

Rendered cattle are never scored. Home base sees, for each still actually captured:
  * detections drawn by `DetectionModel` from the animals visible in that still
    (results/data/detection_error_model.json: chance of finding an animal by its length in
    pixels and crowding, with random effects per flight, per still and per animal, so misses
    cluster as on real farms and a hard-to-see animal stays hard in every still; position noise;
    false detections from fixed objects in the world);
  * a geotag with error drawn by `GeotagModel` (results/data/flight_error_model.json: a
    standard deviation per flight drawn from the real flights);
  * for each overlapping pair, a registration outcome drawn by `RegistrationModel` (accepted
    share and residual of real pairs; otherwise the pair falls back to the geotags).
Two settings are assumptions, not fits, and are swept: the chance a fixed false-detection object
is detected in one view (`p_view`), and how the visible share of a partly hidden animal changes
its chance of being found (proportional). Calves use the size term extrapolated from adults.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from herdsight.utils.config import REPO_ROOT

DATA = REPO_ROOT / "results" / "data"


@dataclass(frozen=True)
class DetectionModel:
    intercept: float
    size: float
    crowded: float
    group_sd: float
    image_sd: float
    animal_sd: float
    length_ref_px: float
    position_rms_share: float
    fp_per_ha_by_group: tuple[float, ...]
    new_farm_effect: float
    version: str
    p_view: float = 0.5

    @staticmethod
    def load(p_view: float = 0.5) -> DetectionModel:
        d = json.loads((DATA / "detection_error_model.json").read_text())
        c = d["detection"]["coefficients"]
        return DetectionModel(
            intercept=c["intercept"],
            size=c["log_length_over_60px"],
            crowded=c["crowded"],
            group_sd=d["detection"]["group_sd"],
            image_sd=d["detection"]["image_sd"],
            animal_sd=d["detection"]["animal_sd"],
            length_ref_px=d["length_ref_px"],
            position_rms_share=d["position_error_share_of_length"]["rms"],
            fp_per_ha_by_group=tuple(d["false_detections_per_ha"]["by_group"].values()),
            new_farm_effect=d["new_farm_case"]["group_effect"],
            version=f"detection_error_model.json ({d['run']}, {d['note']})",
            p_view=p_view,
        )

    def flight_effect(self, rng: np.random.Generator, new_farm: bool = False) -> float:
        return self.new_farm_effect if new_farm else float(rng.normal(0, self.group_sd))

    def object_density_per_ha(self, rng: np.random.Generator) -> float:
        """Fixed false-detection objects per hectare for one simulated world: a real flight's
        false-detection rate divided by the chance an object is detected in one view."""
        return float(rng.choice(self.fp_per_ha_by_group)) / self.p_view

    def found(
        self,
        length_px: ArrayLike,
        crowded: ArrayLike,
        visible: ArrayLike,
        flight_effect: float,
        rng: np.random.Generator,
        animal_effect: ArrayLike | None = None,
    ) -> NDArray[np.bool_]:
        """Which of the visible animals in one still are detected. `animal_effect`: each
        animal's own effect, drawn once per flight (`animal_effects`), so an animal that is hard
        to see stays hard in every still."""
        length = np.asarray(length_px, float)
        if not len(length):
            return np.zeros(0, bool)
        eta = (
            self.intercept
            + self.size * np.log(np.maximum(length, 1.0) / self.length_ref_px)
            + self.crowded * np.asarray(crowded, float)
            + flight_effect
            + rng.normal(0, self.image_sd)
            + (0.0 if animal_effect is None else np.asarray(animal_effect, float))
        )
        p = np.asarray(visible, float) / (1 + np.exp(-eta))
        return rng.random(len(length)) < p

    def animal_effects(self, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
        return rng.normal(0, self.animal_sd, n)

    def position_noise_m(self, length_m: ArrayLike, rng: np.random.Generator) -> NDArray:
        length = np.asarray(length_m, float)
        sd = self.position_rms_share * length / np.sqrt(2)
        return rng.normal(0, 1, (len(length), 2)) * sd[:, None]


@dataclass(frozen=True)
class GeotagModel:
    sigmas: tuple[float, ...]
    weights: tuple[float, ...]
    version: str = "flight_error_model.json geotag"

    @staticmethod
    def load() -> GeotagModel:
        d = json.loads((DATA / "flight_error_model.json").read_text())
        g = d["geotag"]
        flights = list(g["sigma_m_by_flight"])
        w = np.array([g["pairs_by_flight"][f] for f in flights], float)
        return GeotagModel(
            tuple(g["sigma_m_by_flight"][f] for f in flights), tuple((w / w.sum()).tolist())
        )

    def flight_sigma(self, rng: np.random.Generator) -> float:
        return float(rng.choice(self.sigmas, p=self.weights))

    def errors(self, n: int, sigma: float, rng: np.random.Generator) -> NDArray[np.float64]:
        return rng.normal(0, sigma, (n, 2))


@dataclass(frozen=True)
class RegistrationModel:
    accepted_share: float
    residual_px: tuple[float, ...]
    version: str = "flight_error_model.json registration"

    @staticmethod
    def load() -> RegistrationModel:
        d = json.loads((DATA / "flight_error_model.json").read_text())["registration"]
        return RegistrationModel(d["accepted_share"], tuple(d["residual_px"]["values"]))

    def outcome(self, rng: np.random.Generator) -> float | None:
        """Residual in pixels if the pair registers, else None (geotag fallback)."""
        if rng.random() < self.accepted_share:
            return float(rng.choice(self.residual_px))
        return None
