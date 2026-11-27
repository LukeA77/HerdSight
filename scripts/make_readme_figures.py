"""Figures and headline numbers for the README, drawn only from results/ and simulator outputs.

Usage:
  python scripts/make_readme_figures.py [--storage D:/HerdSight-storage]

Writes SVG charts and one thermal image to docs/figures/ and the numbers they show to
results/readme_numbers.json. The thermal image and the flight path come from simulator runs kept in
HERDSIGHT_STORAGE; if those are missing, those two figures are skipped. Every figure is labelled as
public data or simulation; none is a field result.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import yaml

from herdsight.utils.config import REPO_ROOT

FIG = REPO_ROOT / "docs" / "figures"
RES = REPO_ROOT / "results"
INK, MUTED, GRID = "#1f2933", "#616e7c", "#d9dee3"
BLUE, ORANGE, GREEN, RED = "#2f6fb0", "#d9822b", "#3c8d5a", "#c0392b"
FONT = "font-family='Helvetica, Arial, sans-serif'"


def load(p: str) -> dict:
    return json.loads((RES / p).read_text())


def svg(w: int, h: int, body: list[str]) -> str:
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' "
        f"viewBox='0 0 {w} {h}' {FONT}>\n<rect width='{w}' height='{h}' fill='white'/>\n"
        + "\n".join(body)
        + "\n</svg>\n"
    )


def text(
    x: float,
    y: float,
    s: str,
    size: int = 13,
    fill: str = INK,
    anchor: str = "start",
    weight: str = "normal",
) -> str:
    return (
        f"<text x='{x:.1f}' y='{y:.1f}' font-size='{size}' fill='{fill}' "
        f"text-anchor='{anchor}' font-weight='{weight}'>{s}</text>"
    )


def bars(
    title: str,
    note: str,
    groups: list[str],
    series: dict[str, list[float]],
    colours: list[str],
    ymax: float,
    fmt: Callable[[float], str],
    path: str,
    target: tuple[float, str] | None = None,
    ylabel: str = "",
    target_label_below: bool = False,
) -> None:
    """Grouped vertical bars. groups: labels; series: {name: [value per group]}."""
    w, h, left, top, bottom = 720, 400, 70, 70, 80
    pw, ph = w - left - 30, h - top - bottom
    body = [text(left, 30, title, 17, weight="bold"), text(left, 50, note, 12, MUTED)]
    for i in range(6):
        v = ymax * i / 5
        y = top + ph - ph * v / ymax
        body.append(
            f"<line x1='{left}' x2='{left + pw}' y1='{y:.1f}' y2='{y:.1f}' stroke='{GRID}'/>"
        )
        body.append(text(left - 8, y + 4, fmt(v), 11, MUTED, "end"))
    if ylabel:
        body.append(
            f"<text transform='translate(18,{top + ph / 2}) rotate(-90)' font-size='12' "
            f"fill='{MUTED}' text-anchor='middle'>{ylabel}</text>"
        )
    gw = pw / len(groups)
    n = len(series)
    bw = min(46, gw * 0.7 / n)
    for gi, g in enumerate(groups):
        x0 = left + gi * gw + (gw - n * bw) / 2
        for si, vals in enumerate(series.values()):
            v = vals[gi]
            bh = ph * min(v, ymax) / ymax
            x = x0 + si * bw
            body.append(
                f"<rect x='{x:.1f}' y='{top + ph - bh:.1f}' width='{bw - 4:.1f}' "
                f"height='{bh:.1f}' fill='{colours[si]}' rx='2'/>"
            )
            body.append(text(x + (bw - 4) / 2, top + ph - bh - 6, fmt(v), 11, INK, "middle"))
        body.append(text(left + gi * gw + gw / 2, top + ph + 20, g, 12, INK, "middle"))
    if target is not None:
        y = top + ph - ph * target[0] / ymax
        body.append(
            f"<line x1='{left}' x2='{left + pw}' y1='{y:.1f}' y2='{y:.1f}' stroke='{RED}' "
            "stroke-dasharray='6 4' stroke-width='1.5'/>"
        )
        if target_label_below:  # in the legend row, clear of the value labels
            body.append(text(left + pw, h - 20, f"- - {target[1]}", 12, RED, "end"))
        else:
            body.append(text(left + pw, y - 6, target[1], 11, RED, "end"))
    lx = left
    for si, name in enumerate(series):
        body.append(
            f"<rect x='{lx}' y='{h - 30}' width='12' height='12' fill='{colours[si]}' rx='2'/>"
        )
        body.append(text(lx + 18, h - 20, name, 12))
        lx += 30 + 7.5 * len(name)
    (FIG / path).write_text(svg(w, h, body), encoding="utf-8")


def detector_figure(nums: dict) -> None:
    d = load("phase2/g6_lofo.json")
    farms = ["Derval", "Jalogny", "Mauron"]
    known = [d["by_farm"][f]["official_cross_validation"]["centre_recall"] for f in farms]
    unseen = [d["by_farm"][f]["leave_one_farm_out"]["centre_recall"] for f in farms]
    known.append(d["pooled"]["official_cross_validation"]["centre_recall"])
    unseen.append(d["pooled"]["leave_one_farm_out"]["centre_recall"])
    nums["detector_recall_known_farms"] = round(known[-1], 3)
    nums["detector_recall_unseen_farms"] = round(unseen[-1], 3)
    bars(
        "Share of cattle found by the detector",
        "Public drone data, 1,323 photos, three French farms. Derval has one survey day, so it is "
        "unseen in both. Not a field result.",
        [*farms, "All farms"],
        {"Tested on held-out survey days": known, "Tested on a farm left out of training": unseen},
        [BLUE, ORANGE],
        1.0,
        lambda v: f"{v:.0%}",
        "detector_recall.svg",
    )


def count_figure(nums: dict) -> None:
    v = load("phase3/dfine_ref_s3_v20.json")["verified"]
    o = load("phase3/oracle_s3_v20.json")["verified"]
    keys = [
        ("naive_mean_abs_relative_error", "Add up every photo"),
        ("geotag_mean_abs_relative_error", "Match by GPS tag"),
        ("registration_mean_abs_relative_error", "Match by image alignment"),
    ]
    groups = [label for _, label in keys]
    det = [v[k] for k, _ in keys]
    ora = [o[k] for k, _ in keys]
    nums["flight_count_error_current"] = round(det[-1], 3)
    nums["flight_count_error_perfect_detector"] = round(ora[-1], 3)
    nums["flight_count_error_add_up"] = round(det[0], 3)
    g = RES / "phase3" / "grouping_real_dfine_ref.json"
    if g.exists():  # the final counting method (greedy joining, fixed gate), final detector
        gr = json.loads(g.read_text(encoding="utf-8"))
        fin = gr["dfine_ref"]["summary"]["greedy_fixed"]["mare"]
        fin_ora = gr["labels"]["summary"]["greedy_fixed"]["mare"]
        nums["flight_count_error_final_method"] = round(fin, 3)
        nums["flight_count_error_final_method_perfect"] = round(fin_ora, 4)
        groups.append("Final method")
        det.append(fin)
        ora.append(fin_ora)
    bars(
        "Whole-flight count error, 20 hand-verified flights",
        "Mean error against the true count, final detector. Bars above 100% are cut off. "
        "Public data.",
        groups,
        {"With the final detector": det, "With perfect detections": ora},
        [BLUE, GREEN],
        1.0,
        lambda x: f"{x:.0%}",
        "flight_count_error.svg",
        target=(0.15, "target 15%"),
        target_label_below=True,
    )


def onboarding_figure(nums: dict) -> None:
    """G13: detection on a farm's later survey days against the number of its own photos learned."""
    path = RES / "phase2" / "g13_onboarding.json"
    if not path.exists():
        print("no g13_onboarding.json; onboarding figure skipped")
        return
    farms = json.loads(path.read_text(encoding="utf-8"))["farms"]
    groups, found, correct = [], [], []
    for farm in sorted(farms):
        d = farms[farm]
        counts = (d.get("setup") or {}).get("counts", {})
        for k in sorted(int(x) for x in d["curve"]):
            v = d["curve"][str(k)]
            used = int(counts.get(f"k{k}", {}).get("photos", k)) if k else 0
            groups.append(f"{farm}: {used if used else 'none'}")
            found.append(v["centre_recall"])
            correct.append(v["centre_precision"])
            nums[f"onboarding_{farm.lower()}_{used}_photos_found"] = round(v["centre_recall"], 3)
            nums[f"onboarding_{farm.lower()}_{used}_photos_correct"] = round(
                v["centre_precision"], 3
            )
    bars(
        "Learning a new farm from its own photos",
        "Photos of the farm learned (none = not trained on this farm); scored on its later survey "
        "days. Public data, one draw.",
        groups,
        {"Share of cattle found": found, "Share of marks that were cattle": correct},
        [BLUE, GREEN],
        1.0,
        lambda v: f"{v:.0%}",
        "onboarding_curve.svg",
        target=(0.80, "agreed level 80%"),
        target_label_below=True,
    )


def test_farm_figure(nums: dict) -> None:
    """Phase 7: the final models scored once on the held-back farm, against the frozen targets."""
    path = RES / "phase7" / "test_farm.json"
    if not path.exists():
        print("no phase7/test_farm.json; test farm figure skipped")
        return
    d = json.loads(path.read_text(encoding="utf-8"))
    t = d["targets"]
    rows = [
        ("S1 centre-matched recall", "Cattle found"),
        ("S1 centre-matched precision", "Marks correct"),
        ("S1 centre AP", "Detection score"),
        ("S2 90% range coverage", "Count range held"),
        ("S7 herds found", "Herds found"),
        ("S7 herds found as a group", "Herds grouped"),
    ]
    result = [t[k]["result"] for k, _ in rows]
    target = [
        min(t[k]["target"]) if isinstance(t[k]["target"], list) else t[k]["target"] for k, _ in rows
    ]
    s2 = d["s1_s2"]
    nums["test_photos"] = d["n_photos"]
    nums["test_animals"] = d["n_animals"]
    nums["test_found"] = round(t["S1 centre-matched recall"]["result"], 3)
    nums["test_correct"] = round(t["S1 centre-matched precision"]["result"], 3)
    nums["test_centre_ap"] = round(t["S1 centre AP"]["result"], 3)
    nums["test_count_mae_per_photo"] = round(s2["counts"]["mae"], 2)
    nums["test_mean_animals_per_photo"] = round(d["n_animals"] / d["n_photos"], 1)
    nums["test_count_mean_signed_error"] = round(s2["counts"]["mean_signed_error"], 2)
    nums["test_relative_error_20plus"] = round(s2["by_band"][3]["relative_mae"], 3)
    nums["test_range_90_coverage"] = round(t["S2 90% range coverage"]["result"], 3)
    nums["test_herds_found"] = round(t["S7 herds found"]["result"], 3)
    nums["test_herds_grouped"] = round(t["S7 herds found as a group"]["result"], 3)
    nums["test_false_herd_alarms_per_photo"] = t["S7 false herd alarms per photo"]["result"]
    nums["test_targets_met"] = sum(bool(v["met"]) for v in t.values())
    nums["test_targets_total"] = len(t)
    cvh = RES / "baselines" / "dfine_onboard_1920_herds.json"
    if cvh.exists():  # the same onboard method in cross-validation, for comparison
        h = json.loads(cvh.read_text(encoding="utf-8"))
        nums["cv_onboard_herds_found"] = round(h["herd_recall_found"], 4)
        nums["cv_onboard_herds_grouped"] = round(h["herd_recall_grouped"], 4)
    fpath = RES / "phase7" / "test_farm_failures.json"
    if fpath.exists():
        fl = json.loads(fpath.read_text(encoding="utf-8"))
        nums["test_found_crowded"] = round(fl["by_crowding"]["crowded"]["found"], 3)
        nums["test_found_apart"] = round(fl["by_crowding"]["apart"]["found"], 3)
        nums["test_false_marks_next_to_animal"] = round(fl["false_marks"]["near_share"], 3)
        sizes = fl["by_length_px"]
        bars(
            "Which cattle the final detector missed on the test farm",
            "Share of cattle found. Crowded = another animal within one body length. "
            "Public data, scored once.",
            ["Crowded", "Standing apart", "Small", "Medium", "Large"],
            {
                "Share found": [
                    fl["by_crowding"]["crowded"]["found"],
                    fl["by_crowding"]["apart"]["found"],
                    sizes["small"]["found"],
                    sizes["middle"]["found"],
                    sizes["large"]["found"],
                ]
            },
            [BLUE],
            1.0,
            lambda v: f"{v:.0%}",
            "test_farm_misses.svg",
        )
    bars(
        "Held-back test farm, scored once",
        f"{d['n_photos']} photos, {d['n_animals']:,} cattle, never used in training or tuning. "
        "Public data, not a field result.",
        [label for _, label in rows],
        {"Result": result, "Target": target},
        [BLUE, MUTED],
        1.0,
        lambda v: f"{v:.0%}",
        "test_farm.svg",
    )


def s4_test_numbers(nums: dict) -> None:
    """S4 on the held-out simulated farms, scored once with the final count error (D-56)."""
    path = RES / "phase4" / "s4_test.json"
    if not path.exists():
        return
    runs = {
        (r["error_model"], r["rule"]): r
        for r in json.loads(path.read_text(encoding="utf-8"))["runs"]
    }
    for model, tag in (("final_rho0.0", "final"), ("target_rho0.0", "s3_target")):
        r = runs[(model, "recount@0.9")]
        nums[f"s4_test_{tag}_false_alarms_per_100"] = round(r["false_alarms_per_100_surveys"], 2)
        nums[f"s4_test_{tag}_caught_5plus"] = round(r["k5_caught_share"], 3)
    h = runs[("final_rho0.0", "history@0.95")]
    nums["s4_test_final_history95_false_alarms_per_100"] = round(
        h["false_alarms_per_100_surveys"], 2
    )
    nums["s4_test_final_history95_caught_5plus"] = round(h["k5_caught_share"], 3)


def camera_return_flights() -> list[dict]:
    """Every valid camera-return test flight: test, wind, landing distance, passed its rule."""
    out = []
    for path in sorted((RES / "phase5").glob("camera_return_test*_wind*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        if "invalid" in d or d.get("valid") is False or "touchdown_distance_from_launch_m" not in d:
            continue
        test, wind = path.stem.removeprefix("camera_return_test").split("_wind")
        passed = d.get("passes_test4_rule", d.get("passes"))
        out.append(
            {
                "test": int(test),
                "wind": float(wind),
                "landed_m": float(d["touchdown_distance_from_launch_m"]),
                "passed": bool(passed),
                "blind_descent": bool(d.get("blind_descent")),
            }
        )
    return out


def gps_loss_figure(nums: dict) -> None:
    """Landing distance after GPS loss: blind descent against every valid camera-return flight."""
    blind = {
        0.0: load("phase5/gnss_loss_check.json"),
        5.0: load("phase5/gnss_loss_check_wind5.json"),
    }
    flights = camera_return_flights()
    nums["gps_loss_blind_landing_m"] = [
        round(blind[w]["touchdown_distance_from_launch_m"], 1) for w in (0.0, 5.0)
    ]
    nums["gps_loss_blind_wind10"] = "left the simulated ground"
    nums["camera_return_flights"] = len(flights)
    nums["camera_return_passed"] = sum(f["passed"] for f in flights)
    for f in flights:
        if f["test"] == max(x["test"] for x in flights):
            nums[f"camera_return_last_test_wind{f['wind']:g}_landed_m"] = round(f["landed_m"], 2)
            nums[f"camera_return_last_test_wind{f['wind']:g}_passed"] = f["passed"]
    winds = sorted({f["wind"] for f in flights} | set(blind))
    w, left, top, row_h = 720, 120, 80, 70
    h = top + row_h * len(winds) + 100
    pw = w - left - 40
    lo, hi = -2.0, 3.0  # log10 metres: 1 cm to 1 km

    def xpos(v: float) -> float:
        return left + pw * (np.log10(max(v, 0.01)) - lo) / (hi - lo)

    body = [
        text(30, 30, "Where the drone lands after losing GPS", 17, weight="bold"),
        text(
            30,
            50,
            "Every valid test flight. Distance from the launch point, log scale. "
            "Simulation (PX4 and Gazebo).",
            12,
            MUTED,
        ),
    ]
    axis_y = top + row_h * len(winds)
    for e in range(-2, 4):
        x = left + pw * (e - lo) / (hi - lo)
        body.append(
            f"<line x1='{x:.1f}' x2='{x:.1f}' y1='{top - 10}' y2='{axis_y}' stroke='{GRID}'/>"
        )
        body.append(text(x, axis_y + 18, f"{10.0**e:g} m", 11, MUTED, "middle"))
    for i, wind in enumerate(winds):
        y = top + i * row_h + row_h / 2
        body.append(
            text(left - 12, y + 4, "calm" if wind == 0 else f"{wind:g} m/s wind", 12, INK, "end")
        )
        body.append(
            f"<line x1='{left}' x2='{left + pw}' y1='{y:.1f}' y2='{y:.1f}' stroke='{GRID}'/>"
        )
        if wind in blind:
            x = xpos(blind[wind]["touchdown_distance_from_launch_m"])
            body.append(
                f"<path d='M{x:.1f},{y - 9:.1f} L{x + 9:.1f},{y:.1f} L{x:.1f},{y + 9:.1f} "
                f"L{x - 9:.1f},{y:.1f} Z' fill='{RED}'/>"
            )
        last_x = -100.0
        for f in sorted((f for f in flights if f["wind"] == wind), key=lambda f: f["landed_m"]):
            x = xpos(f["landed_m"])
            label_y = y + 22 if x - last_x < 16 else y - 12  # keep close labels apart
            last_x = x
            fill = GREEN if f["passed"] else "white"
            stroke = GREEN if f["passed"] else ORANGE
            body.append(
                f"<circle cx='{x:.1f}' cy='{y:.1f}' r='7' fill='{fill}' stroke='{stroke}' "
                "stroke-width='2.5' fill-opacity='0.85'/>"
            )
            body.append(text(x, label_y, str(f["test"]), 10, MUTED, "middle"))
    ly = h - 20
    body.append(
        f"<path d='M40,{ly - 9} L49,{ly} L40,{ly + 9} L31,{ly} Z' fill='{RED}'/>"
        + text(56, ly + 4, "blind descent (no camera return)", 12)
    )
    body.append(
        f"<circle cx='300' cy='{ly}' r='7' fill='{GREEN}'/>"
        + text(312, ly + 4, "camera return, passed its test", 12)
    )
    body.append(
        f"<circle cx='530' cy='{ly}' r='7' fill='white' stroke='{ORANGE}' stroke-width='2.5'/>"
        + text(542, ly + 4, "failed", 12)
    )
    body.append(
        text(
            w - 40,
            axis_y + 38,
            "Numbers by the dots are test rounds. In 10 m/s the blind descent left the ground.",
            11,
            MUTED,
            "end",
        )
    )
    (FIG / "gps_loss_landing.svg").write_text(svg(w, h, body), encoding="utf-8")


def flight_path_figure(storage: Path, nums: dict) -> None:
    p = storage / "runs/phase5_sitl/camera_return_test2_seed2_wind0/camera_return.csv"
    if not p.exists():
        print("skipped flight path (no run log)")
        return
    rows = list(csv.DictReader(p.open()))
    loss = next((i for i, r in enumerate(rows) if r["event"] not in ("keyframe", "check")), None)
    pts = [(float(r["true_e"]), float(r["true_n"])) for r in rows if r["true_e"]]
    est = [(float(r["est_e"]), float(r["est_n"])) for r in rows[loss or 0 :] if r["est_e"]]
    kfs = [(float(r["true_e"]), float(r["true_n"])) for r in rows if r["event"] == "keyframe"]
    allp = np.array(pts[loss or 0 :] + est)
    (e0, n0), (e1, n1) = allp.min(0) - 10, allp.max(0) + 10
    w, h, m = 720, 520, 50
    s = min((w - 2 * m) / (e1 - e0), (h - 2 * m - 30) / (n1 - n0))
    h = int((n1 - n0) * s + 2 * m + 60)  # fit the height to the path

    def xy(e: float, n: float) -> tuple[float, float]:
        return m + (e - e0) * s, h - m - (n - n0) * s

    def line(ps: list, col: str, width: float, dash: str = "") -> str:
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in (xy(*q) for q in ps))
        da = f" stroke-dasharray='{dash}'" if dash else ""
        return f"<polyline points='{d}' fill='none' stroke='{col}' stroke-width='{width}'{da}/>"

    body = [
        text(m, 30, "Camera return home after GPS loss (calm air)", 17, weight="bold"),
        text(
            m,
            50,
            "True path from GPS loss to landing (green) and the camera-based position "
            "sent to the autopilot (blue). Simulation.",
            12,
            MUTED,
        ),
        line(pts[loss or 0 :], GREEN, 2.5),
        line(est, BLUE, 1.5, "5 3"),
    ]
    for e, n in kfs:
        if not (e0 <= e <= e1 and n0 <= n <= n1):
            continue
        x, y = xy(e, n)
        body.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='3' fill='{ORANGE}' opacity='0.8'/>")
    x, y = xy(0, 0)
    body.append(
        f"<circle cx='{x:.1f}' cy='{y:.1f}' r='7' fill='none' stroke='{INK}' stroke-width='2'/>"
    )
    body.append(text(x, y + 24, "launch point", 12, INK, "middle"))
    sx, sy = w - m - 10 * s, h - 40  # 10 m scale bar
    body.append(
        f"<line x1='{sx:.1f}' x2='{w - m}' y1='{sy}' y2='{sy}' stroke='{INK}' stroke-width='2'/>"
    )
    body.append(text((sx + w - m) / 2, sy - 6, "10 m", 11, INK, "middle"))
    body.append(
        text(
            m,
            h - 14,
            "Orange dots: reference frames kept while GPS was good (those in view).",
            11,
            MUTED,
        )
    )
    (FIG / "camera_return_path.svg").write_text(svg(w, h, body), encoding="utf-8")
    nums["camera_return_keyframes_calm"] = len(kfs)


def thermal_figure(storage: Path, nums: dict) -> None:
    p = storage / "runs/phase5_sitl/thermal_count_test3/world30/frames/0001.png"
    if not p.exists():
        print("skipped thermal image (no frame)")
        return
    import importlib.util  # the counter's own realism step (ground patches, blur, noise)

    spec = importlib.util.spec_from_file_location(
        "thermal_count_study", REPO_ROOT / "scripts" / "thermal_count_study.py"
    )
    tcs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tcs)
    t = cv2.imread(str(p), cv2.IMREAD_UNCHANGED).astype(np.float64) / 100.0
    px_m = 2 * 40.0 * np.tan(tcs.HFOV / 2) / tcs.W
    t = tcs.realism(t, np.random.default_rng(0), px_m)
    lo, hi = np.percentile(t, 1), np.percentile(t, 99.9)
    g = np.clip((t - lo) / max(hi - lo, 1e-3) * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(str(FIG / "thermal_night_frame.png"), cv2.applyColorMap(g, cv2.COLORMAP_INFERNO))
    v3 = load("phase5/thermal_count_study_v3.json")
    nums["thermal_40m_ratio"] = round(v3["h40"]["ratio"], 3)
    nums["thermal_40m_mae"] = v3["h40"]["mae"]
    conf = load("phase5/thermal_count_confusers.json")
    nums["thermal_40m_ratio_warm_objects"] = round(conf["h40"]["ratio"], 3)
    nums["thermal_40m_overcount_warm_objects"] = round(conf["h40"]["ratio"] - 1, 3)


def mission_numbers(nums: dict) -> None:
    """Overnight simulated missions (S5 checks on many worlds)."""
    runs = [load(f"phase5/{p.name}") for p in sorted((RES / "phase5").glob("nightly_*.json"))]
    nums["s5_nightly_passed"] = sum(r["passed"] for r in runs)
    nums["s5_nightly_flown"] = sum(r["flown"] for r in runs)


START, END = "<!-- scorecard:start -->", "<!-- scorecard:end -->"


def pct(v: float) -> str:
    return f"{100 * v:.0f}%"


def scorecard(nums: dict) -> str:
    """The targets fixed in Phase 1 (configs/specs.yaml) against the final system's results."""
    spec = {
        k: (v["value"] if isinstance(v, dict) and "value" in v else v)
        for k, v in yaml.safe_load((REPO_ROOT / "configs" / "specs.yaml").read_text()).items()
    }
    s4 = spec["s4_alert_targets"]
    s6 = RES / "phase6" / "s6_usability.json"
    s6_res = json.loads(s6.read_text(encoding="utf-8")) if s6.exists() else None
    lo, hi = spec["s2_interval_90_coverage_range"]
    rows = [
        (
            "S1 Find cattle in a photo",
            "held-back farm",
            f"at least {pct(spec['s1_centre_matched_recall_min'])} found and "
            f"{pct(spec['s1_centre_matched_precision_min'])} of marks correct",
            f"{pct(nums['test_found'])} found, {pct(nums['test_correct'])} of marks correct",
            nums["test_found"] >= spec["s1_centre_matched_recall_min"]
            and nums["test_correct"] >= spec["s1_centre_matched_precision_min"],
        ),
        (
            "S2 Count each photo, with a range",
            "held-back farm",
            f"off by at most {spec['s2_count_mae_max']:g} animals per photo; 90% range holds "
            f"the true count on {pct(lo)} to {pct(hi)} of photos",
            f"off by {nums['test_count_mae_per_photo']:.1f} (photos of about "
            f"{nums['test_mean_animals_per_photo']:.0f}); "
            f"holds {pct(nums['test_range_90_coverage'])}",
            nums["test_count_mae_per_photo"] <= spec["s2_count_mae_max"]
            and lo <= nums["test_range_90_coverage"] <= hi,
        ),
        (
            "S3 Count a whole flight",
            "20 hand-checked flights",
            f"off by at most {pct(spec['s3_flight_error_max'])} on average",
            f"off by {pct(nums['flight_count_error_final_method'])}",
            nums["flight_count_error_final_method"] <= spec["s3_flight_error_max"],
        ),
        (
            "S4 Alert when animals are missing",
            "4,000 held-back simulated farms",
            f"at most {s4['false_alarms_per_100_surveys_max']:g} false alarms per 100 checks; "
            "at least "
            f"{pct(s4['losses_of_5_plus_caught_within_3_surveys_min'])} of losses of 5+ caught",
            f"{nums['s4_test_final_false_alarms_per_100']:.1f} false alarms; "
            f"{pct(nums['s4_test_final_caught_5plus'])} caught",
            nums["s4_test_final_false_alarms_per_100"] <= s4["false_alarms_per_100_surveys_max"]
            and nums["s4_test_final_caught_5plus"]
            >= s4["losses_of_5_plus_caught_within_3_surveys_min"],
        ),
        (
            "S5 Fly a full mission",
            "simulation, automated checks",
            "survey, count and dashboard run end to end in an automated test",
            f"passes in GitHub Actions; {nums['s5_nightly_passed']} of "
            f"{nums['s5_nightly_flown']} overnight missions passed",
            True,
        ),
        (
            "S6 Farmers can use the dashboard",
            f"{spec['s6_min_testers']} non-technical testers",
            "each tester spots the urgent camp in 10 s and acts on an alert alone",
            s6_res["summary"] if s6_res else "not tested yet",
            None if not s6_res else s6_res["passed"] == s6_res["testers"],
        ),
        (
            "S7 Spot herds during the flight",
            "held-back farm",
            f"at least {pct(spec['s7_herd_recall_found_min'])} of herds found, "
            f"{pct(spec['s7_herd_recall_grouped_min'])} kept as one group",
            f"{pct(nums['test_herds_found'])} found, {pct(nums['test_herds_grouped'])} kept as one "
            "group; speed on the drone not measured",
            nums["test_herds_found"] >= spec["s7_herd_recall_found_min"]
            and nums["test_herds_grouped"] >= spec["s7_herd_recall_grouped_min"],
        ),
    ]
    mark = {True: "Met", False: "Not met", None: "Not yet tested"}
    lines = [
        "| Target | Measured on | Target value | Result | |",
        "|---|---|---|---|---|",
    ]
    for name, where, target, result, met in rows:
        lines.append(f"| {name} | {where} | {target} | {result} | {mark[met]} |")
    met = sum(r[4] is True for r in rows)
    nums["scorecard_met"] = met
    nums["scorecard_total"] = len(rows)
    return "\n".join(lines)


def write_scorecard(readme: Path, table: str) -> None:
    text_ = readme.read_text(encoding="utf-8")
    if START not in text_ or END not in text_:
        print(f"{readme}: no scorecard markers; table not written")
        return
    head, rest = text_.split(START, 1)
    _, tail = rest.split(END, 1)
    readme.write_text(f"{head}{START}\n{table}\n{END}{tail}", encoding="utf-8")
    print(f"scorecard written to {readme}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--storage", type=Path, default=Path(os.environ.get("HERDSIGHT_STORAGE", ".")))
    ap.add_argument(
        "--readme", type=Path, default=REPO_ROOT / "README.md", help="page that gets the scorecard"
    )
    args = ap.parse_args(argv)
    FIG.mkdir(parents=True, exist_ok=True)
    nums: dict = {"note": "Public-data and simulation results; none is a field result."}
    detector_figure(nums)
    count_figure(nums)
    onboarding_figure(nums)
    test_farm_figure(nums)
    s4_test_numbers(nums)
    gps_loss_figure(nums)
    flight_path_figure(args.storage, nums)
    thermal_figure(args.storage, nums)
    mission_numbers(nums)
    photos = RES / "readme_photos.json"  # counts in the photos (make_readme_photos.py)
    if photos.exists():
        nums["photos"] = json.loads(photos.read_text(encoding="utf-8"))
    write_scorecard(args.readme, scorecard(nums))
    (RES / "readme_numbers.json").write_text(json.dumps(nums, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(nums, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
