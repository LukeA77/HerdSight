"""Phase 7: the one-time test on the held-back test farm (brief section 9.5).

Usage:
  python scripts/phase7_test.py --rehearse        dry run on 3 cross-validation photos (no guard,
                                                  output to the storage folder only)
  python scripts/phase7_test.py                   the real run: once, after
                                                  scripts/freeze.py --phase phase7

The two final models and their SHA-256 are named in configs/phase7.yaml (frozen with this
script, the specs and the evaluation settings). The run:
  1. refuses to start unless phase7 is frozen, unchanged and not run before
     (herdsight.utils.run_once), then writes its marker before reading any test photo;
  2. predicts every labelled test-farm photo with the home-base detector (full-resolution tiles)
     and the onboard herd detector (stills reduced to its long side first, D-29);
  3. scores S1 and S2 with herdsight's Phase 1 scorer (scripts/score_baseline.evaluate). The
     operating threshold and the conformal count quantiles are fitted once, on ALL
     cross-validation out-of-fold predictions of the same recipe (calibration_run), and never
     on the test farm;
  4. scores S7 (herds found, found as a group, false triggers) the same way, with the onboard
     threshold fitted on the onboard recipe's out-of-fold predictions;
  5. compares each measure with its frozen target in configs/specs.yaml.
Intervals are bootstrap over the test farm's survey groups, which are few, so they are wide.
Public data (ICAERUS), never a field result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from herdsight.data.icaerus import default_root, iter_images
from herdsight.data.splits import FlightStats
from herdsight.detect.dfine_onnx import DfineOnnxTileDetector
from herdsight.detect.run_tiled import detect_full_image
from herdsight.evaluate.counts import fit_count_conformal
from herdsight.evaluate.herds import score_image
from herdsight.evaluate.stats import group_bootstrap_ci
from herdsight.geo.nadir import focal_px_from_35mm
from herdsight.utils.config import REPO_ROOT, load_config
from herdsight.utils.env_report import env_report
from herdsight.utils.io import load_rgb
from herdsight.utils.paths import storage_path
from herdsight.utils.run_once import claim_test_run

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_baseline import LEVELS, best_threshold, evaluate, load_preds  # noqa: E402

PHASE = "phase7"
RESULT = REPO_ROOT / "results" / "phase7" / "test_farm.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def key_of(s) -> str:  # noqa: ANN001
    return f"{s.farm}/{s.flight}/{s.path.name}"


def predict(images, onnx: Path, tile: int, overlap: int, long_side, dest: Path, threads) -> dict:  # noqa: ANN001
    """Full-photo predictions, one JSON line per photo (resumes after an interruption)."""
    detector = DfineOnnxTileDetector(str(onnx), min_score=0.01, size=tile, threads=threads)
    done = load_preds(dest)
    todo = [s for s in images if key_of(s) not in done]
    print(f"{onnx.parent.name}: {len(images)} photos, {len(todo)} to do", flush=True)
    t0 = time.time()
    with dest.open("a", encoding="utf-8") as fh:
        for i, s in enumerate(todo, 1):
            im = load_rgb(s.path)
            scale = 1.0
            if long_side:
                scale = long_side / max(im.size)
                im = im.resize(
                    (round(im.width * scale), round(im.height * scale)), Image.Resampling.LANCZOS
                )
            boxes, scores = detect_full_image(im, detector, tile, overlap)
            boxes = boxes / scale
            rec = {
                "image": key_of(s),
                "boxes": [[round(v, 2) for v in b] for b in boxes.tolist()],
                "scores": [round(v, 5) for v in scores.tolist()],
            }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if i % 5 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  {(time.time() - t0) / i:.0f} s/photo", flush=True)
    return load_preds(dest)


def cv_truth(sp: dict) -> tuple[dict, dict]:
    folds = {f"fold{k}" for k in range(sp["n_folds"])}
    truth, group_of = {}, {}
    for s, problem in iter_images(default_root(), with_metadata=False):
        if problem is None and s.labelled and sp["flights"].get(f"{s.farm}/{s.flight}") in folds:
            truth[key_of(s)] = s.boxes_xyxy
            group_of[key_of(s)] = FlightStats(s.farm, s.flight, 0, 0).group
    return truth, group_of


def calibrate(run: str, sp: dict, truth: dict, grid: np.ndarray) -> dict:
    """Threshold and conformal quantiles from every out-of-fold prediction of `run`."""
    preds: dict = {}
    for k in range(sp["n_folds"]):
        preds |= load_preds(storage_path("runs", "baselines", run) / f"fold{k}.jsonl")
    keys = sorted(k for k in preds if k in truth)
    if len(keys) < len(truth):
        raise SystemExit(
            f"{run}: {len(truth) - len(keys)} cross-validation photos lack predictions"
        )
    thr, mae = best_threshold(keys, preds, truth, grid)
    pred_c = np.array([(preds[k][1] >= thr).sum() for k in keys], float)
    true_c = np.array([truth[k].shape[0] for k in keys], float)
    return {
        "calibration_run": run,
        "threshold": thr,
        "fit_mae": mae,
        "fit_images": len(keys),
        "q": fit_count_conformal(pred_c, true_c, LEVELS),
    }


def meta(keys: list[str]) -> tuple[dict, dict]:
    alt_of, gsd_in = {}, {}
    with (REPO_ROOT / "results" / "data" / "icaerus_images.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            k = f"{r['farm']}/{r['flight']}/{r['image']}"
            alt_of[k] = float(r["rel_alt_m"]) if r["rel_alt_m"] else None
            if r["rel_alt_m"] and r["focal_35mm"]:
                gsd_in[k] = (float(r["rel_alt_m"]), float(r["focal_35mm"]))
    return {k: alt_of.get(k) for k in keys}, gsd_in


def targets_check(s1s2: dict, s7: dict, specs: dict) -> dict:
    v = {k: specs[k]["value"] for k in specs if isinstance(specs[k], dict) and "value" in specs[k]}
    cm = s1s2.get("centre_matched", {})
    cov = s1s2["intervals"]["0.90"]
    dense = next(b for b in cov["coverage_by_band"] if b["band"][0] == 20)
    rows = {
        "S1 centre-matched recall": (cm.get("recall"), ">=", v["s1_centre_matched_recall_min"]),
        "S1 centre-matched precision": (
            cm.get("precision"),
            ">=",
            v["s1_centre_matched_precision_min"],
        ),
        "S1 centre AP": (s1s2["centre_ap"].get("ap_mean"), ">=", v["s1_centre_ap_min"]),
        "S2 count MAE per photo": (s1s2["counts"]["mae"], "<=", v["s2_count_mae_max"]),
        "S2 90% range coverage": (cov["coverage"], "in", v["s2_interval_90_coverage_range"]),
        "S2 90% range coverage, 20+ animals": (
            dense["coverage"],
            ">=",
            v["s2_interval_90_coverage_20plus_min"],
        ),
    }
    rows["S7 herds found"] = (s7["herd_recall_found"], ">=", v["s7_herd_recall_found_min"])
    rows["S7 herds found as a group"] = (
        s7["herd_recall_grouped"],
        ">=",
        v["s7_herd_recall_grouped_min"],
    )
    rows["S7 false herd alarms per photo"] = (
        s7["false_triggers_per_image"],
        "<=",
        v["s7_false_herd_alarms_per_image_max"],
    )
    out = {}
    for name, (got, op, want) in rows.items():
        if got is None:
            met = None
        elif op == ">=":
            met = got >= want
        elif op == "<=":
            met = got <= want
        else:
            met = want[0] <= got <= want[1]
        out[name] = {"result": got, "target": want, "rule": op, "met": met}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rehearse", action="store_true")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--home-onnx", type=Path, default=None, help="--rehearse only: model to use")
    ap.add_argument("--onboard-onnx", type=Path, default=None, help="--rehearse only")
    args = ap.parse_args(argv)
    p7 = load_config("phase7")
    ev, cfg = load_config("evaluation"), load_config("baselines")
    specs = yaml.safe_load((REPO_ROOT / "configs" / "specs.yaml").read_text(encoding="utf-8"))
    sp = json.loads((REPO_ROOT / "results" / "splits.json").read_text())["icaerus_v2"]
    models = {}
    for role in ("home_base", "onboard"):
        m = dict(p7.require(role))
        override = args.home_onnx if role == "home_base" else args.onboard_onnx
        if override is not None:
            if not args.rehearse:
                raise SystemExit("--home-onnx and --onboard-onnx are for --rehearse only")
            path = override
        else:
            path = storage_path(*Path(m["onnx"]).parts)
            if m["sha256"] is None:
                raise SystemExit(f"configs/phase7.yaml: {role} sha256 is not set yet")
        got = sha256(path)
        if override is None and got != m["sha256"]:
            raise SystemExit(f"{role} model {path} has SHA-256 {got}, not the frozen {m['sha256']}")
        models[role] = {**m, "path": path, "sha256": got}

    if args.rehearse:
        out_dir = storage_path("runs", "phase7", "rehearsal", create=True)
        images = [
            s
            for s, problem in iter_images(default_root(), with_metadata=False)
            if problem is None
            and s.labelled
            and sp["flights"].get(f"{s.farm}/{s.flight}") == "fold0"
        ]
        images = sorted(images, key=key_of)[::150][:3]
        dest = out_dir / "rehearsal.json"
    else:
        claim_test_run(PHASE, RESULT)
        out_dir = storage_path("runs", "phase7", create=True)
        images = [
            s
            for s, problem in iter_images(default_root(), with_metadata=False)
            if problem is None and s.labelled and s.farm == sp["test_farm"]
        ]
        images = sorted(images, key=key_of)
        dest = RESULT
    started = datetime.now(UTC).isoformat(timespec="seconds")

    tile = int(cfg.require("dfine_tile_px"))
    home = predict(
        images, models["home_base"]["path"], tile, int(cfg.require("tile_overlap_px")), None,
        out_dir / "home_base.jsonl", args.threads,
    )  # fmt: skip
    onboard = predict(
        images, models["onboard"]["path"], tile, int(cfg.require("onboard_tile_overlap_px")),
        int(models["onboard"]["long_side_px"]), out_dir / "onboard.jsonl", args.threads,
    )  # fmt: skip

    truth = {key_of(s): s.boxes_xyxy for s in images}
    group_of = {key_of(s): FlightStats(s.farm, s.flight, 0, 0).group for s in images}
    keys = sorted(truth)
    cvt, _ = cv_truth(sp)
    start, stop, step = ev.require("operating_threshold_grid")
    grid = np.round(np.arange(start, stop + 1e-9, step), 4)
    fit_home = calibrate(models["home_base"]["calibration_run"], sp, cvt, grid)
    fit_onb = calibrate(models["onboard"]["calibration_run"], sp, cvt, grid)
    alt_of, gsd_in = meta(keys)
    s1s2 = evaluate(keys, home, truth, dict.fromkeys(keys, "test"), group_of, alt_of,
                    {"test": fit_home}, ev)  # fmt: skip

    link_m, min_animals = float(ev.require("herd_link_m")), int(ev.require("herd_min_animals"))
    rows, groups = [], []
    by_key = {key_of(s): s for s in images}
    for k in keys:
        if k not in gsd_in:
            continue
        alt, f35 = gsd_in[k]
        s = by_key[k]
        gsd = alt / focal_px_from_35mm(f35, s.width, s.height)
        b, sc = onboard[k]
        rows.append(score_image(truth[k], b[sc >= fit_onb["threshold"]], gsd, link_m, min_animals))
        groups.append(group_of[k])
    th = np.array([r.true_herds for r in rows], float)
    fo = np.array([r.found for r in rows], float)
    gr = np.array([r.grouped for r in rows], float)
    ft = np.array([r.false_triggers for r in rows], float)
    rng = np.random.default_rng(int(ev.require("bootstrap_seed")))
    idx, garr = np.arange(len(rows)), np.array(groups)

    def ci(stat):  # noqa: ANN001, ANN202
        if len(set(groups)) < 2:
            return None
        return list(group_bootstrap_ci(idx, garr, stat, int(ev.require("n_boot")), rng))

    s7 = {
        "n_images_scored": len(rows),
        "true_herds": int(th.sum()),
        "herd_recall_found": float(fo.sum() / max(th.sum(), 1)),
        "herd_recall_found_ci95": ci(lambda i: float(fo[i].sum() / max(th[i].sum(), 1))),
        "herd_recall_grouped": float(gr.sum() / max(th.sum(), 1)),
        "herd_recall_grouped_ci95": ci(lambda i: float(gr[i].sum() / max(th[i].sum(), 1))),
        "false_triggers_per_image": float(ft.mean()) if len(ft) else None,
        "latency": "not measured: requires a Raspberry Pi 5 (D-10, D-29)",
    }
    result = {
        "note": "Held-back test farm (ICAERUS Other_farms), scored once. Public data, not a "
        "field result. Thresholds and count ranges fitted on cross-validation only."
        if not args.rehearse
        else "REHEARSAL on 3 cross-validation photos: checks the code path only; not a result.",
        "started_utc": started,
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "test_farm": sp["test_farm"] if not args.rehearse else "fold0 sample",
        "n_photos": len(keys),
        "n_groups": len(set(group_of.values())),
        "n_animals": int(sum(t.shape[0] for t in truth.values())),
        "models": {r: {k: str(v) for k, v in m.items()} for r, m in models.items()},
        "calibration": {
            "home_base": {**fit_home, "q": {f"{lv:.2f}": q for lv, q in fit_home["q"].items()}},
            "onboard": {**fit_onb, "q": {f"{lv:.2f}": q for lv, q in fit_onb["q"].items()}},
        },
        "s1_s2": s1s2,
        "s7": s7,
        "targets": targets_check(s1s2, s7, specs),
        "evaluation_config_hash": ev.hash(),
        "environment": env_report(),
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result["targets"], indent=1))
    print(f"written: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
