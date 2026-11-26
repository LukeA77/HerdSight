"""Farm onboarding curve (G13): how detection on a farm's later survey days improves with the number
of its own labelled photos.

Usage: python scripts/g13_curve.py <kaggle output folder> [<kaggle output folder> ...]

Reads the g13_<farm>_k<k>_test_eval.json files written by gpu/kaggle_g13.py (one folder per farm)
and writes results/phase2/g13_onboarding.json: per farm and k, centre-matched recall and precision,
count error per tile and centre AP on the later-day tiles, and the claim rule fixed before the runs
(_private notes, 25 Nov): the offer is supported for a farm if some k <= 200 reaches recall >= 0.80
and precision >= 0.80. The per-tile scorer picks its count threshold on the scored tiles themselves,
which flatters recall and precision a little at every k alike; centre AP does not use a threshold.
Public data, two farms, one draw of photos per k: an indication, not a field result.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from herdsight.utils.config import REPO_ROOT

RECALL_MIN = PRECISION_MIN = 0.80
K_MAX = 200
AP_GAIN_MIN = 0.02


def main(argv: list[str] | None = None) -> int:
    folders = [Path(a) for a in (argv if argv is not None else sys.argv[1:])]
    if not folders:
        sys.exit(__doc__)
    farms: dict[str, dict] = {}
    for folder in folders:
        for f in sorted(folder.rglob("g13_*_test_eval.json")):
            m = re.match(r"g13_(\w+?)_k(\d+)_test_eval\.json", f.name)
            if not m:
                continue
            farm, k = m.group(1), int(m.group(2))
            e = json.loads(f.read_text(encoding="utf-8"))
            farms.setdefault(farm, {})[k] = {
                "test_tiles": e["tiles"],
                "test_animals": e["animals"],
                "centre_recall": round(e["centre_matched_recall"], 4),
                "centre_precision": round(e["centre_matched_precision"], 4),
                "count_mae_per_tile": round(e["count_mae_per_tile"], 4),
                "centre_ap": round(e["centre_ap"], 4),
            }
        for f in folder.rglob("onb_*_photos.json"):
            rec = json.loads(f.read_text(encoding="utf-8"))
            setup = farms.setdefault(rec["farm"], {}).setdefault(
                "setup", {"pool_days": rec["pool_days"], "seed": rec["seed"], "counts": {}}
            )
            if (setup["pool_days"], setup["seed"]) != (rec["pool_days"], rec["seed"]):
                sys.exit(f"{f}: different pool days or seed from an earlier run of {rec['farm']}")
            setup["counts"].update(rec["counts"])  # later runs (e.g. G13b) add their k
    out: dict = {
        "note": "Public data (ICAERUS), two farms, one random draw of photos per k; tile scores "
        "on the farm's later survey days only. Not a field result.",
        "claim_rule": f"supported if some k <= {K_MAX} reaches centre recall >= {RECALL_MIN} and "
        f"precision >= {PRECISION_MIN}",
        "farms": {},
    }
    for farm, pts in sorted(farms.items()):
        curve = {k: v for k, v in pts.items() if isinstance(k, int)}
        ok = [
            k
            for k, v in curve.items()
            if k <= K_MAX
            and v["centre_recall"] >= RECALL_MIN
            and v["centre_precision"] >= PRECISION_MIN
        ]
        out["farms"][farm] = {
            "setup": pts.get("setup"),
            "curve": {str(k): curve[k] for k in sorted(curve)},
            "supported": bool(ok),
            "fewest_photos_meeting_rule": min(ok) if ok else None,
        }
        beyond = [k for k in curve if k > K_MAX]
        if beyond and K_MAX in curve:
            # G13b rule (fixed before the run): more photos keep helping if centre AP rises by at
            # least AP_GAIN_MIN over k = K_MAX and recall rises; 80/80 there is reported, with the
            # k <= K_MAX limit relaxed.
            k_big, ref = max(beyond), curve[K_MAX]
            big = curve[k_big]
            helps = (
                big["centre_ap"] >= ref["centre_ap"] + AP_GAIN_MIN
                and big["centre_recall"] > ref["centre_recall"]
            )
            out["farms"][farm]["more_photos"] = {
                "k": k_big,
                "rule": f"centre AP at k={k_big} at least {AP_GAIN_MIN} above k={K_MAX} and "
                "recall higher",
                "centre_ap_change": round(big["centre_ap"] - ref["centre_ap"], 4),
                "centre_recall_change": round(big["centre_recall"] - ref["centre_recall"], 4),
                "keeps_helping": helps,
                "reading": "more photos keep helping"
                if helps
                else f"flattens between {K_MAX} and {k_big} photos",
                "meets_80_80_at_k": big["centre_recall"] >= RECALL_MIN
                and big["centre_precision"] >= PRECISION_MIN,
            }
    dest = REPO_ROOT / "results" / "phase2" / "g13_onboarding.json"
    dest.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
