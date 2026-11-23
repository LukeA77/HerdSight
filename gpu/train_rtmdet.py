"""Phase 2 step G7: train RTMDet-R (mmrotate 1.0.0rc1) on oriented tile labels.

Usage: python gpu/train_rtmdet.py --run g7_fold0 --labels DIR --images DIR --fold fold0 --work DIR

Uses the authors' rotated_rtmdet_s-3x-dota config: its 36-epoch schedule, optimiser, batch of 8,
flips, rotations and ImageNet-pretrained backbone. The only changes are one class (cattle), 640 px
tiles, the project's dataset, a lower score floor so the operating threshold is fitted later, and
at most 300 detections per tile, as for D-FINE.
No checkpoint is chosen on validation data: the last one is kept.

--cpu-smoke N is a local check on N tiles for one epoch, with a box loss that runs on CPU. A model
from that check is never kept. Writes WORK/RUN/: last.pth, config.py and run.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path

import mmrotate
import torch
from mmengine.config import Config
from mmengine.dataset import BaseDataset
from mmengine.runner import Runner
from mmrotate.registry import DATASETS
from mmrotate.utils import register_all_modules

TILE = 640
EPOCHS = 36
BASE = "rotated_rtmdet/rotated_rtmdet_s-3x-dota.py"


def quad(o: list) -> list:
    cx, cy, w, h, a = o
    ux, uy = math.cos(a) * w / 2, math.sin(a) * w / 2
    vx, vy = -math.sin(a) * h / 2, math.cos(a) * h / 2
    return [
        cx - ux - vx,
        cy - uy - vy,
        cx + ux - vx,
        cy + uy - vy,
        cx + ux + vx,
        cy + uy + vy,
        cx - ux + vx,
        cy - uy + vy,
    ]


@DATASETS.register_module()
class HerdObbDataset(BaseDataset):
    """COCO-style tiles whose annotations carry an "obb"."""

    METAINFO = {"classes": ("cattle",), "palette": [(40, 230, 70)]}

    def load_data_list(self) -> list:
        coco = json.loads(Path(self.ann_file).read_text())
        anns: dict = {}
        for a in coco["annotations"]:
            anns.setdefault(a["image_id"], []).append(a)
        out = []
        for im in coco["images"]:
            out.append(
                {
                    "img_id": im["id"],
                    "file_name": im["file_name"],
                    "img_path": os.path.join(self.data_prefix["img_path"], im["file_name"]),
                    "height": im["height"],
                    "width": im["width"],
                    "instances": [
                        {"bbox": quad(a["obb"]), "bbox_label": 0, "ignore_flag": 0}
                        for a in anns.get(im["id"], [])
                    ],
                }
            )
        return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--fold", required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu-smoke", type=int, default=0)
    args = ap.parse_args()
    if not args.cpu_smoke and not torch.cuda.is_available():
        raise SystemExit("no GPU")
    register_all_modules()
    out = (args.work / args.run).resolve()
    out.mkdir(parents=True, exist_ok=True)
    ann = args.labels.resolve() / f"train_{args.fold}.json"
    if args.cpu_smoke:
        coco = json.loads(ann.read_text())
        keep = sorted({a["image_id"] for a in coco["annotations"]})[: args.cpu_smoke]
        coco["images"] = [i for i in coco["images"] if i["id"] in keep]
        coco["annotations"] = [a for a in coco["annotations"] if a["image_id"] in keep]
        ann = out / "smoke.json"
        ann.write_text(json.dumps(coco))
    root = Path(mmrotate.__file__).parent / ".mim" / "configs"
    cfg = Config.fromfile(str(root / BASE))
    cfg.work_dir = str(out)
    cfg.randomness = {"seed": args.seed}
    cfg.model.bbox_head.num_classes = 1
    cfg.model.test_cfg.score_thr = 0.01
    cfg.model.test_cfg.max_per_img = 300  # per tile, the same limit as D-FINE's 300 queries
    workers = 0 if args.cpu_smoke else args.workers
    pipeline = [
        {"type": "mmdet.LoadImageFromFile"},
        {"type": "mmdet.LoadAnnotations", "with_bbox": True, "box_type": "qbox"},
        {"type": "ConvertBoxType", "box_type_mapping": {"gt_bboxes": "rbox"}},
        {"type": "mmdet.Resize", "scale": (TILE, TILE), "keep_ratio": True},
        {
            "type": "mmdet.RandomFlip",
            "prob": 0.75,
            "direction": ["horizontal", "vertical", "diagonal"],
        },
        {"type": "RandomRotate", "prob": 0.5, "angle_range": 180, "rect_obj_labels": []},
        {"type": "mmdet.Pad", "size": (TILE, TILE), "pad_val": {"img": (114, 114, 114)}},
        {"type": "mmdet.PackDetInputs"},
    ]
    cfg.train_dataloader = {
        "batch_size": 2 if args.cpu_smoke else 8,
        "num_workers": workers,
        "persistent_workers": workers > 0,
        "pin_memory": False,
        "sampler": {"type": "DefaultSampler", "shuffle": True},
        "batch_sampler": None,
        "dataset": {
            "type": "HerdObbDataset",
            "ann_file": str(ann),
            "data_root": "",
            "data_prefix": {"img_path": str(args.images.resolve())},
            "pipeline": pipeline,
        },
    }
    for name in ("val", "test"):
        cfg[f"{name}_dataloader"] = None
        cfg[f"{name}_cfg"] = None
        cfg[f"{name}_evaluator"] = None
    epochs = 1 if args.cpu_smoke else EPOCHS
    cfg.train_cfg = {"type": "EpochBasedTrainLoop", "max_epochs": epochs}
    cfg.default_hooks.checkpoint = {
        "type": "CheckpointHook",
        "interval": epochs,
        "max_keep_ckpts": 1,
    }
    if args.cpu_smoke:
        cfg.model.bbox_head.loss_bbox = {"type": "GDLoss", "loss_type": "gwd", "loss_weight": 5.0}
        cfg.param_scheduler = [
            {"type": "LinearLR", "start_factor": 0.1, "by_epoch": False, "begin": 0, "end": 2}
        ]
        cfg.default_hooks.logger = {"type": "LoggerHook", "interval": 1}
    t0 = time.time()
    runner = Runner.from_cfg(cfg)
    runner.train()
    shutil.copy2(out / f"epoch_{epochs}.pth", out / "last.pth")
    (out / f"epoch_{epochs}.pth").unlink()
    cfg.dump(str(out / "config.py"))
    record = {
        "run": args.run,
        "fold": args.fold,
        "base_config": BASE,
        "epochs": epochs,
        "seed": args.seed,
        "cpu_smoke": bool(args.cpu_smoke),
        "seconds": round(time.time() - t0, 1),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "versions": {
            "torch": torch.__version__,
            "mmrotate": mmrotate.__version__,
        },
        "last_pth_sha256": sha256(out / "last.pth"),
    }
    (out / "run.json").write_text(json.dumps(record, indent=1) + "\n")
    print(f"{args.run} finished in {record['seconds']} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
