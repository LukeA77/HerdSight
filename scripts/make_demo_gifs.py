"""Two short animated demos for the README: step 1 (the survey) and step 2 (counting the photos).

Usage: python scripts/make_demo_gifs.py [--out docs/figures]

demo_1_survey.gif  Step 1, in simulation: a top-down map of a simulated camp and the drone flying
                   its survey lanes and taking photos, from a stored Gazebo flight (s5_batch_seed5,
                   the flight used by the replay check).
demo_2_count.gif   Step 2, on two real photos: the first two photos of one hand-checked ICAERUS
                   flight (CC BY 4.0), which overlap and were lined up by image alignment. First
                   where the photos were taken and the ground they share, then the two photos side
                   by side with the final detector's marks, a line joining each cow to the same cow
                   in the other photo (the counting method's own grouping: greedy joining, fixed
                   gate), the cows with no partner, and the count next to the hand count of the same
                   two photos. The pictures are turned a quarter turn so the drone moves left to
                   right, as in step 1. Rule for the flight: among hand-checked flights with at least
                   40 cattle, the one whose flight count is closest to its hand count.

Every caption sits in a strip under the picture, never on it, and stays up long enough to read
while watching. Each demo says whether it is simulation or public data; neither is a field result.
Writes the numbers shown to results/readme_demos.json.
"""

from __future__ import annotations

# ruff: noqa: E501  (caption strings)
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from herdsight.count.clustering import greedy_join
from herdsight.count.truth import verified_identities
from herdsight.data.icaerus import default_root, iter_images
from herdsight.evaluate.centre_match import box_centres
from herdsight.utils.config import REPO_ROOT
from herdsight.utils.paths import storage_path

W, PIC_H, STRIP = 800, 450, 100  # picture area, and the caption strip under it
GREEN, YELLOW, RED = (40, 225, 90), (255, 205, 40), (230, 60, 50)
BLUE, PINK, NEW = (70, 165, 255), (255, 110, 190), (255, 255, 255)
INK, MUTED, PAPER = (30, 34, 40), (105, 110, 118), (250, 250, 247)
STRIP_BG = (24, 28, 32)
FONT_DIRS = (Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/dejavu"))


CAPTIONS = REPO_ROOT / "docs" / "demo_captions.md"


def load_captions(path: Path = CAPTIONS) -> dict[str, tuple[str, float | None]]:
    """Caption text and on-screen seconds by code (for example "2-09"), from docs/demo_captions.md."""
    out: dict[str, tuple[str, float | None]] = {}
    code, secs, body = None, None, []
    for line in path.read_text(encoding="utf-8").splitlines() + ["### end"]:
        if line.startswith("### "):
            if code:
                out[code] = (" ".join(x.strip() for x in body if x.strip()), secs)
            head = line[4:]
            code = head.split()[0]
            m = re.search(r"\(([\d.]+) s", head)
            secs = float(m.group(1)) if m else None
            body = []
        elif code and not line.startswith("#"):
            body.append(line)
    return out


CAP = load_captions() if CAPTIONS.exists() else {}


def cap(code: str, default: str, **numbers: object) -> str:
    return CAP.get(code, (default, None))[0].format(**numbers)


def secs(code: str, default_ms: int) -> int:
    t = CAP.get(code, ("", None))[1]
    return int(t * 1000) if t else default_ms


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = (
        ("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf")
        if bold
        else ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf")
    )
    for d in FONT_DIRS:
        for n in names:
            if (d / n).exists():
                return ImageFont.truetype(str(d / n), size)
    return ImageFont.load_default()


def wrap(d: ImageDraw.ImageDraw, text: str, f: ImageFont.ImageFont, width: float) -> list[str]:
    lines = [""]
    for w in text.split():
        trial = (lines[-1] + " " + w).strip()
        if d.textlength(trial, font=f) > width and lines[-1]:
            lines.append(w)
        else:
            lines[-1] = trial
    return lines


def frame(
    pic: Image.Image, text: str, counter: int | None = None, key: list | None = None
) -> Image.Image:
    """The picture with a caption strip underneath it (the words never cover the picture)."""
    out = Image.new("RGB", (W, PIC_H + STRIP), STRIP_BG)
    out.paste(pic, (0, 0))
    d = ImageDraw.Draw(out)
    f = font(23, True)
    right = 0
    if counter is not None:
        right = 190
        d.rounded_rectangle(
            [W - right + 10, PIC_H + 14, W - 14, PIC_H + STRIP - 14], radius=8, fill=(45, 52, 58)
        )
        d.text(
            (W - right + 24, PIC_H + 20),
            cap("L-05", "Cows counted"),
            font=font(15, True),
            fill=(200, 205, 210),
        )
        d.text(
            (W - right + 24, PIC_H + 40), str(counter), font=font(32, True), fill=(255, 255, 255)
        )
    if key:  # a key of coloured dots and words instead of a sentence
        fk = font(26, True)
        total = sum(40 + d.textlength(f"- {w}", font=fk) + 50 for _, _, w in key) - 50
        x = (W - total) / 2
        cy = PIC_H + STRIP / 2
        for fill, edge, word in key:
            d.ellipse([x, cy - 12, x + 24, cy + 12], fill=fill, outline=edge, width=3)
            d.text((x + 36, cy - 17), f"- {word}", font=fk, fill=(255, 255, 255))
            x += 40 + d.textlength(f"- {word}", font=fk) + 50
        return out
    lines = wrap(d, text, f, W - right - 40)
    y = PIC_H + (STRIP - len(lines) * 31) / 2
    for ln in lines:
        x = 20 if counter is not None else (W - d.textlength(ln, font=f)) / 2
        d.text((x, y), ln, font=f, fill=(255, 255, 255))
        y += 31
    return out


def tag(img: Image.Image, text: str, x: int = 10) -> Image.Image:
    """A small label in a top corner of the picture."""
    img = img.copy()
    d = ImageDraw.Draw(img, "RGBA")
    f = font(15, True)
    w = d.textlength(text, font=f)
    d.rounded_rectangle([x, 10, x + 20 + w, 40], radius=6, fill=(18, 22, 26, 200))
    d.text((x + 10, 16), text, font=f, fill=(255, 255, 255))
    return img


def card(lines: list[tuple[str, int, bool, tuple]]) -> Image.Image:
    img = Image.new("RGB", (W, PIC_H + STRIP), PAPER)
    d = ImageDraw.Draw(img)
    total = sum(s + 18 for _, s, _, _ in lines)
    y = (PIC_H + STRIP - total) // 2
    for text, s, bold, col in lines:
        f = font(s, bold)
        for ln in wrap(d, text, f, W - 80):
            d.text(((W - d.textlength(ln, font=f)) / 2, y), ln, font=f, fill=col)
            y += s + 10
        y += 8
    return img


def save_gif(frames: list[Image.Image], durations: list[int], path: Path) -> None:
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=96) for f in frames]
    pal[0].save(
        path, save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=True
    )
    secs = sum(durations) / 1000
    print(f"{path.name}: {len(frames)} frames, {secs:.0f} s, {path.stat().st_size / 1e6:.1f} MB")


# --------------------------------------------------------------------- step 1: the survey (sim)
def demo_survey(out: Path) -> dict:
    name = "s5_batch_seed5"
    run = storage_path("runs", "phase5_sitl", name)
    pose, herd = [], []
    for line in run.joinpath("bbox.jsonl").open(encoding="utf-8"):
        d = json.loads(line)
        if d["sensor"] == "pose":
            pose.append((d["wall_s"], d["sim_s"], d["drone"]))
        elif d["sensor"] == "herd":
            herd.append((d["sim_s"], d["xy"]))
    pose.sort(key=lambda p: p[1])
    herd.sort(key=lambda h: h[0])
    walls = np.array([p[0] for p in pose])
    stills = []
    for row in csv.reader(run.joinpath("stills.csv").open()):
        w = int(row[0]) + int(row[1]) / 1e9
        i = int(np.clip(np.searchsorted(walls, w), 0, len(pose) - 1))
        stills.append((pose[i][1], float(row[4]), float(row[5])))
    plan = yaml.safe_load(run.joinpath("plan.yaml").open())
    fw, fh = plan["footprint_m"]
    sdf = run.joinpath("herdsight_farm.sdf").read_text()
    fence_re = r'<model name="fence_\d+">.*?<pose>([-\d.]+) ([-\d.]+) [-\d.]+ 0 0 ([-\d.]+)</pose>.*?<size>([\d.]+) '
    fences = [tuple(float(v) for v in m.groups()) for m in re.finditer(fence_re, sdf, re.S)]
    ends = [
        (x + np.cos(a) * ln / 2 * k, y + np.sin(a) * ln / 2 * k)
        for x, y, a, ln in fences
        for k in (-1, 1)
    ]
    pts = ends + [(p[2][0], p[2][1]) for p in pose]
    x0, x1 = min(p[0] for p in pts) - 12, max(p[0] for p in pts) + 12
    y0, y1 = min(p[1] for p in pts) - 12, max(p[1] for p in pts) + 12
    margin = 14
    sc = min((W - 2 * margin) / (x1 - x0), (PIC_H - 2 * margin) / (y1 - y0))
    ox = (W - (x1 - x0) * sc) / 2

    def px(x: float, y: float) -> tuple[float, float]:
        return (ox + (x - x0) * sc, PIC_H - margin - (y - y0) * sc)

    grass = Image.new("RGB", (W, PIC_H), (118, 158, 82))
    rng = np.random.default_rng(1)
    gd = ImageDraw.Draw(grass)
    for _ in range(1100):
        cx, cy, r = rng.uniform(0, W), rng.uniform(0, PIC_H), rng.uniform(2, 9)
        t = int(rng.uniform(-14, 14))
        gd.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(118 + t, 158 + t, 82 + t // 2))
    for x, y, a, ln in fences:
        gd.line(
            [
                px(x - np.cos(a) * ln / 2, y - np.sin(a) * ln / 2),
                px(x + np.cos(a) * ln / 2, y + np.sin(a) * ln / 2),
            ],
            fill=(105, 75, 45),
            width=5,
        )
    herd_t = np.array([h[0] for h in herd])
    t0, t1 = pose[0][1], pose[-1][1]

    def scene(t: float) -> tuple[Image.Image, int]:
        img = grass.copy()
        dr = ImageDraw.Draw(img, "RGBA")
        taken = [s for s in stills if s[0] <= t]
        for _, sx, sy in taken:
            a, b = px(sx - fw / 2, sy + fh / 2), px(sx + fw / 2, sy - fh / 2)
            dr.rectangle([a, b], outline=(255, 255, 255, 150), fill=(255, 255, 255, 30))
        hi = int(np.clip(np.searchsorted(herd_t, t), 0, len(herd) - 1))
        for x, y in herd[hi][1]:
            cx, cy = px(x, y)
            dr.ellipse([cx - 6, cy - 4, cx + 6, cy + 4], fill=(250, 250, 245), outline=(40, 40, 40))
        trail = [px(p[2][0], p[2][1]) for p in pose if p[1] <= t]
        if len(trail) > 1:
            dr.line(trail, fill=(250, 210, 60, 230), width=3)
        if trail:
            cx, cy = trail[-1]
            dr.ellipse([cx - 9, cy - 9, cx + 9, cy + 9], fill=RED, outline=(255, 255, 255), width=3)
        return img, len(taken)

    def photos(img: Image.Image, n: int) -> Image.Image:
        img = img.copy()
        d = ImageDraw.Draw(img, "RGBA")
        d.rounded_rectangle([W - 170, 10, W - 10, 78], radius=8, fill=(18, 22, 26, 200))
        d.text(
            (W - 156, 16), cap("L-02", "Photos taken"), font=font(14, True), fill=(220, 225, 230)
        )
        d.text((W - 156, 34), str(n), font=font(30, True), fill=(255, 255, 255))
        return img

    frames = [
        card(
            [
                (cap("1-02", "Step 1"), 24, True, MUTED),
                (cap("1-01", "The drone surveys a camp"), 38, True, INK),
                (cap("1-03", "A simulated drone and farm"), 20, False, MUTED),
            ]
        )
    ]
    durs = [secs("1-01", 2000)]
    # start just before the drone moves off, so the clip does not open on a parked drone
    start = pose[0][2]
    moved = next(p[1] for p in pose if np.hypot(p[2][0] - start[0], p[2][1] - start[1]) > 3)
    t0 = max(t0, moved - 0.02 * (t1 - t0))
    img, _ = scene(t0)
    frames.append(
        frame(
            tag(img, cap("L-01", "Simulation")),
            "",
            key=[
                (RED, (255, 255, 255), cap("1-04", "drone")),
                ((250, 250, 245), (40, 40, 40), cap("1-04b", "cow")),
            ],
        )
    )
    durs.append(secs("1-04", 2000))
    lines = [
        (0.0, cap("1-05", "The drone flies back and forth over the camp.")),
        (0.35, cap("1-06", "It takes a photo every few seconds, and the photos overlap.")),
        (0.75, cap("1-07", "It keeps going until the whole camp is covered.")),
    ]
    n_frames = 80
    for k in range(n_frames + 1):
        t = t0 + (t1 - t0) * k / n_frames
        img, n = scene(t)
        text = [s for at, s in lines if k / n_frames >= at][-1]
        frames.append(frame(photos(tag(img, cap("L-01", "Simulation")), n), text))
        durs.append(110)
    img, n = scene(t1)
    frames.append(
        frame(
            photos(tag(img, cap("L-01", "Simulation")), n),
            cap(
                "1-08",
                "That makes {photos} photos. Next, HerdSight counts the cows in them.",
                photos=n,
            ),
        )
    )
    durs.append(secs("1-08", 3500))
    save_gif(frames, durs, out / "demo_1_survey.gif")
    return {"flight": name, "photos": len(stills)}


# ---------------------------------------------------------------- step 2: counting (real photos)
def pair_data() -> dict:
    """The first two photos of the chosen hand-checked flight, with the detector's marks, the
    counting method's grouping, the image alignment between them and the hand count."""
    g = json.loads((REPO_ROOT / "results/phase3/grouping_real_dfine_ref.json").read_text())
    fl = g["dfine_ref"]["flights"]
    key = min(
        (k for k, v in fl.items() if v["truth"] >= 40),
        key=lambda k: abs(fl[k]["greedy_fixed"] - fl[k]["truth"]) / fl[k]["truth"],
    )
    farm, flight = key.split("/")
    d = np.load(storage_path("processed", "phase3") / "pairs_dfine_ref.npz")
    counts = json.loads(str(d["counts"]))[key][:2]
    det = json.loads(str(d["detections"]))[key]
    m = (d["flight"] == key) & (d["i"] < 2) & (d["j"] < 2)
    cols = (d["i"][m], d["a"][m], d["j"][m], d["b"][m], d["log_odds_fixed"][m])
    pairs = [
        ((int(i), int(a)), (int(j), int(b)), float(lo))
        for i, a, j, b, lo in zip(*cols, strict=True)
    ]
    groups = greedy_join(counts, pairs)
    links = json.loads(
        (storage_path("processed", "s3_links") / f"{farm}__{flight}.json").read_text(
            encoding="utf-8"
        )
    )
    hmat = next(np.asarray(p["H_ab"]) for p in links["pairs"] if (p["i"], p["j"]) == (0, 1))
    hand = verified_identities(key, links).groups
    paths = {
        s.path.name: s.path
        for s, _ in iter_images(default_root(), with_metadata=False)
        if f"{s.farm}/{s.flight}" == key
    }
    boxes = [np.asarray(det["boxes"][i], float).reshape(-1, 4) for i in range(2)]
    # a line joins photo 1's cow a to photo 2's cow b only when the grouping put them together;
    # the grouping never joins two cows of the same photo (one per photo per group)
    lines = []
    for grp in groups:
        in0 = [k for im, k in grp if im == 0]
        in1 = [k for im, k in grp if im == 1]
        assert len(in0) <= 1 and len(in1) <= 1, "a group holds two cows of one photo"
        if in0 and in1:
            lines.append((in0[0], in1[0]))
    return {
        "key": key,
        "imgs": [cv2.imread(str(paths[n])) for n in det["images"][:2]],
        "boxes": boxes,
        "H": hmat,
        "lines": lines,
        "count": len(groups),
        "hand": sum(1 for grp in hand if any(im in (0, 1) for im, _ in grp)),
    }


def to_pil(bgr: np.ndarray, size: tuple[int, int]) -> Image.Image:
    return Image.fromarray(
        cv2.cvtColor(cv2.resize(bgr, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    )


def layout(pd: dict) -> dict:
    """Both photos on one canvas (photo 2 placed by the image alignment), turned a quarter turn so
    the drone's direction of travel points right. Returns each photo alone on that canvas, their
    outlines and the cows' boxes, all in canvas pixels."""
    a, b, hm = pd["imgs"][0], pd["imgs"][1], pd["H"]
    h, w = a.shape[:2]
    c2 = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], float) @ hm.T
    c2 = c2[:, :2] / c2[:, 2:]
    lo = np.minimum(c2.min(0), 0)
    hi = np.maximum(c2.max(0), [w, h])
    off = -lo
    size = (int(hi[0] - lo[0]), int(hi[1] - lo[1]))
    shift = np.array([[1, 0, off[0]], [0, 1, off[1]], [0, 0, 1]])
    only1 = cv2.warpPerspective(a, shift, size)
    only2 = cv2.warpPerspective(b, shift @ hm, size)
    ch = size[1]

    def turn(p: np.ndarray) -> np.ndarray:  # matches cv2.ROTATE_90_CLOCKWISE
        p = np.asarray(p, float).reshape(-1, 2)
        return np.column_stack([ch - 1 - p[:, 1], p[:, 0]])

    def turn_boxes(bx: np.ndarray) -> np.ndarray:
        c = turn(np.vstack([bx[:, :2], bx[:, 2:]]))
        p0, p1 = c[: len(bx)], c[len(bx) :]
        return np.column_stack([np.minimum(p0, p1), np.maximum(p0, p1)])

    b0 = pd["boxes"][0] + np.tile(off, 2)
    c1 = np.column_stack([box_centres(pd["boxes"][1]), np.ones(len(pd["boxes"][1]))]) @ hm.T
    c1 = c1[:, :2] / c1[:, 2:] + off
    half = (pd["boxes"][1][:, 2:] - pd["boxes"][1][:, :2]) / 2
    b1 = np.column_stack([c1 - half, c1 + half])
    rect1 = np.array([[0, 0], [w, 0], [w, h], [0, h]], float) + off
    return {
        "only1": cv2.rotate(only1, cv2.ROTATE_90_CLOCKWISE),
        "only2": cv2.rotate(only2, cv2.ROTATE_90_CLOCKWISE),
        "rect1": turn(rect1),
        "rect2": turn(c2 + off),
        "b0": turn_boxes(b0),
        "b1": turn_boxes(b1),
    }


def demo_count(out: Path) -> dict:
    pd = pair_data()
    L = layout(pd)
    b0, b1 = L["b0"], L["b1"]
    n0, n1 = len(b0), len(b1)
    joined = {a: b for a, b in pd["lines"]}
    partner_of_1 = {b: a for a, b in pd["lines"]}
    alone0 = [k for k in range(n0) if k not in joined]
    new1 = [k for k in range(n1) if k not in partner_of_1]
    n_lines, n_new = len(pd["lines"]), len(new1)
    assert pd["count"] == n0 + n_new
    overlap = cv2.intersectConvexConvex(
        L["rect1"].astype(np.float32), L["rect2"].astype(np.float32)
    )[1].reshape(-1, 2)

    # ---- part 1: where the two photos were taken (the whole of both, with margins)
    both = np.maximum(L["only1"], L["only2"])
    bh, bw = both.shape[:2]
    s = min((W - 60) / bw, (PIC_H - 40) / bh)
    ox, oy = (W - bw * s) / 2, (PIC_H - bh * s) / 2
    over_pic = Image.new("RGB", (W, PIC_H), PAPER)
    over_pic.paste(to_pil(both, (int(bw * s), int(bh * s))), (int(ox), int(oy)))

    def ov(p: np.ndarray) -> list[tuple[float, float]]:
        return [(x * s + ox, y * s + oy) for x, y in np.asarray(p).reshape(-1, 2)]

    def badge(d: ImageDraw.ImageDraw, xy: tuple, text: str, col: tuple) -> None:
        x, y = xy
        d.ellipse([x - 17, y - 17, x + 17, y + 17], fill=col, outline=(255, 255, 255), width=2)
        f = font(20, True)
        d.text((x - d.textlength(text, font=f) / 2, y - 14), text, font=f, fill=(255, 255, 255))

    def overview(show2: bool, shade: bool, box: tuple | None = None) -> Image.Image:
        img = over_pic.copy()
        d = ImageDraw.Draw(img, "RGBA")
        if shade:
            d.polygon(ov(overlap), fill=(255, 255, 255, 90))
        d.polygon(ov(L["rect1"]), outline=BLUE, width=4)
        c1 = np.mean(ov(L["rect1"]), axis=0)
        badge(d, (c1[0], oy + 30), "1", BLUE)
        if show2:
            d.polygon(ov(L["rect2"]), outline=PINK, width=4)
            c2 = np.mean(ov(L["rect2"]), axis=0)
            badge(d, (c2[0], oy + 30), "2", PINK)
            ay = oy + bh * s - 30
            d.line([(c1[0], ay), (c2[0] - 14, ay)], fill=(255, 255, 255), width=5)
            d.polygon(
                [(c2[0], ay), (c2[0] - 18, ay - 11), (c2[0] - 18, ay + 11)], fill=(255, 255, 255)
            )
            f = font(16, True)
            label = cap("L-04", "drone flying this way")
            tw = d.textlength(label, font=f)
            mx = (c1[0] + c2[0]) / 2
            d.rounded_rectangle(
                [mx - tw / 2 - 8, ay - 38, mx + tw / 2 + 8, ay - 12],
                radius=6,
                fill=(18, 22, 26, 200),
            )
            d.text((mx - tw / 2, ay - 35), label, font=f, fill=(255, 255, 255))
        if box is not None:
            x, y, bw_, bh_ = box
            (ax, ay_), (bx_, by) = ov(np.array([[x, y], [x + bw_, y + bh_]]))
            ax, bx_ = max(ax, ox), min(bx_, ox + bw * s)
            ay_, by = max(ay_, oy), min(by, oy + bh * s)
            d.rectangle([ax, ay_, bx_, by], outline=(255, 255, 255), width=4)
        return img

    # ---- part 2: the two photos side by side, the same size. Photo 2's view is moved along by the
    # distance the drone flew between the photos, so photo 2 shows new ground on its right and the
    # herd sits further left in it, as it would on the camera.
    gap = 8
    pw = (W - gap) // 2
    move = L["rect2"].mean(0) - L["rect1"].mean(0)
    # the window must hold photo 1's cows, and photo 2's cows once moved back by the same distance
    cent = np.vstack([box_centres(b0), box_centres(b1) - move])
    lo, hi = cent.min(0), cent.max(0)
    span = (hi - lo) * 1.08 + 60
    gw = max(span[0], span[1] * pw / PIC_H)
    gh = gw * PIC_H / pw
    r1lo, r1hi = L["rect1"].min(0), L["rect1"].max(0)
    r2lo, r2hi = L["rect2"].min(0), L["rect2"].max(0)
    # photo 1's window: around the herd, inside photo 1, and moved along it stays inside photo 2
    xmin = max(r1lo[0], r2lo[0] - move[0])
    xmax = min(r1hi[0], r2hi[0] - move[0]) - gw
    ymin = max(r1lo[1], r2lo[1] - move[1])
    ymax = min(r1hi[1], r2hi[1] - move[1]) - gh
    mid = (lo + hi) / 2
    wx = float(np.clip(mid[0] - gw / 2, xmin, max(xmin, xmax)))
    wy = float(np.clip(mid[1] - gh / 2, ymin, max(ymin, ymax)))
    win = (wx, wy, float(gw), float(gh))
    win2 = (wx + float(move[0]), wy + float(move[1]), float(gw), float(gh))
    k = pw / gw

    def panel(img: np.ndarray, w: tuple) -> Image.Image:
        x, y, w_, h_ = w
        pad = np.zeros((int(h_) + 2, int(w_) + 2, 3), np.uint8)
        xs, ys = max(0, int(x)), max(0, int(y))
        xe, ye = min(img.shape[1], int(x + w_)), min(img.shape[0], int(y + h_))
        pad[ys - int(y) : ye - int(y), xs - int(x) : xe - int(x)] = img[ys:ye, xs:xe]
        return to_pil(pad, (pw, PIC_H))

    p1, p2 = panel(L["only1"], win), panel(L["only2"], win2)

    def pv(p: np.ndarray, right: bool) -> np.ndarray:
        w = win2 if right else win
        q = (np.asarray(p, float).reshape(-1, 2) - [w[0], w[1]]) * k
        return q + ([pw + gap, 0] if right else [0, 0])

    v0 = pv(b0, False).reshape(-1, 4)
    v1 = pv(b1, True).reshape(-1, 4)
    c0, c1v = box_centres(v0), box_centres(v1)
    for a, b in pd["lines"]:  # every line runs from the left photo to the right photo
        assert c0[a][0] < pw and c1v[b][0] > pw + gap, "a line would stay inside one photo"

    def sides(shade: bool = True) -> Image.Image:
        img = Image.new("RGB", (W, PIC_H), (0, 0, 0))
        img.paste(p1, (0, 0))
        img.paste(p2, (pw + gap, 0))
        d = ImageDraw.Draw(img, "RGBA")
        if shade:
            mask = Image.new("L", (W, PIC_H), 110)
            md = ImageDraw.Draw(mask)
            md.polygon([tuple(p) for p in pv(overlap, False)], fill=0)
            md.polygon([tuple(p) for p in pv(overlap, True)], fill=0)
            img.paste(Image.new("RGB", (W, PIC_H), (0, 0, 0)), (0, 0), mask)
            d = ImageDraw.Draw(img, "RGBA")
        d.rectangle([pw, 0, pw + gap, PIC_H], fill=(0, 0, 0))
        img = tag(img, cap("L-03", "Photo 1"), 10)
        return tag(img, cap("L-03b", "Photo 2"), pw + gap + 10)

    def boxes(img: Image.Image, which: list, v: np.ndarray, col: tuple, width: int = 3) -> None:
        d = ImageDraw.Draw(img)
        for i in which:
            x0, y0, x1, y1 = v[i]
            d.rectangle([x0 - 2, y0 - 2, x1 + 2, y1 + 2], outline=col, width=width)

    def draw_lines(img: Image.Image, upto: int, alpha: int = 255) -> None:
        d = ImageDraw.Draw(img, "RGBA")
        for a, b in pd["lines"][:upto]:
            d.line([tuple(c0[a]), tuple(c1v[b])], fill=(0, 0, 0, min(alpha, 160)), width=5)
            d.line([tuple(c0[a]), tuple(c1v[b])], fill=(*YELLOW, alpha), width=2)

    def ring(img: Image.Image, which: list, v: np.ndarray, col: tuple) -> None:
        d = ImageDraw.Draw(img)
        for i in which:
            cx, cy = box_centres(v[i : i + 1])[0]
            d.ellipse([cx - 17, cy - 17, cx + 17, cy + 17], outline=(0, 0, 0), width=6)
            d.ellipse([cx - 17, cy - 17, cx + 17, cy + 17], outline=col, width=3)

    frames: list[Image.Image] = [
        card(
            [
                (cap("2-02", "Step 2"), 24, True, MUTED),
                (cap("2-01", "Counting the cows in the photos"), 36, True, INK),
                (
                    cap(
                        "2-03", "Two real photos from one survey flight, taken a few seconds apart"
                    ),
                    19,
                    False,
                    MUTED,
                ),
            ]
        )
    ]
    durs = [secs("2-01", 3000)]
    frames.append(
        frame(
            overview(False, False), cap("2-04", "The blue frame is the first photo the drone took.")
        )
    )
    durs.append(secs("2-04", 3200))
    frames.append(
        frame(
            overview(True, False),
            cap(
                "2-05",
                "The drone flew on a little and took a second photo, shown by the pink frame.",
            ),
        )
    )
    durs.append(secs("2-05", 4200))
    frames.append(
        frame(
            overview(True, True),
            cap("2-06", "The two photos overlap. Any cow in the light area is in both photos."),
        )
    )
    durs.append(secs("2-06", 4500))
    frames.append(
        frame(
            overview(True, True, win),
            cap("2-07", "The white box marks the herd. Next, we look at that area in both photos."),
        )
    )
    durs.append(secs("2-07", 4000))

    base = sides()
    frames.append(
        frame(
            base,
            cap(
                "2-08",
                "On the left is the white box in photo 1. On the right is the same ground in photo 2. The darker parts are in only one of the photos.",
            ),
        )
    )
    durs.append(secs("2-08", 5200))
    marked = base.copy()
    boxes(marked, list(range(n0)), v0, GREEN)
    boxes(marked, list(range(n1)), v1, GREEN)
    frames.append(
        frame(
            marked,
            cap(
                "2-09",
                "HerdSight puts a green box on every cow it finds: {photo1_cows} in photo 1 and {photo2_cows} in photo 2.",
                photo1_cows=n0,
                photo2_cows=n1,
            ),
            counter=n0,
        )
    )
    durs.append(secs("2-09", 5000))
    # a few example lines first, chosen so each one has a clear path (no other cow close to it)
    allc = np.vstack([c0, c1v])

    def clearance(a: int, b: int) -> float:
        p, q = c0[a], c1v[b]
        t = np.clip(((allc - p) @ (q - p)) / max(float((q - p) @ (q - p)), 1e-9), 0, 1)
        dist = np.hypot(*(allc - (p + t[:, None] * (q - p))).T)
        far = np.hypot(*(allc - p).T) > 12
        far &= np.hypot(*(allc - q).T) > 12
        return float(dist[far].min()) if far.any() else 999.0

    examples = sorted(pd["lines"], key=lambda ab: -clearance(*ab))[:4]
    ex = marked.copy()
    d = ImageDraw.Draw(ex)
    for n, (a, b) in enumerate(examples, 1):
        boxes(ex, [a], v0, YELLOW, 4)
        boxes(ex, [b], v1, YELLOW, 4)
        p, q = tuple(c0[a]), tuple(c1v[b])
        d.line([p, q], fill=(0, 0, 0), width=7)
        d.line([p, q], fill=YELLOW, width=4)
        for x, y in (p, q):
            d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=YELLOW, outline=(0, 0, 0))
        frames.append(
            frame(
                ex.copy(),
                cap("2-10", "A yellow line joins a cow in photo 1 to the same cow in photo 2.")
                if n < 3
                else cap("2-11", "These are cows HerdSight has seen twice: once in each photo."),
                counter=n0,
            )
        )
        durs.append(1600 if n < len(examples) else 3200)
    matched = marked.copy()
    boxes(matched, [a for a, _ in pd["lines"]], v0, YELLOW, 4)
    boxes(matched, [b for _, b in pd["lines"]], v1, YELLOW, 4)
    lined = matched.copy()
    draw_lines(lined, n_lines, alpha=170)
    frames.append(
        frame(
            lined,
            cap(
                "2-12",
                "It finds {lines} cows like that. Each one gets a line, and its boxes turn yellow in both photos.",
                lines=n_lines,
            ),
            counter=n0,
        )
    )
    durs.append(secs("2-12", 5200))
    frames.append(
        frame(
            matched,
            cap(
                "2-13",
                "The yellow cows in photo 2 were already counted in photo 1, so they are not counted again.",
            ),
            counter=n0,
        )
    )
    durs.append(secs("2-13", 5000))
    left = matched.copy()
    ring(left, alone0, v0, (120, 200, 255))
    frames.append(
        frame(
            left,
            cap(
                "2-14",
                "The {photo1_alone} circled cows in photo 1 stay green.",
                photo1_alone=len(alone0),
            ),
            counter=n0,
        )
    )
    durs.append(secs("2-14", 6200))
    right = matched.copy()
    ring(right, new1, v1, NEW)
    frames.append(
        frame(
            right,
            cap("2-15", "The {new} circled cows in photo 2 stay green too.", new=n_new),
            counter=n0,
        )
    )
    durs.append(secs("2-15", 5500))
    frames.append(
        frame(
            right,
            cap(
                "2-16",
                "So the count is {photo1_cows} plus {new}: {count} cows.",
                photo1_cows=n0,
                new=n_new,
                count=pd["count"],
            ),
            counter=pd["count"],
        )
    )
    durs.append(secs("2-16", 4500))
    diff = pd["count"] - pd["hand"]
    tail = (
        cap("2-19", "It counted {difference} too many.", difference=diff)
        if diff > 0
        else (
            cap("2-19b", "It counted {difference} too few.", difference=-diff)
            if diff < 0
            else cap("2-19c", "It got exactly the same number.")
        )
    )
    head = [
        (
            cap("2-17", "HerdSight counted {count} different cows", count=pd["count"]),
            34,
            True,
            (40, 120, 70),
        ),
        (cap("2-18", "Counted by hand: {hand}", hand=pd["hand"]), 24, False, INK),
    ]
    frames += [card(head), card([*head, (tail, 18, False, MUTED)])]
    durs += [secs("2-17", 3000), secs("2-19", 5000)]
    save_gif(frames, durs, out / "demo_2_count.gif")
    return {
        "flight": pd["key"],
        "photos": 2,
        "cows_found": [n0, n1],
        "lines": n_lines,
        "photo_1_without_line": len(alone0),
        "new_in_photo_2": n_new,
        "count": pd["count"],
        "hand_count": pd["hand"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "figures")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    info = {
        "note": "Step 1 is simulation; step 2 uses public drone photos (ICAERUS). Not field results.",
        "survey": demo_survey(args.out),
        "count": demo_count(args.out),
    }
    (REPO_ROOT / "results" / "readme_demos.json").write_text(
        json.dumps(info, indent=1) + "\n", encoding="utf-8"
    )
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
