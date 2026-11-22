"""Draw the S4 trade-off curve (Phase 4 exit, brief §12) from the 4000-farm validation run.

Usage: python scripts/plot_s4.py

Reads results/phase4/s4_validation_4000farms.json and writes docs/figures/s4_curve_validation.svg:
for each count error model, false alarms per 100 surveys (log scale) against the share of losses
of 5 or more animals caught within 3 surveys, one line per alert rule across the interval levels (the dial), with 95% intervals over simulated farms,
and the frozen S4 target (configs/specs.yaml s4_alert_targets) shaded. Plain SVG, no plotting
library. Simulation results, never field results.
"""

# ruff: noqa: E501  (SVG markup lines)

from __future__ import annotations

import json
import math
import sys

from herdsight.utils.config import REPO_ROOT, load_config

RULES = (  # fixed colour and marker per rule; drawn in this order (recount on top)
    ("interval", "Interval", "#2a78d6", "circle"),
    ("farm", "Whole farm", "#1baf7a", "diamond"),
    ("recount", "Recount", "#eb6834", "square"),
    ("history", "History", "#6250d6", "triangle"),
)
MODELS = (
    ("measured_rho0.0", "Phase 1 count error, independent checks"),
    ("target_rho0.0", "S3 target count error, independent checks"),
    ("measured_rho0.5", "Phase 1 count error, correlated checks (0.5)"),
    ("target_rho0.5", "S3 target count error, correlated checks (0.5)"),
)
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3de", "#ffffff"
X_MIN, X_MAX = 0.1, 50.0
PW, PH, ML, MT, GAP_X, GAP_Y = 380, 250, 64, 92, 56, 84


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def marker(shape: str, x: float, y: float, colour: str) -> str:
    ring = f'stroke="{SURFACE}" stroke-width="2"'
    if shape == "circle":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{colour}" {ring}/>'
    if shape == "square":
        return f'<rect x="{x - 4.5:.1f}" y="{y - 4.5:.1f}" width="9" height="9" rx="1.5" fill="{colour}" {ring}/>'
    if shape == "triangle":
        pts = f"{x:.1f},{y - 6.5:.1f} {x + 6:.1f},{y + 4.5:.1f} {x - 6:.1f},{y + 4.5:.1f}"
        return f'<polygon points="{pts}" fill="{colour}" {ring}/>'
    pts = f"{x:.1f},{y - 6:.1f} {x + 6:.1f},{y:.1f} {x:.1f},{y + 6:.1f} {x - 6:.1f},{y:.1f}"
    return f'<polygon points="{pts}" fill="{colour}" {ring}/>'


def panel(runs: dict, model: str, title: str, ox: float, oy: float, target: dict) -> list[str]:
    def sx(v: float) -> float:
        v = min(max(v, X_MIN), X_MAX)
        return ox + PW * (math.log10(v) - math.log10(X_MIN)) / math.log10(X_MAX / X_MIN)

    def sy(v: float) -> float:
        return oy + PH * (1 - v)

    out = [
        f'<text x="{ox}" y="{oy - 14}" font-size="13" font-weight="600" fill="{INK}">{esc(title)}</text>'
    ]
    tx, ty = (
        sx(target["false_alarms_per_100_surveys_max"]),
        sy(target["losses_of_5_plus_caught_within_3_surveys_min"]),
    )
    out.append(
        f'<rect x="{ox}" y="{oy}" width="{tx - ox:.1f}" height="{ty - oy:.1f}" fill="#eceae4"/>'
    )
    out.append(f'<text x="{ox + 6}" y="{oy + 15}" font-size="11" fill="{MUTED}">S4 target</text>')
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        y = sy(v)
        out.append(f'<line x1="{ox}" x2="{ox + PW}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}"/>')
        out.append(
            f'<text x="{ox - 8}" y="{y + 4:.1f}" font-size="11" fill="{MUTED}" text-anchor="end">{int(v * 100)}%</text>'
        )
    for v in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50):
        x = sx(v)
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{oy}" y2="{oy + PH}" stroke="{GRID}"/>')
        out.append(
            f'<text x="{x:.1f}" y="{oy + PH + 16}" font-size="11" fill="{MUTED}" text-anchor="middle">{v:g}</text>'
        )
    naive = runs[(model, "naive")]
    nx, ny = sx(naive["false_alarms_per_100_surveys"]), sy(naive["k5_caught_share"])
    out.append(
        f'<circle cx="{nx:.1f}" cy="{ny:.1f}" r="5" fill="none" stroke="{MUTED}" stroke-width="2"/>'
    )
    out.append(
        f'<text x="{nx - 9:.1f}" y="{ny + 4:.1f}" font-size="11" fill="{MUTED}" text-anchor="end">Naive</text>'
    )
    for key, _label, colour, shape in RULES:
        pts = sorted(
            (
                r
                for (m, k), r in runs.items()
                if m == model and k.startswith(f"{key}@") and "_" not in k
            ),
            key=lambda r: float(r["rule"].split("@")[1]),
        )
        xy = [(sx(r["false_alarms_per_100_surveys"]), sy(r["k5_caught_share"])) for r in pts]
        for r, (x, y) in zip(pts, xy, strict=True):
            lo, hi = r["false_alarms_per_100_surveys_ci95"]
            clo, chi = r["k5_caught_share_ci95"]
            out.append(
                f'<path d="M{sx(lo):.1f},{y:.1f}H{sx(hi):.1f}M{x:.1f},{sy(clo):.1f}V{sy(chi):.1f}" stroke="{colour}" stroke-opacity="0.45" stroke-width="1"/>'
            )
        out.append(
            f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in xy)}" fill="none" stroke="{colour}" stroke-width="2"/>'
        )
        out += [marker(shape, x, y, colour) for x, y in xy]
        if key == "interval":
            for r, (x, y) in zip(pts, xy, strict=True):
                lv = round(float(r["rule"].split("@")[1]) * 100)
                out.append(
                    f'<text x="{x + 4:.1f}" y="{y + 17:.1f}" font-size="10" fill="{MUTED}">{lv}%</text>'
                )
    out.append(f'<rect x="{ox}" y="{oy}" width="{PW}" height="{PH}" fill="none" stroke="{GRID}"/>')
    return out


def main() -> int:
    d = json.loads((REPO_ROOT / "results" / "phase4" / "s4_validation_4000farms.json").read_text())
    if d["set"] != "validation":
        raise SystemExit("the figure is drawn from the validation set")
    target = load_config("specs").require("s4_alert_targets")
    runs = {(r["error_model"], r["rule"]): r for r in d["runs"]}
    width = ML + 2 * PW + GAP_X + 30
    height = MT + 2 * PH + GAP_Y + 76
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" font-family="Segoe UI, Helvetica, Arial, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="{SURFACE}"/>',
        f'<text x="{ML}" y="26" font-size="16" font-weight="600" fill="{INK}">S4: false alarms against losses caught, by alert rule and dial setting</text>',
        f'<text x="{ML}" y="46" font-size="12" fill="{MUTED}">{d["farms"]} simulated farms, validation set: simulation, not field results. Bars: 95% intervals over farms. Labels: interval level (the dial).</text>',
    ]
    lx = ML
    for _, label, colour, shape in RULES:
        svg.append(marker(shape, lx + 6, 66, colour))
        svg.append(f'<text x="{lx + 16}" y="70" font-size="12" fill="{INK}">{label}</text>')
        lx += 30 + 8 * len(label) + 20
    svg.append(
        f'<circle cx="{lx + 6}" cy="66" r="5" fill="none" stroke="{MUTED}" stroke-width="2"/>'
    )
    svg.append(
        f'<text x="{lx + 16}" y="70" font-size="12" fill="{INK}">Naive (count below records)</text>'
    )
    for i, (model, title) in enumerate(MODELS):
        ox = ML + (i % 2) * (PW + GAP_X)
        oy = MT + 20 + (i // 2) * (PH + GAP_Y)
        svg += panel(runs, model, title, ox, oy, target)
    by = height - 14
    svg.append(
        f'<text x="{ML + PW + GAP_X / 2}" y="{by}" font-size="12" fill="{INK}" text-anchor="middle">False alarms per 100 drone checks (log scale)</text>'
    )
    svg.append(
        f'<text transform="translate(16 {MT + 20 + PH + GAP_Y / 2}) rotate(-90)" font-size="12" fill="{INK}" text-anchor="middle">Losses of 5+ animals caught within 3 checks</text>'
    )
    svg.append("</svg>")
    dest = REPO_ROOT / "docs" / "figures" / "s4_curve_validation.svg"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(svg) + "\n", encoding="utf-8")
    print("written", dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
