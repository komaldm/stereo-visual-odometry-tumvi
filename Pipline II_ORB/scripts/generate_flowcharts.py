"""
Publication-quality VO pipeline flowcharts.
Generates monocular, stereo, and combined comparison diagrams.
Uses graphviz (if dot binary available) or matplotlib fallback.
Outputs: outputs/flowcharts/*.{png,svg,pdf}
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs" / "flowcharts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Attempt graphviz; fall back to matplotlib
# ──────────────────────────────────────────────────────────────────────────────
try:
    import subprocess
    subprocess.run(["dot", "-V"], capture_output=True, check=True)
    HAS_GRAPHVIZ_BIN = True
except Exception:
    HAS_GRAPHVIZ_BIN = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
from matplotlib.font_manager import FontProperties
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  LOW-LEVEL DRAWING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

FONT_BOLD = FontProperties(family="DejaVu Sans", weight="bold", size=8.5)
FONT_NORM = FontProperties(family="DejaVu Sans", size=8)
FONT_SMALL = FontProperties(family="DejaVu Sans", size=7)
FONT_TINY = FontProperties(family="DejaVu Sans", size=6.5)

# Section palette
PAL = {
    "input":    ("#e8f5e9", "#2e7d32"),   # green fill / border
    "init":     ("#e0f4f8", "#00838f"),   # teal
    "track":    ("#fff8e1", "#f57f17"),   # amber
    "pose":     ("#fde8c8", "#e65100"),   # orange
    "fallback": ("#fce4ec", "#c62828"),   # red
    "map":      ("#ede7f6", "#6a1b9a"),   # purple
    "output":   ("#f0f0f0", "#37474f"),   # gray
    "decision": ("#fffde7", "#f9a825"),   # yellow
    "stereo":   ("#e3f2fd", "#1565c0"),   # blue
    "recovery": ("#fff3e0", "#bf360c"),   # deep orange
    "scale_amb":("#fafafa", "#b71c1c"),   # mono scale marker
    "scale_met":("#fafafa", "#0d47a1"),   # stereo scale marker
}


def _draw_rounded_box(ax, x, y, w, h, fill, border, label, fontprop=None,
                      alpha=1.0, zorder=3, linewidth=1.4):
    """Draw a rounded rectangle with centred text."""
    fp = fontprop or FONT_NORM
    rx = x - w / 2
    ry = y - h / 2
    box = FancyBboxPatch(
        (rx, ry), w, h,
        boxstyle="round,pad=0.015",
        facecolor=fill, edgecolor=border, linewidth=linewidth,
        alpha=alpha, zorder=zorder,
    )
    ax.add_patch(box)
    ax.text(
        x, y, label,
        ha="center", va="center",
        fontproperties=fp,
        color="#1a1a1a",
        wrap=False,
        zorder=zorder + 1,
        multialignment="center",
    )


def _draw_diamond(ax, x, y, w, h, fill, border, label, fontprop=None, zorder=3):
    fp = fontprop or FONT_NORM
    pts = np.array([
        [x,       y + h/2],
        [x + w/2, y      ],
        [x,       y - h/2],
        [x - w/2, y      ],
    ])
    poly = Polygon(pts, closed=True, facecolor=fill, edgecolor=border,
                   linewidth=1.5, zorder=zorder)
    ax.add_patch(poly)
    ax.text(x, y, label, ha="center", va="center",
            fontproperties=fp or FONT_SMALL,
            color="#1a1a1a", zorder=zorder + 1, multialignment="center")


def _arrow(ax, x0, y0, x1, y1, label="", color="#444444",
           connectionstyle="arc3,rad=0.0", zorder=2):
    ax.annotate(
        "",
        xy=(x1, y1), xycoords="data",
        xytext=(x0, y0), textcoords="data",
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=1.3,
            connectionstyle=connectionstyle,
        ),
        zorder=zorder,
    )
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx + 0.02, my, label, fontproperties=FONT_TINY,
                color=color, ha="left", va="center", zorder=zorder + 1)


def _section_label(ax, x, y, text, color):
    ax.text(x, y, text, fontproperties=FONT_BOLD,
            color=color, ha="left", va="top", zorder=5,
            bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                      edgecolor=color, alpha=0.85, linewidth=1.0))


def _section_background(ax, x0, y0, x1, y1, fill, border, alpha=0.28, zorder=0):
    """Shaded region for a pipeline section."""
    rx, ry = min(x0, x1), min(y0, y1)
    rw, rh = abs(x1 - x0), abs(y1 - y0)
    bg = FancyBboxPatch(
        (rx, ry), rw, rh,
        boxstyle="round,pad=0.02",
        facecolor=fill, edgecolor=border, linewidth=1.2,
        alpha=alpha, zorder=zorder,
    )
    ax.add_patch(bg)


# ══════════════════════════════════════════════════════════════════════════════
#  MONOCULAR FLOWCHART
# ══════════════════════════════════════════════════════════════════════════════

def build_monocular_flowchart():
    FIG_W, FIG_H = 18, 34
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # ── Coordinates (x, y) – y increases downward in logical space,
    #    but matplotlib uses y-up. We flip: y_plot = FIG_H - y_logical.
    def Y(y_log):
        return FIG_H - y_log

    BW = 3.6    # box width  (wide)
    BH = 0.52   # box height
    NW = 3.0    # narrow box
    DW = 3.0    # decision diamond width
    DH = 0.62   # decision diamond height
    X_LEFT = 3.8
    X_MID  = 9.0
    X_RIGHT = 14.2

    # ─────────────────────────── SECTION A: INPUT ───────────────────────────
    _section_background(ax, 0.3, Y(3.5), 8.2, Y(0.3),
                        *PAL["input"][:2], alpha=0.22)
    _section_label(ax, 0.5, Y(0.4), "A  INPUT & STATE", PAL["input"][1])

    nodes_A = [
        (X_LEFT, Y(0.9),  BW, BH, "TUM VI Dataset Loader\n(room2_512_16)"),
        (X_LEFT, Y(1.7),  BW, BH, "Frame Read  |  Timestamp (ns)"),
        (X_LEFT, Y(2.5),  BW, BH, "Fisheye Undistortion\n(balance=0.0, new K)"),
        (X_LEFT, Y(3.2),  BW, BH, "ORB Detection\n(nfeatures=2000, 8 levels, scale=1.2)"),
    ]
    for (x, y, w, h, lbl) in nodes_A:
        _draw_rounded_box(ax, x, y, w, h, *PAL["input"], lbl)
    _arrow(ax, X_LEFT, Y(0.9)-BH/2, X_LEFT, Y(1.7)+BH/2)
    _arrow(ax, X_LEFT, Y(1.7)-BH/2, X_LEFT, Y(2.5)+BH/2)
    _arrow(ax, X_LEFT, Y(2.5)-BH/2, X_LEFT, Y(3.2)+BH/2)

    # ─────────────────────────── SECTION INIT ───────────────────────────────
    _section_background(ax, 0.3, Y(8.5), 8.2, Y(3.6),
                        *PAL["init"][:2], alpha=0.20)
    _section_label(ax, 0.5, Y(3.7), "INIT  INITIALIZATION", PAL["init"][1])

    nodes_I = [
        (X_LEFT, Y(4.2),  BW, BH, "Search Best Frame Pair\n(offsets 2–20, widen to 60)"),
        (X_LEFT, Y(5.0),  BW, BH, "Feature Matching\n(Lowe ratio=0.75, mutual=True)"),
        (X_LEFT, Y(5.8),  BW, BH, "Essential Matrix + RANSAC\n(p=0.999, reproj=1.0 px)"),
        (X_LEFT, Y(6.55), DW, DH, "≥50 inliers &\n≥120 triangulated?"),   # decision
        (X_LEFT, Y(7.4),  BW, BH, "Triangulation + Filter\n(depth ≤100m, reproj ≤3px, parallax ≥0.5°)"),
        (X_LEFT, Y(8.2),  BW, BH, "Seed Map  –  Insert 3D points\nSet T_wc0=I, T_wc1=R|t"),
    ]
    _draw_rounded_box(ax, *nodes_I[0][:4], *PAL["init"], nodes_I[0][4])
    _draw_rounded_box(ax, *nodes_I[1][:4], *PAL["init"], nodes_I[1][4])
    _draw_rounded_box(ax, *nodes_I[2][:4], *PAL["init"], nodes_I[2][4])
    _draw_diamond(ax, nodes_I[3][0], nodes_I[3][1], *nodes_I[3][2:4], *PAL["decision"], nodes_I[3][4])
    _draw_rounded_box(ax, *nodes_I[4][:4], *PAL["init"], nodes_I[4][4])
    _draw_rounded_box(ax, *nodes_I[5][:4], *PAL["init"], nodes_I[5][4])

    _arrow(ax, X_LEFT, Y(3.2)-BH/2, X_LEFT, Y(4.2)+BH/2)
    _arrow(ax, X_LEFT, Y(4.2)-BH/2, X_LEFT, Y(5.0)+BH/2)
    _arrow(ax, X_LEFT, Y(5.0)-BH/2, X_LEFT, Y(5.8)+BH/2)
    _arrow(ax, X_LEFT, Y(5.8)-BH/2, X_LEFT, Y(6.55)+DH/2)
    # No branch (expand offset): arrow left
    ax.annotate("", xy=(0.6, Y(4.2)), xytext=(X_LEFT-DW/2, Y(6.55)),
                arrowprops=dict(arrowstyle="-|>", color=PAL["fallback"][1], lw=1.2,
                                connectionstyle="arc3,rad=-0.3"), zorder=2)
    ax.text(0.45, Y(5.6), "No\n(widen\nto 60)", fontproperties=FONT_TINY,
            color=PAL["fallback"][1], ha="center", va="center")
    _arrow(ax, X_LEFT, Y(6.55)-DH/2, X_LEFT, Y(7.4)+BH/2, "Yes")
    _arrow(ax, X_LEFT, Y(7.4)-BH/2, X_LEFT, Y(8.2)+BH/2)

    # ─────────────────────────── SECTION B: TRACKING ────────────────────────
    _section_background(ax, 0.3, Y(12.8), 8.2, Y(8.6),
                        *PAL["track"][:2], alpha=0.20)
    _section_label(ax, 0.5, Y(8.7), "B  FEATURE MATCHING  (per frame)", PAL["track"][1])

    nodes_B = [
        (X_LEFT, Y(9.3),  BW, BH, "Query Local Map\n(window=80 frames, reproj=4px)"),
        (X_LEFT, Y(10.1), BW, BH, "Query Global Map\n(min_obs=2, reproj=2.5px)"),
        (X_LEFT, Y(10.9), BW, BH, "Lowe's Ratio Filter (0.75)\n→ Best-per-keypoint dedup"),
        (X_LEFT, Y(11.7), BW, BH, "Spatial Bucketing (4×4 grid, 10/cell)"),
        (X_LEFT, Y(12.5), BW, BH, "Coverage Check (4×4 grid)\nmax_cell_frac ≤ 0.65"),
    ]
    for (x, y, w, h, lbl) in nodes_B:
        _draw_rounded_box(ax, x, y, w, h, *PAL["track"], lbl)

    _arrow(ax, X_LEFT, Y(8.2)-BH/2, X_LEFT, Y(9.3)+BH/2)
    _arrow(ax, X_LEFT, Y(9.3)-BH/2, X_LEFT, Y(10.1)+BH/2)
    _arrow(ax, X_LEFT, Y(10.1)-BH/2, X_LEFT, Y(10.9)+BH/2)
    _arrow(ax, X_LEFT, Y(10.9)-BH/2, X_LEFT, Y(11.7)+BH/2)
    _arrow(ax, X_LEFT, Y(11.7)-BH/2, X_LEFT, Y(12.5)+BH/2)

    # ─────────────────────────── SECTION C: POSE EST ────────────────────────
    _section_background(ax, 0.3, Y(17.0), 8.2, Y(13.0),
                        *PAL["pose"][:2], alpha=0.20)
    _section_label(ax, 0.5, Y(13.1), "C  POSE ESTIMATION", PAL["pose"][1])

    nodes_C = [
        (X_LEFT, Y(13.7), BW, BH, "solvePnPRansac\n(ITERATIVE, p=0.999, reproj=4px local / 2.5px global)"),
        (X_LEFT, Y(14.5), BW, BH, "solvePnPRefineLM\n(Levenberg-Marquardt refinement)"),
        (X_LEFT, Y(15.3), BW, BH, "Post-LM Tight Filter\n(reproj ≤ 2.0 px, keep ≥6 inliers)"),
        (X_LEFT, Y(16.1), DW, DH, "inliers ≥ 20?"),   # decision
        (X_LEFT, Y(16.85),BW, BH, "Motion Validator\n(≤2m step, ≤25° rotation)"),
    ]
    _draw_rounded_box(ax, *nodes_C[0][:4], *PAL["pose"], nodes_C[0][4])
    _draw_rounded_box(ax, *nodes_C[1][:4], *PAL["pose"], nodes_C[1][4])
    _draw_rounded_box(ax, *nodes_C[2][:4], *PAL["pose"], nodes_C[2][4])
    _draw_diamond(ax, nodes_C[3][0], nodes_C[3][1], *nodes_C[3][2:4], *PAL["decision"], nodes_C[3][4])
    _draw_rounded_box(ax, *nodes_C[4][:4], *PAL["pose"], nodes_C[4][4])

    _arrow(ax, X_LEFT, Y(12.5)-BH/2, X_LEFT, Y(13.7)+BH/2)
    _arrow(ax, X_LEFT, Y(13.7)-BH/2, X_LEFT, Y(14.5)+BH/2)
    _arrow(ax, X_LEFT, Y(14.5)-BH/2, X_LEFT, Y(15.3)+BH/2)
    _arrow(ax, X_LEFT, Y(15.3)-BH/2, X_LEFT, Y(16.1)+DH/2)
    _arrow(ax, X_LEFT, Y(16.1)-DH/2, X_LEFT, Y(16.85)+BH/2, "Yes")

    # ─────────────────────────── SECTION D: FALLBACKS ───────────────────────
    _section_background(ax, 8.5, Y(21.5), 17.5, Y(8.6),
                        *PAL["fallback"][:2], alpha=0.14)
    _section_label(ax, 8.7, Y(8.7), "D  FALLBACK & RECOVERY PATHS", PAL["fallback"][1])

    # Essential fallback (middle column)
    nodes_E = [
        (X_MID, Y(9.5),  NW, BH, "Essential Matrix Fallback\n(scale = median(last 10 steps))"),
        (X_MID, Y(10.3), NW, BH, "Estimate E + Recover Pose\n(RANSAC p=0.999, thr=1px)"),
        (X_MID, Y(11.1), DW, DH, "inliers ≥ 40?"),
        (X_MID, Y(12.0), NW, BH, "Scale t by median step"),
        (X_MID, Y(12.8), NW, BH, "Motion Validation\n(≤2m, ≤25°)"),
        (X_MID, Y(13.6), DW, DH, "Consec. fallback\n< 10?"),
        (X_MID, Y(14.5), NW, BH, "Accept Essential Pose\n(reset CV counter)"),
    ]
    for i, (x, y, w, h, lbl) in enumerate(nodes_E):
        if i in (2, 5):
            _draw_diamond(ax, x, y, w, h, *PAL["decision"], lbl)
        else:
            _draw_rounded_box(ax, x, y, w, h, *PAL["fallback"], lbl)
    _arrow(ax, X_MID, Y(9.5)-BH/2,  X_MID, Y(10.3)+BH/2)
    _arrow(ax, X_MID, Y(10.3)-BH/2, X_MID, Y(11.1)+DH/2)
    _arrow(ax, X_MID, Y(11.1)-DH/2, X_MID, Y(12.0)+BH/2, "Yes")
    _arrow(ax, X_MID, Y(12.0)-BH/2, X_MID, Y(12.8)+BH/2)
    _arrow(ax, X_MID, Y(12.8)-BH/2, X_MID, Y(13.6)+DH/2)
    _arrow(ax, X_MID, Y(13.6)-DH/2, X_MID, Y(14.5)+BH/2, "Yes")

    # CV fallback (right column)
    nodes_CV = [
        (X_RIGHT, Y(11.5), NW, BH, "Constant-Velocity Fallback\n(dampened motion model)"),
        (X_RIGHT, Y(12.3), DW, DH, "CV budget\n< 8 consec.?"),
        (X_RIGHT, Y(13.2), NW, BH, "damp = 0.88 ^ n\nPropagate Δ T"),
        (X_RIGHT, Y(14.0), NW, BH, "Append CV Pose\n(counted as accepted)"),
    ]
    _draw_rounded_box(ax, *nodes_CV[0][:4], *PAL["fallback"], nodes_CV[0][4])
    _draw_diamond(ax, *nodes_CV[1][:4], *PAL["decision"], nodes_CV[1][4])
    _draw_rounded_box(ax, *nodes_CV[2][:4], *PAL["fallback"], nodes_CV[2][4])
    _draw_rounded_box(ax, *nodes_CV[3][:4], *PAL["fallback"], nodes_CV[3][4])
    _arrow(ax, X_RIGHT, Y(11.5)-BH/2, X_RIGHT, Y(12.3)+DH/2)
    _arrow(ax, X_RIGHT, Y(12.3)-DH/2, X_RIGHT, Y(13.2)+BH/2, "Yes")
    _arrow(ax, X_RIGHT, Y(13.2)-BH/2, X_RIGHT, Y(14.0)+BH/2)

    # Tracking-lost entry
    _draw_rounded_box(ax, X_RIGHT, Y(9.5), NW, BH, *PAL["fallback"],
                      "↳ TRACKING LOST\n(pending_rebuild = True)")

    # PnP "No" branch → Essential fallback
    _arrow(ax, X_LEFT+BW/2, Y(16.1), X_MID-NW/2, Y(9.5),
           "No (≥30 mts)", color=PAL["fallback"][1],
           connectionstyle="arc3,rad=-0.25")

    # PnP "No" branch → CV fallback (if essential also fails)
    _arrow(ax, X_LEFT+BW/2, Y(16.85),
           X_RIGHT-NW/2, Y(11.5),
           "CV needed", color=PAL["fallback"][1],
           connectionstyle="arc3,rad=0.2")

    # Fallback limit → tracking_lost
    _arrow(ax, X_MID+NW/2, Y(13.6),
           X_RIGHT-NW/2, Y(9.5),
           "No (limit)", color=PAL["fallback"][1],
           connectionstyle="arc3,rad=0.3")
    _arrow(ax, X_RIGHT+NW/2, Y(12.3),
           X_RIGHT-NW/2, Y(9.5)+BH/2 + 0.3,
           "No (budget\nexhausted)", color=PAL["fallback"][1],
           connectionstyle="arc3,rad=0.5")

    # ─────────────────────────── SECTION L: TRACKING LOST ───────────────────
    _section_background(ax, 8.5, Y(21.5), 17.5, Y(14.9),
                        *PAL["map"][:2], alpha=0.12)
    _section_label(ax, 8.7, Y(15.0), "L  TRACKING LOST  &  RELOCALIZATION", PAL["map"][1])

    nodes_L = [
        (X_MID,   Y(15.5), BW,  BH, "Tracking Lost Mode\n(frame skipped from trajectory)"),
        (X_MID,   Y(16.3), BW,  BH, "Local Map PnP\n(window=80, reproj=4px)"),
        (X_MID,   Y(17.1), BW,  BH, "Keyframe Similarity Score\n(descriptor match + 4×4 coverage ≥0.20)"),
        (X_MID,   Y(17.9), BW,  BH, "Top-5 Keyframe Global PnP\n(reproj=2.5px, spatial bucket, max_cell_frac=0.65)"),
        (X_MID,   Y(18.7), DW,  DH, "inliers ≥14 &\nratio ≥0.10?"),
        (X_MID,   Y(19.55),BW,  BH, "Motion Gate\n(≤2m/≤25° OR strong geometry)"),
        (X_MID,   Y(20.35),DW,  DH, "Accept relocal\npose?"),
        (X_MID,   Y(21.2), BW,  BH, "Recovery Probation\n(5 frames, min_inliers=18)"),
        (X_RIGHT, Y(17.5), NW,  BH, "Absolute Reset Path\n(inliers≥45, ratio≥0.45,\nmatches≥60, reproj<3px)"),
        (X_RIGHT, Y(18.5), NW,  BH, "Check Ref. Pose Distance\n(ref_trans ≤5m, ref_rot ≤120°)"),
        (X_LEFT,  Y(17.0), NW,  BH, "Essential-Matrix Fallback\n(scale=median, in tracking_lost)"),
        (X_LEFT,  Y(17.8), DW,  DH, "Fallback limit &\nbaseline ≥0.75m?"),
        (X_LEFT,  Y(18.7), NW,  BH, "Rebuild Map from Strong Anchor\n(matches ≥50, insert pts)"),
        (X_LEFT,  Y(19.6), DW,  DH, "Rebuild ≥12 pts\ninserted?"),
        (X_LEFT,  Y(20.5), NW,  BH, "Exit tracking_lost\n(consecutive_pnp_failures = 0)"),
    ]
    for i, (x, y, w, h, lbl) in enumerate(nodes_L):
        if i in (4, 6, 11, 13):
            _draw_diamond(ax, x, y, w, h, *PAL["decision"], lbl)
        elif i in (8, 9, 10, 12, 14):
            _draw_rounded_box(ax, x, y, w, h, *PAL["fallback"], lbl)
        else:
            _draw_rounded_box(ax, x, y, w, h, *PAL["map"], lbl)

    # Tracking lost → branches
    _arrow(ax, X_MID, Y(15.5)-BH/2, X_MID, Y(16.3)+BH/2)
    _arrow(ax, X_MID, Y(16.3)-BH/2, X_MID, Y(17.1)+BH/2, "local")
    _arrow(ax, X_MID, Y(17.1)-BH/2, X_MID, Y(17.9)+BH/2, "global top-5")
    _arrow(ax, X_MID, Y(17.9)-BH/2, X_MID, Y(18.7)+DH/2)
    _arrow(ax, X_MID, Y(18.7)-DH/2, X_MID, Y(19.55)+BH/2, "Yes")
    _arrow(ax, X_MID, Y(19.55)-BH/2, X_MID, Y(20.35)+DH/2)
    _arrow(ax, X_MID, Y(20.35)-DH/2, X_MID, Y(21.2)+BH/2, "Accept")

    # No branch on relocal → absolute reset
    _arrow(ax, X_MID+DW/2, Y(18.7),
           X_RIGHT-NW/2, Y(17.5),
           "No", color="#6a1b9a",
           connectionstyle="arc3,rad=-0.2")
    _arrow(ax, X_RIGHT, Y(17.5)-BH/2, X_RIGHT, Y(18.5)+BH/2)
    _arrow(ax, X_RIGHT+NW/2, Y(18.5),
           X_MID+DW/2, Y(20.35),
           "pass", color="#6a1b9a",
           connectionstyle="arc3,rad=-0.2")

    # Essential fallback in lost mode
    _arrow(ax, X_MID-BW/2, Y(16.3),
           X_LEFT+NW/2, Y(17.0),
           "reloc\nfailed", color=PAL["fallback"][1],
           connectionstyle="arc3,rad=0.3")
    _arrow(ax, X_LEFT, Y(17.0)-BH/2, X_LEFT, Y(17.8)+DH/2)
    _arrow(ax, X_LEFT, Y(17.8)-DH/2, X_LEFT, Y(18.7)+BH/2, "Yes")
    _arrow(ax, X_LEFT, Y(18.7)-BH/2, X_LEFT, Y(19.6)+DH/2)
    _arrow(ax, X_LEFT, Y(19.6)-DH/2, X_LEFT, Y(20.5)+BH/2, "Yes")
    # Exit tracking_lost → back to normal tracking loop
    _arrow(ax, X_LEFT, Y(20.5)-BH/2,
           X_LEFT-1.5, Y(20.5)-BH/2 - 0.4,
           "", color=PAL["map"][1], connectionstyle="arc3,rad=0.0")
    ax.annotate("", xy=(X_LEFT-1.5, Y(9.0)),
                xytext=(X_LEFT-1.5, Y(20.9)),
                arrowprops=dict(arrowstyle="-|>", color=PAL["map"][1], lw=1.2), zorder=2)
    _arrow(ax, X_LEFT-1.5, Y(9.0), X_LEFT-BW/2, Y(9.3),
           "re-enter\ntracking", color=PAL["map"][1],
           connectionstyle="arc3,rad=0.0")

    # Recovery probation → back to normal
    _arrow(ax, X_MID+BW/2, Y(21.2),
           X_LEFT+BW/2+0.1, Y(16.85),
           "probation\npassed", color=PAL["map"][1],
           connectionstyle="arc3,rad=-0.25")

    # Connect lost mode entry
    _arrow(ax, X_RIGHT, Y(9.5)-BH/2,
           X_MID+BW/2, Y(15.5),
           "enter\nlost mode", color=PAL["fallback"][1],
           connectionstyle="arc3,rad=0.15")

    # ─────────────────────────── SECTION E: MAP UPDATE ──────────────────────
    _section_background(ax, 0.3, Y(29.6), 8.2, Y(21.6),
                        *PAL["map"][:2], alpha=0.20)
    _section_label(ax, 0.5, Y(21.7), "E  MAP UPDATE  &  KEYFRAME MANAGEMENT", PAL["map"][1])

    nodes_M = [
        (X_LEFT, Y(22.3), BW, BH, "Keyframe Decision\n(trans ≥0.25m OR rot ≥5° OR inliers<40)"),
        (X_LEFT, Y(23.1), BW, BH, "Keyframe Triangulation\n(last KF → curr, Lowe ratio 0.75)"),
        (X_LEFT, Y(23.9), BW, BH, "Filter Points\n(parallax ≥0.35°, depth ≤60m, reproj ≤2px)"),
        (X_LEFT, Y(24.7), BW, BH, "Uniqueness Check\n(radius = 0.05 m)"),
        (X_LEFT, Y(25.5), DW, DH, "≥6 new pts\nOR weak tracking?"),
        (X_LEFT, Y(26.35),BW, BH, "Insert Map Points + Descriptors\nAdd Keyframe Record"),
        (X_LEFT, Y(27.15),BW, BH, "Track Map Points\n(increment seen / missed counts)"),
        (X_LEFT, Y(27.95),BW, BH, "Map Pruning\n(min_obs=2, max_miss=60/80)"),
        (X_LEFT, Y(28.75),DW, DH, "map_size < 100?"),
        (X_LEFT, Y(29.45),BW, BH, "Emergency Rebuild\n(pending_rebuild = True)"),
    ]
    for i, (x, y, w, h, lbl) in enumerate(nodes_M):
        if i in (4, 8):
            _draw_diamond(ax, x, y, w, h, *PAL["decision"], lbl)
        else:
            _draw_rounded_box(ax, x, y, w, h, *PAL["map"], lbl)

    _arrow(ax, X_LEFT, Y(16.85)-BH/2, X_LEFT, Y(22.3)+BH/2)
    _arrow(ax, X_LEFT, Y(22.3)-BH/2, X_LEFT, Y(23.1)+BH/2)
    _arrow(ax, X_LEFT, Y(23.1)-BH/2, X_LEFT, Y(23.9)+BH/2)
    _arrow(ax, X_LEFT, Y(23.9)-BH/2, X_LEFT, Y(24.7)+BH/2)
    _arrow(ax, X_LEFT, Y(24.7)-BH/2, X_LEFT, Y(25.5)+DH/2)
    _arrow(ax, X_LEFT, Y(25.5)-DH/2, X_LEFT, Y(26.35)+BH/2, "Yes")
    _arrow(ax, X_LEFT, Y(26.35)-BH/2, X_LEFT, Y(27.15)+BH/2)
    _arrow(ax, X_LEFT, Y(27.15)-BH/2, X_LEFT, Y(27.95)+BH/2)
    _arrow(ax, X_LEFT, Y(27.95)-BH/2, X_LEFT, Y(28.75)+DH/2)
    _arrow(ax, X_LEFT, Y(28.75)-DH/2, X_LEFT, Y(29.45)+BH/2, "Yes")

    # Strong anchor update (side note)
    _draw_rounded_box(ax, X_MID, Y(24.0), NW, BH, *PAL["map"],
                      "Update Strong Anchor\n(inliers≥30, baseline≥0.75m, gap≥10f)")
    _draw_rounded_box(ax, X_MID, Y(24.9), NW, BH, *PAL["map"],
                      "Rebuild from Anchor\n(matches≥50 → triangulate → insert)")
    _arrow(ax, X_LEFT+BW/2, Y(22.3), X_MID-NW/2, Y(24.0),
           "if local/global\ntrack success",
           color=PAL["map"][1], connectionstyle="arc3,rad=-0.2")
    _arrow(ax, X_MID, Y(24.0)-BH/2, X_MID, Y(24.9)+BH/2)
    _arrow(ax, X_MID+NW/2, Y(24.9),
           X_LEFT+BW/2, Y(28.75),
           "pending\nrebuild", color=PAL["map"][1],
           connectionstyle="arc3,rad=-0.2")

    # ─────────────────────────── SECTION F: OUTPUT ──────────────────────────
    _section_background(ax, 0.3, Y(33.8), 17.5, Y(29.8),
                        *PAL["output"][:2], alpha=0.20)
    _section_label(ax, 0.5, Y(29.9), "F  OUTPUT  &  TRAJECTORY", PAL["output"][1])

    OUT_Y = [30.4, 31.2, 32.0, 32.8, 33.5]
    out_nodes = [
        "Append Pose T_wc + Timestamp (ns)\n(up-to-scale world frame)",
        "Update CV Velocity Model\nΔT for next const-vel prediction",
        "Frame Log CSV\n(frame_idx, timestamp_s, source, pnp_inliers, reproj_rmse_px, …)",
        "VOMapVisualizer\n(trajectory + map pts + GT overlay)",
        "Save TUM Trajectory TXT\n+ Run Summary JSON  →  ATE/RPE Evaluation",
    ]
    for oy, lbl in zip(OUT_Y, out_nodes):
        _draw_rounded_box(ax, X_MID, Y(oy), 8.0, BH, *PAL["output"], lbl, fontprop=FONT_SMALL)

    _arrow(ax, X_LEFT, Y(29.45)-BH/2, X_MID, Y(30.4)+BH/2)
    for oy1, oy2 in zip(OUT_Y[:-1], OUT_Y[1:]):
        _arrow(ax, X_MID, Y(oy1)-BH/2, X_MID, Y(oy2)+BH/2)

    # Loop-back arrow: output → next frame
    ax.annotate("", xy=(17.2, Y(0.9)), xytext=(17.2, Y(33.5)-BH/2),
                arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.3), zorder=2)
    _arrow(ax, 17.2, Y(0.9), X_LEFT+BW/2, Y(0.9), "next frame",
           color="#555555")

    # Scale ambiguity note
    ax.text(8.7, Y(33.3),
            "⚠  MONOCULAR: Scale Ambiguity\nt recovered up-to-scale (||t||=1)\n"
            "Metric steps estimated from\nrecent accepted translation norms",
            fontproperties=FONT_SMALL, color="#b71c1c",
            ha="left", va="center", zorder=6,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff9c4",
                      edgecolor="#b71c1c", alpha=0.95, linewidth=1.5))

    # ── Title ──
    ax.text(FIG_W/2, FIG_H - 0.15,
            "Monocular Visual Odometry Pipeline  (ORB-PnP-Map)",
            ha="center", va="top",
            fontproperties=FontProperties(family="DejaVu Sans", weight="bold", size=14),
            color="#1a1a2e", zorder=7)

    plt.tight_layout(pad=0.1)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  STEREO FLOWCHART
# ══════════════════════════════════════════════════════════════════════════════

def build_stereo_flowchart():
    FIG_W, FIG_H = 18, 26
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    def Y(y_log):
        return FIG_H - y_log

    BW = 3.6
    BH = 0.52
    NW = 3.0
    DW = 3.0
    DH = 0.62
    X_LEFT  = 3.8
    X_MID   = 9.0
    X_RIGHT = 14.2

    # ─────────────────────────── SECTION A: INPUT ───────────────────────────
    _section_background(ax, 0.3, Y(3.9), 17.5, Y(0.3),
                        *PAL["input"][:2], alpha=0.22)
    _section_label(ax, 0.5, Y(0.4), "A  INPUT  &  CALIBRATION", PAL["input"][1])

    inp_nodes = [
        (3.5,  Y(1.0), 3.0, BH, "TUM VI Dataset Loader\n(room2_512_16, stereo)"),
        (7.0,  Y(1.0), 3.0, BH, "Left Image\n(cam0)"),
        (11.0, Y(1.0), 3.0, BH, "Right Image\n(cam1)"),
        (15.0, Y(1.0), 3.0, BH, "camchain.yaml\nCalibration"),
        (7.0,  Y(2.0), 3.0, BH, "Fisheye Rectification\n(cv2 fisheye SVD)"),
        (11.0, Y(2.0), 3.0, BH, "Compute Baseline\nb = 0.101 m (rectified)"),
        (9.0,  Y(3.1), 3.8, BH, "Rectified Left + Right Pair\n→ Stereo Frame Processor"),
    ]
    for (x, y, w, h, lbl) in inp_nodes:
        _draw_rounded_box(ax, x, y, w, h, *PAL["input"], lbl, fontprop=FONT_SMALL)
    _arrow(ax, 3.5, Y(1.0)-BH/2, 7.0, Y(2.0)+BH/2)
    _arrow(ax, 7.0, Y(1.0)-BH/2, 7.0, Y(2.0)+BH/2)
    _arrow(ax, 11.0, Y(1.0)-BH/2, 11.0, Y(2.0)+BH/2)
    _arrow(ax, 15.0, Y(1.0)-BH/2, 11.0, Y(2.0)+BH/2)
    _arrow(ax, 7.0, Y(2.0)-BH/2, 9.0, Y(3.1)+BH/2)
    _arrow(ax, 11.0, Y(2.0)-BH/2, 9.0, Y(3.1)+BH/2)

    # ─────────────────────────── SECTION B: STEREO PROCESSING ───────────────
    _section_background(ax, 0.3, Y(8.8), 17.5, Y(4.0),
                        *PAL["stereo"][:2], alpha=0.20)
    _section_label(ax, 0.5, Y(4.1), "B  STEREO DEPTH PROCESSING  (METRIC SCALE)", PAL["stereo"][1])

    stereo_nodes = [
        (X_LEFT,  Y(4.7), BW, BH, "SGBM Disparity\n(num_disp=64, block=5, 3WAY mode)"),
        (X_RIGHT, Y(4.7), BW, BH, "ORB Feature Detection\n(2000 feats, FAST=20, 8 levels)"),
        (X_LEFT,  Y(5.5), BW, BH, "Depth Reconstruction\nZ = f · b / d  (metric!)"),
        (X_RIGHT, Y(5.5), BW, BH, "5×5 Patch Disparity Sampling\n(std < 1.5, median)"),
        (X_MID,   Y(6.3), BW, BH, "Depth Validity Filter\n(0.4 m ≤ Z ≤ 12.0 m)"),
        (X_MID,   Y(7.1), BW, BH, "3D Point per Feature  X,Y,Z in Left Camera\nX=(u-cx)·Z/fx, Y=(v-cy)·Z/fy"),
        (X_MID,   Y(7.9), BW, BH, "Spatial Bucketing  6×8 Grid\n(max 40 features/cell, keep by response)"),
        (X_MID,   Y(8.6), DW, DH, "Stereo 3D Feature Set\n(keypts, descs, metric 3D pts)"),
    ]
    for i, (x, y, w, h, lbl) in enumerate(stereo_nodes):
        if i == 7:
            # "pseudo-output" node – use filled rounded box with border
            _draw_rounded_box(ax, x, y, w, h, "#bbdefb", PAL["stereo"][1], lbl,
                              linewidth=2.0, fontprop=FONT_SMALL)
        else:
            _draw_rounded_box(ax, x, y, w, h, *PAL["stereo"], lbl, fontprop=FONT_SMALL)

    _arrow(ax, 9.0, Y(3.1)-BH/2, X_LEFT, Y(4.7)+BH/2)
    _arrow(ax, 9.0, Y(3.1)-BH/2, X_RIGHT, Y(4.7)+BH/2)
    _arrow(ax, X_LEFT,  Y(4.7)-BH/2, X_LEFT,  Y(5.5)+BH/2)
    _arrow(ax, X_RIGHT, Y(4.7)-BH/2, X_RIGHT, Y(5.5)+BH/2)
    _arrow(ax, X_LEFT,  Y(5.5)-BH/2, X_MID, Y(6.3)+BH/2)
    _arrow(ax, X_RIGHT, Y(5.5)-BH/2, X_MID, Y(6.3)+BH/2)
    _arrow(ax, X_MID, Y(6.3)-BH/2, X_MID, Y(7.1)+BH/2)
    _arrow(ax, X_MID, Y(7.1)-BH/2, X_MID, Y(7.9)+BH/2)
    _arrow(ax, X_MID, Y(7.9)-BH/2, X_MID, Y(8.6)+DH/2)

    # Metric scale badge
    ax.text(15.8, Y(6.5),
            "📐 METRIC SCALE\nfrom stereo baseline\n(b = 0.101 m)",
            fontproperties=FONT_SMALL, color=PAL["stereo"][1],
            ha="center", va="center", zorder=6,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#e3f2fd",
                      edgecolor=PAL["stereo"][1], alpha=0.9, linewidth=2.0))

    # ─────────────────────────── SECTION C: POSE EST ────────────────────────
    _section_background(ax, 0.3, Y(14.2), 8.2, Y(8.9),
                        *PAL["pose"][:2], alpha=0.20)
    _section_label(ax, 0.5, Y(9.0), "C  POSE ESTIMATION", PAL["pose"][1])

    pose_nodes = [
        (X_LEFT, Y(9.6),  BW, BH, "ORB Matching  (prev → curr)\nLowe ratio 0.70, mutual cross-check"),
        (X_LEFT, Y(10.4), BW, BH, "Cell Bucketing (4×4, max 12/cell)\nDepth Consistency Filter (0 < Z ≤ 12m)"),
        (X_LEFT, Y(11.2), BW, BH, "solvePnPRansac\n(ITERATIVE, p=0.999, reproj=2px, min=25)"),
        (X_LEFT, Y(12.0), BW, BH, "solvePnPRefineLM\n(Levenberg–Marquardt)"),
        (X_LEFT, Y(12.8), DW, DH, "inliers ≥ 25?"),
        (X_LEFT, Y(13.65),BW, BH, "T_prev_curr → T_w_curr\n= T_w_prev  @  inv(T_prev_curr)"),
    ]
    for i, (x, y, w, h, lbl) in enumerate(pose_nodes):
        if i == 4:
            _draw_diamond(ax, x, y, w, h, *PAL["decision"], lbl)
        else:
            _draw_rounded_box(ax, x, y, w, h, *PAL["pose"], lbl, fontprop=FONT_SMALL)

    _arrow(ax, X_MID, Y(8.6)-DH/2, X_LEFT, Y(9.6)+BH/2)
    _arrow(ax, X_LEFT, Y(9.6)-BH/2,  X_LEFT, Y(10.4)+BH/2)
    _arrow(ax, X_LEFT, Y(10.4)-BH/2, X_LEFT, Y(11.2)+BH/2)
    _arrow(ax, X_LEFT, Y(11.2)-BH/2, X_LEFT, Y(12.0)+BH/2)
    _arrow(ax, X_LEFT, Y(12.0)-BH/2, X_LEFT, Y(12.8)+DH/2)
    _arrow(ax, X_LEFT, Y(12.8)-DH/2, X_LEFT, Y(13.65)+BH/2, "Yes")

    # ─────────────────────────── SECTION D: RECOVERY ────────────────────────
    _section_background(ax, 8.5, Y(21.0), 17.5, Y(8.9),
                        *PAL["recovery"][:2], alpha=0.16)
    _section_label(ax, 8.7, Y(9.0), "D  RECOVERY  &  FALLBACK PATHS", PAL["recovery"][1])

    rec_nodes = [
        (X_MID,   Y(9.8),  NW, BH, "Dense Feature Retry\n(lower FAST thr, ORB fallback)"),
        (X_MID,   Y(10.6), DW, DH, "Dense retry\nsucceeds?"),
        (X_MID,   Y(11.5), NW, BH, "Loose Retry\n(reproj=3.5px, no mutual, min=8)"),
        (X_MID,   Y(12.3), DW, DH, "Loose retry\nsucceeds?"),
        (X_RIGHT, Y(10.6), NW, BH, "3D–3D RANSAC Rigid\n(SVD, thr=0.15m, ≥15 pts, 200 iters)"),
        (X_RIGHT, Y(11.5), DW, DH, "3D-3D\nsucceeds?"),
        (X_MID,   Y(13.2), NW, BH, "Constant-Velocity Fallback\n(damp = 0.88 ^ n)"),
        (X_MID,   Y(14.0), DW, DH, "CV budget\n< 8 consec.?"),
        (X_MID,   Y(14.9), NW, BH, "Propagate Δ T (dampened)\nAppend CV pose"),
        (X_RIGHT, Y(13.5), NW, BH, "Frame Skipped\n(tracking_failure++)"),
    ]
    for i, (x, y, w, h, lbl) in enumerate(rec_nodes):
        if i in (1, 3, 5, 7):
            _draw_diamond(ax, x, y, w, h, *PAL["decision"], lbl)
        elif i == 9:
            _draw_rounded_box(ax, x, y, w, h, *PAL["fallback"], lbl)
        else:
            _draw_rounded_box(ax, x, y, w, h, *PAL["recovery"], lbl, fontprop=FONT_SMALL)

    # "No" from PnP → dense retry
    _arrow(ax, X_LEFT+BW/2, Y(12.8),
           X_MID-NW/2, Y(9.8), "No", color=PAL["fallback"][1],
           connectionstyle="arc3,rad=-0.3")
    _arrow(ax, X_MID, Y(9.8)-BH/2, X_MID, Y(10.6)+DH/2)
    _arrow(ax, X_MID, Y(10.6)-DH/2, X_MID, Y(11.5)+BH/2, "No")
    _arrow(ax, X_MID, Y(11.5)-BH/2, X_MID, Y(12.3)+DH/2)
    _arrow(ax, X_MID, Y(12.3)-DH/2, X_MID, Y(13.2)+BH/2, "No→CV")
    _arrow(ax, X_MID, Y(13.2)-BH/2, X_MID, Y(14.0)+DH/2)
    _arrow(ax, X_MID, Y(14.0)-DH/2, X_MID, Y(14.9)+BH/2, "Yes")
    # Dense retry → 3D-3D
    _arrow(ax, X_MID+DW/2, Y(10.6), X_RIGHT-NW/2, Y(10.6),
           "prev 3D avail.", color=PAL["recovery"][1])
    _arrow(ax, X_RIGHT, Y(10.6)-BH/2, X_RIGHT, Y(11.5)+DH/2)
    # 3D-3D success → accepted
    _arrow(ax, X_RIGHT, Y(11.5)-DH/2,
           X_LEFT+BW/2, Y(13.65),
           "Yes → metric\nrigid pose",
           color=PAL["recovery"][1], connectionstyle="arc3,rad=0.3")
    # Dense success → accepted
    _arrow(ax, X_MID+DW/2, Y(10.6)+0.1,
           X_LEFT+BW/2, Y(13.65),
           "Yes\n(use dense)", color=PAL["recovery"][1],
           connectionstyle="arc3,rad=-0.4")
    # Loose success → accepted
    _arrow(ax, X_MID+DW/2, Y(12.3),
           X_LEFT+BW/2, Y(13.65),
           "Yes\n(loose)", color=PAL["recovery"][1],
           connectionstyle="arc3,rad=-0.3")
    # CV failure or budget exhausted
    _arrow(ax, X_MID+DW/2, Y(14.0),
           X_RIGHT-NW/2, Y(13.5), "No\n(exhausted)",
           color=PAL["fallback"][1], connectionstyle="arc3,rad=-0.2")

    # ─────────────────────────── SECTION F: OUTPUT ──────────────────────────
    _section_background(ax, 0.3, Y(25.6), 17.5, Y(21.2),
                        *PAL["output"][:2], alpha=0.20)
    _section_label(ax, 0.5, Y(21.3), "F  OUTPUT  &  TRAJECTORY", PAL["output"][1])

    out_y_vals = [21.8, 22.6, 23.4, 24.2, 25.0, 25.4]
    out_lbls = [
        "Append Metric Pose T_w_c + Timestamp (ns)\n(absolute metric scale  —  no ambiguity)",
        "Update CV Velocity Model  ΔT_prev_curr",
        "VOVisualizer\n(left rect image + traj XZ scatter)",
        "Frame Log CSV\n(frame_idx, timestamp_s, status, inlier_count, reproj_rmse_px)",
        "Run Summary JSON\n(frames_attempted, frames_succeeded, cv_fallback_count, …)",
        "Save TUM Trajectory TXT  →  ATE / RPE Evaluation",
    ]
    for oy, lbl in zip(out_y_vals, out_lbls):
        _draw_rounded_box(ax, X_MID, Y(oy), 9.5, BH, *PAL["output"], lbl, fontprop=FONT_SMALL)

    _arrow(ax, X_LEFT, Y(13.65)-BH/2, X_MID, Y(21.8)+BH/2)
    _arrow(ax, X_MID, Y(14.9)-BH/2, X_MID, Y(21.8)+BH/2,
           color="#bf360c", connectionstyle="arc3,rad=0.15")
    for oy1, oy2 in zip(out_y_vals[:-1], out_y_vals[1:]):
        _arrow(ax, X_MID, Y(oy1)-BH/2, X_MID, Y(oy2)+BH/2)

    # Loop-back
    ax.annotate("", xy=(17.5, Y(0.9)), xytext=(17.5, Y(25.4)-BH/2),
                arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.3), zorder=2)
    ax.annotate("", xy=(15.0-1.5, Y(1.0)+BH/2),
                xytext=(17.5, Y(0.9)),
                arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.3), zorder=2)
    ax.text(17.65, Y(13.0), "next\nframe",
            fontproperties=FONT_TINY, color="#555555", ha="left", va="center")

    # Metric scale note
    ax.text(1.0, Y(20.4),
            "✓  STEREO: Metric Scale\n"
            "Z = f · b / d  (d = disparity, b = 0.101 m)\n"
            "All poses in absolute metric coordinates\n"
            "No Sim3 scale correction needed at evaluation",
            fontproperties=FONT_SMALL, color=PAL["stereo"][1],
            ha="left", va="center", zorder=6,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#e3f2fd",
                      edgecolor=PAL["stereo"][1], alpha=0.95, linewidth=1.8))

    # Title
    ax.text(FIG_W/2, FIG_H - 0.15,
            "Stereo Visual Odometry Pipeline  (SGBM-ORB-PnP  |  Metric Scale)",
            ha="center", va="top",
            fontproperties=FontProperties(family="DejaVu Sans", weight="bold", size=14),
            color="#1a1a2e", zorder=7)

    plt.tight_layout(pad=0.1)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  COMBINED COMPARISON FIGURE
# ══════════════════════════════════════════════════════════════════════════════

def build_combined_comparison():
    FIG_W, FIG_H = 22, 14
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor("white")

    for ax in axes:
        ax.set_xlim(0, 11)
        ax.set_ylim(0, 14)
        ax.axis("off")
        ax.set_facecolor("white")

    def Y(y_log, h=14):
        return h - y_log

    BW, BH = 7.5, 0.58
    BN = 5.5    # narrow
    DW, DH = 5.5, 0.68
    CX = 5.5    # centre x

    # ─── LEFT: MONOCULAR ───
    ax = axes[0]
    ax.text(CX, Y(0.35), "Monocular VO\n(ORB + PnP-Map)",
            ha="center", va="top",
            fontproperties=FontProperties(family="DejaVu Sans", weight="bold", size=13),
            color="#2e7d32")

    mono_rows = [
        (1.1,  PAL["input"],    "Single Camera (cam0)"),
        (1.85, PAL["input"],    "Fisheye Undistortion  (balance=0.0)"),
        (2.6,  PAL["init"],     "Initialization: E-matrix + Triangulation\n(offsets 2–20, score = n_tri + 5·parallax)"),
        (3.45, PAL["track"],    "ORB 2000  →  Lowe 0.75  →  Spatial Bucket"),
        (4.2,  PAL["track"],    "Local Map PnP  (window=80 frames)"),
        (4.95, PAL["pose"],     "solvePnPRansac  +  solvePnPRefineLM"),
        (5.7,  PAL["pose"],     "Post-LM 2px Tight Filter  →  Motion Validation"),
        (6.55, PAL["fallback"], "Essential Fallback\n(E-mat, scale = median(last 10 steps))"),
        (7.4,  PAL["fallback"], "Const-Velocity Fallback  (damp = 0.88ⁿ, n ≤ 8)"),
        (8.2,  PAL["map"],      "Relocalization  (local + keyframe global PnP)"),
        (8.95, PAL["map"],      "Recovery Probation  (5 frames, inliers ≥ 18)"),
        (9.7,  PAL["map"],      "Keyframe Triangulation  +  Map Pruning"),
        (10.5, PAL["map"],      "Rebuild from Strong Anchor  (baseline ≥ 0.75 m)"),
        (11.3, PAL["output"],   "TUM Trajectory  →  Sim3 ATE/RPE Evaluation"),
    ]
    for (yt, pal, lbl) in mono_rows:
        _draw_rounded_box(ax, CX, Y(yt), BN, BH, *pal, lbl, fontprop=FONT_SMALL)

    for i in range(len(mono_rows) - 1):
        y1 = Y(mono_rows[i][0]) - BH/2
        y2 = Y(mono_rows[i+1][0]) + BH/2
        _arrow(ax, CX, y1, CX, y2)

    # Scale ambiguity banner
    ax.text(CX, Y(12.3),
            "⚠  SCALE AMBIGUITY\n||t|| = 1  (up-to-scale)\nSim3 alignment needed at evaluation",
            ha="center", va="center",
            fontproperties=FONT_SMALL, color="#b71c1c",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff9c4",
                      edgecolor="#b71c1c", alpha=0.95, linewidth=2.0))

    ax.text(CX, Y(13.55), "Coverage: 91.9%  |  ATE ≈ 1.26 m",
            ha="center", va="center",
            fontproperties=FONT_SMALL, color="#555555",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#f9f9f9",
                      edgecolor="#aaa", alpha=0.9))

    # ─── RIGHT: STEREO ───
    ax = axes[1]
    ax.text(CX, Y(0.35), "Stereo VO\n(SGBM-ORB-PnP  |  Metric Scale)",
            ha="center", va="top",
            fontproperties=FontProperties(family="DejaVu Sans", weight="bold", size=13),
            color="#1565c0")

    stereo_rows = [
        (1.1,  PAL["input"],    "Stereo Pair (cam0 + cam1)\nBaseline b = 0.101 m"),
        (1.95, PAL["input"],    "Fisheye Rectification  (cv2 fisheye SVD)"),
        (2.7,  PAL["stereo"],   "SGBM Disparity\n(64 disp, block=5, 3WAY)"),
        (3.55, PAL["stereo"],   "Depth  Z = f·b/d   (metric!)\n5×5 Patch Sampling  |  std < 1.5"),
        (4.4,  PAL["stereo"],   "Depth Filter  0.4 m ≤ Z ≤ 12.0 m\n6×8 Grid Bucket  (max 40/cell)"),
        (5.25, PAL["track"],    "ORB 2000  →  Lowe 0.70  →  Mutual Check  →  4×4 Cell Bucket"),
        (6.1,  PAL["pose"],     "solvePnPRansac  (2px, min=25)  +  solvePnPRefineLM"),
        (6.95, PAL["recovery"], "Dense ORB Retry  (low FAST threshold)"),
        (7.7,  PAL["recovery"], "Loose Retry  (3.5px, no mutual, min=8)"),
        (8.45, PAL["recovery"], "3D–3D RANSAC Rigid Transform\n(SVD, thr=0.15m, ≥15 pts)"),
        (9.3,  PAL["fallback"], "Const-Velocity Fallback  (0.88ⁿ, n ≤ 8)"),
        (10.1, PAL["output"],   "Metric Pose  T_w_c  (no scale correction needed)"),
        (10.85,PAL["output"],   "TUM Trajectory  →  SE3 ATE/RPE Evaluation"),
    ]
    for (yt, pal, lbl) in stereo_rows:
        _draw_rounded_box(ax, CX, Y(yt), BN, BH, *pal, lbl, fontprop=FONT_SMALL)

    for i in range(len(stereo_rows) - 1):
        y1 = Y(stereo_rows[i][0]) - BH/2
        y2 = Y(stereo_rows[i+1][0]) + BH/2
        _arrow(ax, CX, y1, CX, y2)

    # Metric scale banner
    ax.text(CX, Y(12.3),
            "✓  METRIC SCALE\nZ from stereo disparity (b=0.101m)\nNo Sim3 alignment needed",
            ha="center", va="center",
            fontproperties=FONT_SMALL, color="#1565c0",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#e3f2fd",
                      edgecolor="#1565c0", alpha=0.95, linewidth=2.0))

    ax.text(CX, Y(13.55), "Coverage: 100%  |  ATE metric (SE3 alignment)",
            ha="center", va="center",
            fontproperties=FONT_SMALL, color="#555555",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#f9f9f9",
                      edgecolor="#aaa", alpha=0.9))

    # ── Centre evolution arrow ──
    fig.text(0.5, 0.5,
             "→\nMONO\n→\nSTEREO\n(metric\nscale)",
             ha="center", va="center",
             fontsize=10, color="#555555", weight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5",
                       edgecolor="#999", alpha=0.8))

    # Main title
    fig.text(0.5, 0.99,
             "From Monocular VO (Scale Ambiguity)  →  Stereo VO (Metric Scale)  on TUM VI",
             ha="center", va="top",
             fontsize=14, fontweight="bold", color="#1a1a2e")

    plt.tight_layout(pad=0.8)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  SAVE HELPER
# ══════════════════════════════════════════════════════════════════════════════

def save_figure(fig, stem: str):
    """Save fig as PNG (300 DPI), SVG, and PDF."""
    paths = []
    for ext, kwargs in [
        ("png", {"dpi": 300, "bbox_inches": "tight", "facecolor": "white"}),
        ("svg", {"bbox_inches": "tight", "facecolor": "white"}),
        ("pdf", {"bbox_inches": "tight", "facecolor": "white"}),
    ]:
        p = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(str(p), format=ext, **kwargs)
        paths.append(p)
        print(f"  Saved: {p}")
    plt.close(fig)
    return paths


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Generating VO pipeline flowcharts …")
    print(f"Output directory: {OUT_DIR}")
    print("=" * 60)

    print("\n[1/3] Monocular VO pipeline …")
    fig_mono = build_monocular_flowchart()
    save_figure(fig_mono, "monocular_vo_pipeline")

    print("\n[2/3] Stereo VO pipeline …")
    fig_stereo = build_stereo_flowchart()
    save_figure(fig_stereo, "stereo_vo_pipeline")

    print("\n[3/3] Combined comparison …")
    fig_comb = build_combined_comparison()
    save_figure(fig_comb, "combined_pipeline_comparison")

    print("\n" + "=" * 60)
    print("All flowcharts generated successfully.")
    print(f"Files in: {OUT_DIR}")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"  {f.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
