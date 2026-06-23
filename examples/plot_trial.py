"""
Plot trial data from corridor_tracker.py — layout similar to Peng et al. Fig 5.

Usage:
    python examples/plot_trial.py data/reto_bengala_*.csv data/reto_colete_*.csv
    python examples/plot_trial.py data/reto_*.csv --corridor reto --out results/reto.png

Each CSV is one condition. Labels come from filename or --labels argument.
"""
import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import yaml

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("csvs", nargs="+", help="CSV files, one per condition")
parser.add_argument("--corridor", default=None,
                    help="Corridor key in corridors.yaml (auto-detected from filename if omitted)")
parser.add_argument("--config",  default="config/corridors.yaml")
parser.add_argument("--labels",  nargs="*", default=None,
                    help="Condition labels (default: inferred from filename)")
parser.add_argument("--out", default=None,
                    help="Save figure to file (e.g. results/trial.png)")
args = parser.parse_args()

# ── Load config ───────────────────────────────────────────────────────────────
with open(args.config) as f:
    cfg = yaml.safe_load(f)

corridor_key = args.corridor
if corridor_key is None:
    # Try to infer from first filename
    for key in cfg["corridors"]:
        if key in Path(args.csvs[0]).stem:
            corridor_key = key
            break
if corridor_key is None:
    raise ValueError("Could not infer corridor from filename. Use --corridor.")

corridor = cfg["corridors"][corridor_key]
print(f"Corridor: {corridor_key}")

# ── Load CSVs ─────────────────────────────────────────────────────────────────
COLORS = ["#e05c5c", "#5c8de0", "#5cb85c", "#e0a83a"]

def load_csv(path):
    import csv
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    return rows

trials = []
for i, csv_path in enumerate(args.csvs):
    data = load_csv(csv_path)
    label = (args.labels[i] if args.labels and i < len(args.labels)
             else Path(csv_path).stem.split("_")[1] if "_" in Path(csv_path).stem
             else f"Cond {i+1}")
    trials.append({"label": label, "data": data, "color": COLORS[i % len(COLORS)]})
    print(f"  {label}: {len(data)} frames, "
          f"duration={data[-1]['t']:.1f}s, "
          f"avg_speed={np.mean([r['speed_mps'] for r in data]):.2f} m/s")

# ── Summary stats ─────────────────────────────────────────────────────────────
def total_distance(data):
    d = 0.0
    for i in range(1, len(data)):
        dx = data[i]["x"] - data[i-1]["x"]
        dy = data[i]["y"] - data[i-1]["y"]
        d += math.sqrt(dx**2 + dy**2)
    return d

stats = []
for t in trials:
    d = t["data"]
    stats.append({
        "label":     t["label"],
        "time_s":    d[-1]["t"],
        "dist_m":    total_distance(d),
        "avg_speed": np.mean([r["speed_mps"] for r in d]),
    })

# ── Figure layout: [trajectory | yaw | speed | stats] ────────────────────────
fig = plt.figure(figsize=(16, 6))
fig.patch.set_facecolor("#1a1a2e")

ax_traj  = fig.add_axes([0.02, 0.08, 0.28, 0.84])   # top-view corridor
ax_yaw   = fig.add_axes([0.34, 0.55, 0.30, 0.37])   # head orientation
ax_speed = fig.add_axes([0.34, 0.08, 0.30, 0.37])   # walking speed
ax_time  = fig.add_axes([0.68, 0.60, 0.09, 0.30])   # stat: time
ax_dist  = fig.add_axes([0.79, 0.60, 0.09, 0.30])   # stat: distance
ax_vel   = fig.add_axes([0.90, 0.60, 0.09, 0.30])   # stat: avg speed

for ax in [ax_traj, ax_yaw, ax_speed, ax_time, ax_dist, ax_vel]:
    ax.set_facecolor("#16213e")
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")

# ── Trajectory ────────────────────────────────────────────────────────────────
w = corridor.get("width_m", corridor.get("corridor_width_m", 1.2))

for wall in corridor.get("walls", []):
    if len(wall) == 2:
        xs = [wall[0][0], wall[1][0]]
        ys = [wall[0][1], wall[1][1]]
        ax_traj.plot(xs, ys, color="white", linewidth=3)

for t in trials:
    xs = [r["x"] for r in t["data"]]
    ys = [r["y"] for r in t["data"]]
    ax_traj.plot(xs, ys, color=t["color"], linewidth=1.5, alpha=0.85)

sx, sy = corridor["start"]
gx, gy = corridor["goal"]
ax_traj.plot(sx, sy, "o", color="#00ff88", markersize=10, zorder=5)
ax_traj.plot(gx, gy, "x", color="#ff4444", markersize=12, markeredgewidth=2.5, zorder=5)

for m in corridor["markers"]:
    ax_traj.plot(m["x"], m["y"], "s", color="#00cfff", markersize=6)
    ax_traj.text(m["x"]+0.05, m["y"]+0.05, str(m["id"]),
                 color="#00cfff", fontsize=7)

ax_traj.set_aspect("equal")
ax_traj.set_title(corridor["description"], color="white", fontsize=9, pad=4)
ax_traj.set_xlabel("x (m)", color="white", fontsize=8)
ax_traj.set_ylabel("y (m)", color="white", fontsize=8)

legend_handles = [
    mpatches.Patch(color="#00ff88", label="Start"),
    mpatches.Patch(color="#ff4444", label="Goal"),
]
for t in trials:
    legend_handles.append(mpatches.Patch(color=t["color"], label=t["label"]))
ax_traj.legend(handles=legend_handles, fontsize=7, facecolor="#0f0f2e",
               labelcolor="white", loc="best")

# ── Head orientation ──────────────────────────────────────────────────────────
for t in trials:
    ts  = [r["t"]       for r in t["data"]]
    yaw = [r["yaw_deg"] for r in t["data"]]
    ax_yaw.plot(ts, yaw, color=t["color"], linewidth=1.2, label=t["label"])

ax_yaw.set_ylabel("Head orientation (°)", color="white", fontsize=8)
ax_yaw.set_title("Head orientation", color="white", fontsize=9)
ax_yaw.axhline(0, color="#444466", linewidth=0.5)
ax_yaw.legend(fontsize=7, facecolor="#0f0f2e", labelcolor="white")
ax_yaw.set_xticklabels([])

# ── Walking speed ─────────────────────────────────────────────────────────────
for t in trials:
    ts    = [r["t"]         for r in t["data"]]
    speed = [r["speed_mps"] for r in t["data"]]
    ax_speed.fill_between(ts, speed, alpha=0.5, color=t["color"])
    ax_speed.plot(ts, speed, color=t["color"], linewidth=0.8)

ax_speed.set_xlabel("Time (s)", color="white", fontsize=8)
ax_speed.set_ylabel("Walking speed (m/s)", color="white", fontsize=8)
ax_speed.set_title("Walking speed", color="white", fontsize=9)
ax_speed.set_ylim(bottom=0)

# ── Bar charts: time / distance / avg speed ───────────────────────────────────
labels      = [s["label"]     for s in stats]
times       = [s["time_s"]    for s in stats]
distances   = [s["dist_m"]    for s in stats]
avg_speeds  = [s["avg_speed"] for s in stats]
colors      = [t["color"]     for t in trials]
x           = range(len(stats))

for ax, vals, title, unit in [
    (ax_time,  times,      "Time",         "s"),
    (ax_dist,  distances,  "Distance",     "m"),
    (ax_vel,   avg_speeds, "Avg speed",    "m/s"),
]:
    bars = ax.bar(x, vals, color=colors, width=0.6)
    ax.set_title(title, color="white", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=7, color="white", rotation=20)
    ax.set_ylabel(unit, color="white", fontsize=7)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02 * max(vals),
                f"{val:.1f}", ha="center", va="bottom", color="white", fontsize=7)

fig.suptitle(f"Corridor: {corridor_key}", color="white", fontsize=11, y=0.98)

if args.out:
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {args.out}")
else:
    plt.show()
