"""
run_experiment.py

Experiment comparing fixed and adaptive gaits.

"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List
import mujoco
import numpy as np
from controller import apply_controller, reset_smoother

def run_trial(adaptive: bool, duration: float = 90.0, mode: str = "sine") -> Dict[str, float | str]:
    root = Path(__file__).resolve().parents[1]
    model = mujoco.MjModel.from_xml_path(str(root / "models" / "snake_terrain.xml"))
    data = mujoco.MjData(model)
    reset_smoother()

    steps = int(duration / model.opt.timestep)
    root_body_id = model.body("snake_root").id

    times: List[float] = []
    xs: List[float] = []
    ys: List[float] = []
    efforts: List[float] = []
    terrains: List[str] = []

    for step in range(steps):
        metrics = apply_controller(
            model, data, t=float(data.time), step=step,
            adaptive=adaptive, mode=mode, traction_assist=True,
            smooth_params=True,
        )
        mujoco.mj_step(model, data)

        times.append(float(data.time))
        xs.append(float(data.xpos[root_body_id, 0]))
        ys.append(float(data.xpos[root_body_id, 1]))
        efforts.append(float(metrics["effort_proxy"]))
        terrains.append(str(metrics["terrain"]))

    forward = ys[-1] - ys[0]
    drift = abs(xs[-1] - xs[0])
    avg_speed = forward / duration
    avg_effort = float(np.mean(efforts))
    success = 1.0 if forward >= 4.0 else 0.0

    name = "adaptive" if adaptive else "fixed"
    out = root / "results" / f"trajectory_{name}.csv"
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "step", "x", "y", "terrain", "effort"])
        for i, (time_s, x, y, terrain, effort) in enumerate(zip(times, xs, ys, terrains, efforts)):
            writer.writerow([time_s, i, x, y, terrain, effort])

    return {
        "controller": name,
        "duration_s": duration,
        "forward_m": forward,
        "avg_speed_mps": avg_speed,
        "lateral_drift_m": drift,
        "avg_effort_proxy": avg_effort,
        "success": success,
    }

def write_dict_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def compute_per_terrain(results_dir: Path) -> list[dict[str, float | str]]:
    rows = []
    for name in ["fixed", "adaptive"]:
        traj_path = results_dir / f"trajectory_{name}.csv"
        data = np.genfromtxt(traj_path, delimiter=",", names=True, dtype=None, encoding=None)
        for terrain in ["normal", "low_friction", "high_friction", "rough", "slope"]:
            mask = data["terrain"] == terrain
            if not np.any(mask):
                continue
            t = data["time"][mask]
            x = data["x"][mask]
            y = data["y"][mask]
            effort = data["effort"][mask]
            dt = max(float(t[-1] - t[0]), 1e-9)
            rows.append({
                "controller": name,
                "terrain": terrain,
                "time_s": float(dt),
                "forward_m": float(y[-1] - y[0]),
                "avg_speed_mps": float((y[-1] - y[0]) / dt),
                "drift_range_m": float(np.max(x) - np.min(x)),
                "avg_effort_proxy": float(np.mean(effort)),
            })
    return rows

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)

    rows = [run_trial(adaptive=False), run_trial(adaptive=True)]
    write_dict_csv(results_dir / "results.csv", rows)

    per_terrain = compute_per_terrain(results_dir)
    write_dict_csv(results_dir / "per_terrain_results.csv", per_terrain)

    lines = ["Fixed vs Adaptive Snake Terrain Experiment", "=" * 52]
    for row in rows:
        lines.append(
            f"{row['controller']:>8s}: "
            f"forward={row['forward_m']:.3f} m, "
            f"speed={row['avg_speed_mps']:.3f} m/s, "
            f"drift={row['lateral_drift_m']:.3f} m, "
            f"effort={row['avg_effort_proxy']:.3f}, "
            f"success={int(row['success'])}"
        )
    lines.append("")
    lines.append("Per-terrain results saved to results/per_terrain_results.csv")
    (results_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    try:
        import matplotlib.pyplot as plt
        plt.figure()
        for name in ["fixed", "adaptive"]:
            data = np.genfromtxt(results_dir / f"trajectory_{name}.csv", delimiter=",", names=True, dtype=None, encoding=None)
            x = np.asarray(data["x"], dtype=float)[::250]
            y = np.asarray(data["y"], dtype=float)[::250]
            keep = np.ones(len(x), dtype=bool)
            if len(x) > 1:
                dx = np.abs(np.diff(x, prepend=x[0]))
                dy = np.abs(np.diff(y, prepend=y[0]))
                keep = (dx < 0.15) & (dy < 0.25)
            plt.plot(x[keep], y[keep], label=name, linewidth=2)

        plt.axvline(0.0, linestyle="--", linewidth=1)
        plt.xlabel("lateral x position / drift (m)")
        plt.ylabel("forward y position (m)")
        plt.title("Snake trajectory: fixed vs adaptive gait")
        plt.legend()
        plt.grid(True)
        plt.savefig(results_dir / "trajectory.png", dpi=160, bbox_inches="tight")
        print(f"Saved {results_dir / 'trajectory.png'}")
    except Exception as e:
        print(f"Plot skipped: {e}")

if __name__ == "__main__":
    main()
