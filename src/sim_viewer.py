"""
sim_viewer.py

Live MuJoCo viewer for the terrain-aware snake.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import mujoco
import mujoco.viewer

from controller import apply_controller, reset_smoother


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed", action="store_true", help="Use fixed gait instead of adaptive gait.")
    parser.add_argument("--mode", choices=["sine", "state"], default="sine",
                        help="sine = smooth traveling wave; state = original segmented +2/+1/-1/-2 style.")
    parser.add_argument("--no-traction-assist", action="store_true",
                        help="Disable reduced-order forward traction force. The snake will still wiggle, but may not travel far.")
    parser.add_argument("--duration", type=float, default=90.0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    model_path = root / "models" / "snake_terrain.xml"

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # Start camera looking at the snake.
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = model.body("snake_root").id
        viewer.cam.distance = 6.0
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25

        reset_smoother()
        step = 0
        last_print = 0.0
        start_wall = time.time()

        while viewer.is_running():
            t = float(data.time)

            metrics = apply_controller(
                model,
                data,
                t=t,
                step=step,
                adaptive=not args.fixed,
                mode=args.mode,
                traction_assist=not args.no_traction_assist,
                smooth_params=True,
            )

            mujoco.mj_step(model, data)
            viewer.sync()

            if t - last_print >= 1.0:
                print(
                    f"t={t:5.2f} terrain={metrics['terrain']:>12s} "
                    f"forward_y={metrics['y']:6.3f} drift_x={metrics['x']:6.3f} "
                    f"A={metrics['amplitude']:.2f} f={metrics['frequency']:.2f} "
                    f"effort={metrics['effort_proxy']:.3f}"
                )
                last_print = t

            step += 1

            if t >= args.duration:
                break

            # Keep close to real time.
            elapsed_wall = time.time() - start_wall
            if data.time > elapsed_wall:
                time.sleep(min(0.002, data.time - elapsed_wall))


if __name__ == "__main__":
    main()
