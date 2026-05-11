"""
render_frames.py

Non-viewer rendering path for macOS.

This does NOT use mujoco.viewer.launch_passive and does NOT require mjpython.
It runs the same MuJoCo simulation headlessly and saves image frames to:

    results/frames/

It also tries to create:

    results/snake_sim.gif

Run:
    python src/render_frames.py

Optional:
    python src/render_frames.py --fixed
    python src/render_frames.py --mode state
    python src/render_frames.py --duration 20
    python src/render_frames.py --every 20

Install:
    pip install mujoco numpy matplotlib imageio

If imageio is not installed, the script still saves PNG frames.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from controller import apply_controller, reset_smoother


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed", action="store_true", help="Use fixed gait instead of adaptive gait.")
    parser.add_argument("--mode", choices=["sine", "state"], default="sine")
    parser.add_argument("--duration", type=float, default=90.0)
    parser.add_argument("--every", type=int, default=30, help="Save one frame every N simulation steps.")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    model_path = root / "models" / "snake_terrain.xml"
    results_dir = root / "results"
    frames_dir = results_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Clean old frames.
    for p in frames_dir.glob("frame_*.png"):
        p.unlink()

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    # Use the camera defined in the XML if available; otherwise use free camera.
    camera_name = "track"
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)

    reset_smoother()
    total_steps = int(args.duration / model.opt.timestep)
    saved = 0

    # matplotlib is used only for writing PNG frames.
    import matplotlib.pyplot as plt

    for step in range(total_steps):
        metrics = apply_controller(
            model,
            data,
            t=float(data.time),
            step=step,
            adaptive=not args.fixed,
            mode=args.mode,
            traction_assist=True,
            smooth_params=True,
        )

        mujoco.mj_step(model, data)

        if step % args.every == 0:
            if camera_id >= 0:
                renderer.update_scene(data, camera=camera_name)
            else:
                renderer.update_scene(data)

            img = renderer.render()
            frame_path = frames_dir / f"frame_{saved:04d}.png"
            plt.imsave(frame_path, img)

            if saved % 10 == 0:
                print(
                    f"saved {frame_path.name} "
                    f"t={data.time:.2f} terrain={metrics['terrain']} "
                    f"x={metrics['x']:.2f} y={metrics['y']:.2f}"
                )
            saved += 1

    print(f"\nSaved {saved} frames to {frames_dir}")

    # Try to create a GIF if imageio is available.
    try:
        import imageio.v2 as imageio

        gif_path = results_dir / "snake_sim.gif"
        frame_paths = sorted(frames_dir.glob("frame_*.png"))

        # Use a subset if there are too many frames.
        images = [imageio.imread(p) for p in frame_paths]
        imageio.mimsave(gif_path, images, duration=0.05)
        print(f"Saved GIF to {gif_path}")
    except Exception as e:
        print(f"GIF skipped: {e}")
        print("You still have PNG frames in results/frames/.")


if __name__ == "__main__":
    main()
