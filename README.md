# Terrain Surveyor Snake Robot

This is a working MuJoCo version of the serial snake robot concept that adapts to the terrain environment.

## Install

```bash
pip install mujoco numpy matplotlib
```

On macOS, the viewer must be run with `mjpython`, not normal `python`.

## Run the live simulator

On macOS:

```bash
mjpython src/sim_viewer.py
```

On Linux/Windows:

```bash
python src/sim_viewer.py
```

Useful variants:

```bash
mjpython src/sim_viewer.py --fixed
mjpython src/sim_viewer.py --mode state
mjpython src/sim_viewer.py --no-traction-assist
```

- `--fixed`: uses the same gait everywhere.
- default: uses terrain-aware adaptive gait.
- `--mode state`: uses the original-style segmented +2/+1/-1/-2 state machine.
- default `--mode sine`: uses a smoother traveling sine wave.
- `--no-traction-assist`: disables the reduced-order propulsion force. The snake still wiggles, but may not move very far.

## Run the experiment

```bash
python src/run_experiment.py
```

This saves:

```text
results/results.csv
results/trajectory_fixed.csv
results/trajectory_adaptive.csv
results/summary.txt
results/trajectory.png
```
