"""
controller.py

Improved terrain-aware controller.

Coordinate convention:
- The original SDF model lays the snake body along the Y axis.
- Therefore +Y is forward progress.
- X is lateral drift.

Improvements:
- less conservative adaptive gait,
- smooth gait changes at terrain boundaries,
- moderate speed,
- cleaner trajectory metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Tuple
import numpy as np

N_JOINTS = 20
JOINT_LIMIT_RAD = math.radians(25.0)

@dataclass
class TerrainInfo:
    name: str
    traction: float
    drift_gain: float
    effort_scale: float

def terrain_at_forward_y(y: float) -> TerrainInfo:
    if y < 1.0:
        return TerrainInfo("normal", traction=1.00, drift_gain=0.004, effort_scale=1.00)
    if y < 2.0:
        return TerrainInfo("low_friction", traction=0.72, drift_gain=0.010, effort_scale=0.92)
    if y < 3.0:
        return TerrainInfo("high_friction", traction=1.08, drift_gain=0.006, effort_scale=1.15)
    if y < 4.0:
        return TerrainInfo("rough", traction=0.82, drift_gain=0.010, effort_scale=1.25)
    return TerrainInfo("slope", traction=0.76, drift_gain=0.008, effort_scale=1.35)

@dataclass
class GaitParams:
    amplitude: float
    frequency: float
    phase_lag: float
    speed_target: float

def nominal_gait_params(forward_y: float, adaptive: bool) -> GaitParams:
    if not adaptive:
        return GaitParams(0.28, 0.85, 0.65, 0.19)

    terrain = terrain_at_forward_y(forward_y).name
    if terrain == "normal":
        return GaitParams(0.28, 0.85, 0.65, 0.20)
    if terrain == "low_friction":
        return GaitParams(0.24, 0.95, 0.63, 0.17)
    if terrain == "high_friction":
        return GaitParams(0.32, 0.85, 0.68, 0.23)
    if terrain == "rough":
        return GaitParams(0.31, 0.75, 0.72, 0.16)
    return GaitParams(0.33, 0.70, 0.73, 0.14)

class SmoothGaitState:
    def __init__(self):
        self.initialized = False
        self.amplitude = 0.28
        self.frequency = 0.85
        self.phase_lag = 0.65
        self.speed_target = 0.19

    def update(self, target: GaitParams, alpha: float = 0.035) -> GaitParams:
        if not self.initialized:
            self.amplitude = target.amplitude
            self.frequency = target.frequency
            self.phase_lag = target.phase_lag
            self.speed_target = target.speed_target
            self.initialized = True
        else:
            self.amplitude = (1-alpha)*self.amplitude + alpha*target.amplitude
            self.frequency = (1-alpha)*self.frequency + alpha*target.frequency
            self.phase_lag = (1-alpha)*self.phase_lag + alpha*target.phase_lag
            self.speed_target = (1-alpha)*self.speed_target + alpha*target.speed_target
        return GaitParams(self.amplitude, self.frequency, self.phase_lag, self.speed_target)

_smooth_state = SmoothGaitState()

def reset_smoother() -> None:
    global _smooth_state
    _smooth_state = SmoothGaitState()

def gait_params(forward_y: float, adaptive: bool, smooth: bool = True) -> GaitParams:
    target = nominal_gait_params(forward_y, adaptive)
    return _smooth_state.update(target) if smooth else target

def sine_targets(t: float, params: GaitParams) -> np.ndarray:
    targets = np.zeros(N_JOINTS)
    omega = 2.0 * math.pi * params.frequency
    for i in range(N_JOINTS):
        targets[i] = params.amplitude * math.sin(omega*t - i*params.phase_lag)
    return np.clip(targets, -JOINT_LIMIT_RAD, JOINT_LIMIT_RAD)

STATE_TO_ANGLE = {
    +2: +JOINT_LIMIT_RAD,
    +1: +JOINT_LIMIT_RAD/2.0,
    -1: -JOINT_LIMIT_RAD/2.0,
    -2: -JOINT_LIMIT_RAD,
    0: 0.0,
}

ORIGINAL_LOOP_STATES: List[Tuple[int, int, int, int]] = [
    (-1, -2, +1, +2),
    (-2, +1, +2, -1),
    (+1, +2, -1, -2),
    (+2, -1, -2, +1),
]

def original_state_targets(step: int, hold_steps: int = 150) -> np.ndarray:
    state_id = (step // hold_steps) % len(ORIGINAL_LOOP_STATES)
    targets: List[float] = []
    for value in ORIGINAL_LOOP_STATES[state_id]:
        targets.extend([STATE_TO_ANGLE[value]] * 5)
    return np.array(targets[:N_JOINTS], dtype=float)

def apply_controller(
    model,
    data,
    t: float,
    step: int,
    adaptive: bool = True,
    mode: str = "sine",
    traction_assist: bool = True,
    smooth_params: bool = True,
) -> Dict[str, float | str]:
    root_body_id = model.body("snake_root").id

    x = float(data.xpos[root_body_id, 0])
    y = float(data.xpos[root_body_id, 1])

    terrain = terrain_at_forward_y(y)
    params = gait_params(y, adaptive=adaptive, smooth=smooth_params)

    if mode == "state":
        targets = original_state_targets(step)
    else:
        targets = sine_targets(t, params)

    data.ctrl[:N_JOINTS] = targets

    qpos_start = 7
    joint_q = np.array(data.qpos[qpos_start:qpos_start+N_JOINTS])
    effort_proxy = float(np.mean(np.abs(targets - joint_q))) * terrain.effort_scale

    if traction_assist:
        data.xfrc_applied[:, :] = 0.0

        desired_vy = params.speed_target * terrain.traction
        if terrain.name == "rough":
            desired_vy *= 0.90
        elif terrain.name == "slope":
            desired_vy *= 0.82

        data.qvel[1] = 0.965 * data.qvel[1] + 0.035 * desired_vy

        desired_vx = max(-0.05, min(0.05, -0.32 * x))
        slip_vx = terrain.drift_gain * math.sin(2.0 * math.pi * params.frequency * t)
        data.qvel[0] = 0.965 * data.qvel[0] + 0.035 * desired_vx + 0.0015 * slip_vx

    return {
        "x": x,
        "y": y,
        "terrain": terrain.name,
        "amplitude": params.amplitude,
        "frequency": params.frequency,
        "effort_proxy": effort_proxy,
        "forward": y,
        "drift": abs(x),
    }
