"""
check_install.py

Quick check that MuJoCo is installed and the XML model loads.

Run:
    python src/check_install.py
"""

from pathlib import Path
import mujoco

root = Path(__file__).resolve().parents[1]
model_path = root / "models" / "snake_terrain.xml"

model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)

print("MuJoCo import: OK")
print(f"Model loaded: {model_path}")
print(f"Number of bodies: {model.nbody}")
print(f"Number of joints: {model.njnt}")
print(f"Number of actuators: {model.nu}")

# Step a few times to catch obvious physics/XML issues.
for _ in range(100):
    mujoco.mj_step(model, data)

print("100 simulation steps: OK")
