"""One bounded native simulation/render test; CPU policy not required."""
from pathlib import Path
import json
import time
import numpy as np
from PIL import Image

from quest_repro.simulation import ReachSimulation, set_render_environment

set_render_environment()
from metaworld.policies import SawyerReachV2Policy

root = Path(__file__).resolve().parents[1]
output = root / "outputs/local_simulation_smoke"
output.mkdir(parents=True, exist_ok=True)
started = time.perf_counter()
with ReachSimulation(seed=9999, video=True) as simulation:
    observation, _ = simulation.reset()
    assert observation["images"].shape == (1, 1, 3, 128, 128)
    assert observation["images"].dtype == np.uint8
    assert observation["robot_states"].shape == (1, 1, 8)
    Image.fromarray(simulation.video_frame()).save(output / "initial.png")
    expert, success = SawyerReachV2Policy(), False
    for step in range(150):
        action = np.clip(expert.get_action(simulation.observation), -1, 1).astype(np.float32)
        observation, reward, terminated, truncated, info = simulation.step(action)
        success = bool(info.get("success", False))
        if success or terminated or truncated:
            break
    Image.fromarray(simulation.video_frame()).save(output / "final.png")
report = {"render": "passed", "seed": 9999, "expert_success": success, "steps": step + 1,
          "elapsed_seconds": time.perf_counter() - started,
          "scope": "expert simulation smoke test, not learned-policy success"}
(root / "reports/local_simulation_smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
