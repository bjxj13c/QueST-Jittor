"""Shared MetaWorld reach observation contract; no Torch/Jittor dependency."""
import os
import platform

import numpy as np


def set_render_environment():
    if platform.system() == "Windows":
        os.environ["MUJOCO_GL"] = "glfw"
        # PYOPENGL_PLATFORM=egl is a Linux setting. Restrict changes to this process.
        os.environ.pop("PYOPENGL_PLATFORM", None)
    else:
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


class ReachSimulation:
    def __init__(self, seed, video=False,env_name='reach-v2-goal-observable'):
        set_render_environment()
        import mujoco
        from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
        self.env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name](seed=int(seed))
        self.env.model.cam_pos[2] = [0.75, 0.075, 0.7]
        self.renderer = mujoco.Renderer(self.env.model, height=128, width=128)
        self.video_renderer = mujoco.Renderer(self.env.model, height=480, width=480) if video else None
        self.observation = None

    def reset(self):
        self.observation, info = self.env.reset()
        return self.policy_observation(), info

    def policy_observation(self):
        self.renderer.update_scene(self.env.data, camera="corner2")
        rgb = self.renderer.render()[::-1].copy()
        # 只提取当前与上一时刻的机械臂末端位置和夹爪状态，合计8维。
        state = np.concatenate((self.observation[:4], self.observation[18:22])).astype(np.float32)
        return {"images": rgb.transpose(2, 0, 1)[None, None].copy(),
                "robot_states": state[None, None], "task_id": np.zeros(1, dtype=np.int64)}

    def video_frame(self):
        renderer = self.video_renderer or self.renderer
        renderer.update_scene(self.env.data, camera="corner2")
        return renderer.render()[::-1].copy()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        # 执行前检查4维控制量，避免将NaN或错误形状送入仿真。
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError("Expected finite 4-D control action")
        self.observation, reward, terminated, truncated, info = self.env.step(np.clip(action, -1.0, 1.0))
        return self.policy_observation(), reward, terminated, truncated, info

    @staticmethod
    def close_renderer(renderer):
        if renderer is None:
            return
        if hasattr(renderer, "close"):
            renderer.close()
        else:
            renderer._mjr_context.free()
            renderer._gl_context.free()

    def close(self):
        self.close_renderer(self.video_renderer)
        self.close_renderer(self.renderer)
        self.env.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
