import os

import numpy as np

# MetaDrive's top-down renderer goes through pygame. Without a dummy video driver it
# tries to open a real window and dies over SSH / in Docker.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


def _base_config(cfg):
    from metadrive.envs.top_down_env import TopDownMetaDrive

    defaults = TopDownMetaDrive.default_config()
    out = {
        "use_render": False,
        "manual_control": False,
        "resolution_size": cfg.resolution,
        "frame_stack": cfg.frame_stack,
        "post_stack": cfg.frame_stack,
        "num_scenarios": cfg.num_scenarios,
        "start_seed": cfg.start_seed,
        "traffic_density": cfg.traffic_density,
        "map": cfg.map_size,
        "horizon": cfg.horizon,
        "log_level": 50,
    }
    # 0.4.x renamed rgb_clip -> norm_pixel. We want raw 0-255 either way so the stored
    # frames stay uint8.
    if "norm_pixel" in defaults:
        out["norm_pixel"] = False
    else:
        out["rgb_clip"] = False
    return out


class TrafficSceneEnv:
    """Thin wrapper over MetaDrive's top-down env.

    Exposes uint8 HWC observations at the configured resolution and the Dreamer-style
    (obs, action, reward, cont, is_first) tuple layout. Kept deliberately thin: the
    world model should not care which simulator produced the pixels.
    """

    def __init__(self, cfg):
        from metadrive.envs.top_down_env import TopDownMetaDrive

        self.cfg = cfg
        self._env = TopDownMetaDrive(_base_config(cfg))
        self.action_dim = int(self._env.action_space.shape[0])
        obs, _ = self._env.reset(seed=cfg.start_seed)
        self.obs_shape = self._to_uint8(obs).shape
        self._needs_reset = True

    def _to_uint8(self, obs):
        obs = np.asarray(obs)
        if obs.dtype != np.uint8:
            hi = obs.max()
            obs = obs * 255.0 if hi <= 1.0 + 1e-6 else obs
            obs = np.clip(obs, 0, 255).astype(np.uint8)
        return obs

    def reset(self, seed=None):
        obs, info = self._env.reset(seed=seed)
        self._needs_reset = False
        return self._to_uint8(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(np.asarray(action, np.float32))
        if terminated or truncated:
            self._needs_reset = True
        return self._to_uint8(obs), float(reward), bool(terminated), bool(truncated), info

    @property
    def agent(self):
        return self._env.agent

    @property
    def engine(self):
        return self._env.engine

    def close(self):
        self._env.close()

    def ego_position(self):
        return np.asarray(self._env.agent.position, dtype=np.float32)

    def ego_state(self):
        v = self._env.agent
        return {
            "position": np.asarray(v.position, np.float32),
            "heading": float(v.heading_theta),
            "speed": float(v.speed),
        }
