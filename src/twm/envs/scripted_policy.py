import numpy as np


def wrap_to_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class NoisyLaneFollower:
    """Hand-written PD lane keeper with correlated exploration noise.

    Why not just use MetaDrive's IDM expert: a world model trained on a single
    deterministic policy never sees the action channel vary, so the transition model
    learns to ignore actions entirely. The OU noise below (and the per-episode lateral
    bias) is there purely to make the dataset action-conditioned.
    """

    def __init__(self, rng, noise=0.25, target_speed_kmh=30.0, ou_theta=0.15):
        self.rng = rng
        self.noise = noise
        self.target_speed = target_speed_kmh
        self.ou_theta = ou_theta
        self._ou = np.zeros(2, np.float32)
        self._lat_bias = 0.0

    def reset(self):
        self._ou[:] = 0.0
        # A constant lateral offset for the whole episode makes some episodes hug the
        # left of the lane and some the right, widening the state distribution.
        self._lat_bias = float(self.rng.normal(0.0, 0.6))

    def __call__(self, vehicle):
        steer, throttle = self._base_action(vehicle)
        self._ou += -self.ou_theta * self._ou + self.rng.normal(0, self.noise, 2).astype(np.float32)
        action = np.array([steer, throttle], np.float32) + self._ou
        return np.clip(action, -1.0, 1.0)

    def _base_action(self, vehicle):
        lane = getattr(vehicle, "lane", None)
        if lane is None:
            return 0.0, 0.3
        try:
            longitude, lateral = lane.local_coordinates(vehicle.position)
            # Aim a few metres ahead rather than at the current point, otherwise the
            # controller oscillates at anything above walking pace.
            lookahead = min(longitude + 6.0, lane.length)
            target_heading = lane.heading_theta_at(lookahead)
        except Exception:
            return 0.0, 0.3
        heading_err = wrap_to_pi(target_heading - vehicle.heading_theta)
        lateral_err = lateral - self._lat_bias
        steer = 1.2 * heading_err - 0.12 * lateral_err
        speed_err = self.target_speed - vehicle.speed_km_h
        throttle = np.clip(0.05 * speed_err, -0.6, 0.6)
        return float(np.clip(steer, -1, 1)), float(throttle)
