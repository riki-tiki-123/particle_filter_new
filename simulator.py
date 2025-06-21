# simulator.py

import numpy as np

# Global gravity vector (downward acceleration, in m/s^2)
GRAVITY = np.array([0.0, -9.81])  # y-axis points up, so -9.81 acts downward

class Ball:
    """
    Represents a single ball in 2D space, with given initial position and velocity.
    Used for generating the ground-truth trajectory in the simulator.
    """
    def __init__(self, init_pos, init_vel):
        """
        Args:
            init_pos: (2,) array-like, initial position [x, y] in meters
            init_vel: (2,) array-like, initial velocity [vx, vy] in meters/second
        """
        self.init_pos = np.array(init_pos, dtype=float)
        self.init_vel = np.array(init_vel, dtype=float)

    def position_at(self, t):
        """
        Compute the (x, y) position of the ball at time t (seconds)
        under constant gravity (projectile motion).
        """
        return self.init_pos + self.init_vel * t + 0.5 * GRAVITY * t**2


def simulate_trajectories(balls, t_end, dt):
    """
    Simulate true trajectories for a list of Ball objects.

    Args:
        balls: list of Ball instances (length n_balls)
        t_end: float, end time (seconds)
        dt:    float, time step for simulation (seconds)

    Returns:
        ts: array of time stamps [0, dt, 2dt, ..., t_end]
        true_trajs: list of arrays, one per ball,
                    each of shape (len(ts), 2) for [x, y] over time

    Notes:
        This is used to generate ground-truth data for comparison and
        for producing synthetic observations.
    """
    ts = np.arange(0, t_end + 1e-9, dt)
    true_trajs = []
    for ball in balls:
        traj = np.vstack([ball.position_at(t) for t in ts])
        true_trajs.append(traj)
    return ts, true_trajs


def sample_noisy_observations(ts, true_trajs,
                              obs_dt_mean,
                              obs_noise_std=1.0,
                              dropout_prob=0.1):
    """
    Generate noisy, possibly missing observations from true trajectories.

    Args:
        ts: array of time stamps for the ground-truth trajectories
        true_trajs: list of arrays, one per ball (output from simulate_trajectories)
        obs_dt_mean: mean time (seconds) between sensor observations (randomized)
        obs_noise_std: standard deviation of Gaussian observation noise (meters)
        dropout_prob: probability of *all* observations dropping out at a given frame

    Returns:
        obs_times: list of observation times (float)
        observations: list of observations at each time;
                      each is either:
                          - list of [x, y] (one per ball), or
                          - None (for dropout)

    Method:
        At each step, sample a random time increment (exp dist, mean=obs_dt_mean),
        interpolate each ball's true position to that time,
        add noise (simulate measurement error),
        and possibly drop the frame entirely (simulate sensor outage).
    """
    n_balls = len(true_trajs)
    obs_times, observations = [], []

    t = 0.0
    while t < ts[-1]:
        # Variable interval between observations (to simulate real-world sensor)
        dt = np.random.exponential(obs_dt_mean)
        t += dt
        if t > ts[-1]:
            break

        # Interpolate true position for each ball at time t
        idx = np.searchsorted(ts, t)
        idx0 = max(0, idx - 1)
        idx1 = min(len(ts) - 1, idx)
        w = (t - ts[idx0]) / (ts[idx1] - ts[idx0] + 1e-8)  # interpolation weight

        if np.random.rand() < dropout_prob:
            # Sensor dropped out: no observation for this frame
            observations.append(None)
        else:
            frame = []
            for traj in true_trajs:
                pos = (1 - w) * traj[idx0] + w * traj[idx1]  # linear interpolation
                noisy = pos + np.random.randn(2) * obs_noise_std
                frame.append(noisy.tolist())
            observations.append(frame)

        obs_times.append(t)

    return obs_times, observations


if __name__ == "__main__":
    # Example/test: simulate and print first 10 noisy obs for 3 balls
    rng = np.random.RandomState(0)
    balls = []
    for _ in range(3):
        pos = rng.uniform([0, 0], [50, 50])
        speed = rng.uniform(5, 15)
        angle = rng.uniform(0, np.pi/2)
        vel = speed * np.array([np.cos(angle), np.sin(angle)])
        balls.append(Ball(pos, vel))

    ts, true_trajs = simulate_trajectories(balls, t_end=5.0, dt=0.01)
    obs_times, obs = sample_noisy_observations(ts, true_trajs,
                                              obs_dt_mean=0.1,
                                              obs_noise_std=0.5,
                                              dropout_prob=0.2)
    for t, frame in zip(obs_times[:10], obs[:10]):
        print(f"t={t:.2f}s  obs={frame}")
