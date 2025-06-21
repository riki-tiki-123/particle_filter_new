# particle_filter.py

import numpy as np
import itertools

class ParticleFilter:
    """
    Particle Filter for n 2D balls (projectile motion with gravity and collision):
      - state x = [x1, y1, vx1, vy1, ..., xn, yn, vxn, vyn]  (dim=4*n)
      - observation y = [[x1, y1], [x2, y2], ..., [xn, yn]]  (shape = n x 2, order unknown)
    """

    def __init__(self,
                 n_particles: int,
                 n_balls: int,
                 process_noise_cov: np.ndarray,
                 obs_noise_cov: np.ndarray,
                 init_particles: np.ndarray):
        """
        n_particles      : number of particles
        n_balls          : number of balls (targets) being tracked
        process_noise_cov: Q, (4*n)x(4*n) covariance for process noise
        obs_noise_cov    : R, (2)x(2) covariance for single ball observation noise
        init_particles   : (n_particles, 4*n) initial states (positions, velocities)
        """
        self.N = n_particles
        self.n_balls = n_balls
        self.particles = init_particles.copy()  # (N, 4*n)
        self.weights = np.ones(self.N) / self.N
        self.Q = process_noise_cov
        self.R = obs_noise_cov

        # Precompute for Gaussian likelihood
        self._invR = np.linalg.inv(self.R)
        self._normR = 1.0 / np.sqrt((2*np.pi)**2 * np.linalg.det(self.R))

    def predict(self, dt: float, ball_radius=0.5):
        """
        Step: Propagate all particles forward with gravity and process noise, including elastic collision.
        """
        g = np.array([0.0, -9.81])
        N, n = self.N, self.n_balls

        # Position and velocity update for all balls in each particle
        for ball in range(n):
            idx = slice(4*ball, 4*ball+2)
            vel_idx = slice(4*ball+2, 4*ball+4)
            # Update positions
            self.particles[:, idx] += self.particles[:, vel_idx]*dt + 0.5*g*dt*dt
            # Update velocities (gravity)
            self.particles[:, vel_idx] += g*dt

        # Add zero-mean Gaussian process noise to each particle's whole state
        noise = np.random.multivariate_normal(np.zeros(4*n), self.Q, size=N)
        self.particles += noise

        # After motion: resolve collisions **within each particle**
        self._resolve_collisions(ball_radius)

    def _resolve_collisions(self, ball_radius=0.5):
        """
        For each particle, check every ball pair and apply 2D elastic collision if they're overlapping.
        """
        n = self.n_balls
        for i in range(self.N):
            for a in range(n):
                for b in range(a+1, n):
                    idx_a = 4*a
                    idx_b = 4*b
                    pos_a = self.particles[i, idx_a:idx_a+2]
                    pos_b = self.particles[i, idx_b:idx_b+2]
                    dist = np.linalg.norm(pos_a - pos_b)
                    if dist < 2*ball_radius:  # Overlap detected!
                        va = self.particles[i, idx_a+2:idx_a+4]
                        vb = self.particles[i, idx_b+2:idx_b+4]
                        delta = pos_b - pos_a
                        norm = delta / (np.linalg.norm(delta) + 1e-12)
                        va_n = np.dot(va, norm)
                        vb_n = np.dot(vb, norm)
                        # Exchange normal components (equal mass, elastic)
                        va_new = va + (vb_n - va_n) * norm
                        vb_new = vb + (va_n - vb_n) * norm
                        self.particles[i, idx_a+2:idx_a+4] = va_new
                        self.particles[i, idx_b+2:idx_b+4] = vb_new
                        # Optional: separate balls a bit to avoid stickiness
                        overlap = 2*ball_radius - dist
                        self.particles[i, idx_a:idx_a+2] -= norm * overlap/2
                        self.particles[i, idx_b:idx_b+2] += norm * overlap/2

    def update(self, observations):
        """
        Update particle weights based on observations (list of [x, y] for each ball or None for dropout).
        Handles indistinguishable balls by summing likelihoods for all permutations.
        """
        if observations is None:
            # Missing frame: skip weight update
            return

        obs = np.asarray(observations)  # shape (n,2)
        n = self.n_balls

        # For each particle: compute best association likelihood
        # For moderate n (n <= 4), enumerate all permutations (n!).
        # For large n: can use Hungarian matching or soft assignment, but permutations is fine for this exam.
        perm_indices = list(itertools.permutations(range(n)))
        likelihoods = np.zeros(self.N)
        for p in perm_indices:
            # For each permutation: associate obs[i] with particle's ball p[i]
            # Compute Mahalanobis distance per ball and sum log-likelihoods
            particle_ball_positions = self.particles[:, np.concatenate([[4*i, 4*i+1] for i in p])]  # (N, 2*n)
            obs_flat = obs.flatten()
            # Reshape for broadcast
            diffs = particle_ball_positions - obs_flat
            # Compute -0.5 * sum( diff^T R^-1 diff )
            exponents = -0.5 * np.sum(diffs.reshape(self.N, n, 2) @ self._invR * diffs.reshape(self.N, n, 2), axis=(1,2))
            # Each permutation: likelihood = exp(exponents)
            likelihoods += np.exp(exponents)

        # Normalize over number of permutations and scale
        likelihoods *= self._normR ** n / len(perm_indices)

        # Weight update
        self.weights *= likelihoods
        self.weights += 1e-300  # avoid zeros
        self.weights /= np.sum(self.weights)

    def resample(self):
        """
        Systematic resampling of particles based on current weights.
        """
        N = self.N
        positions = (np.arange(N) + np.random.rand()) / N
        cumulative = np.cumsum(self.weights)
        new_particles = np.zeros_like(self.particles)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative[j]:
                new_particles[i] = self.particles[j]
                i += 1
            else:
                j += 1
        self.particles = new_particles
        self.weights.fill(1.0 / N)

    def estimate(self) -> np.ndarray:
        """
        Return the weighted mean state: shape (4*n,)
        (Note: for multi-ball, usually you want to extract n cluster centers from the cloud!)
        """
        return np.average(self.particles, weights=self.weights, axis=0)

# ---- END FILE ----
