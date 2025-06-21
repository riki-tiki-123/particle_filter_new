import numpy as np
import torch

def preprocess_data(prices, window_size):
    """Generic windowing utility for price series (not used in this project)."""
    states = []
    for t in range(len(prices) - window_size):
        window = prices[t:t + window_size]
        state = [window[i + 1] - window[i] for i in range(len(window) - 1)]
        states.append(state)
    return np.array(states)

def discount_rewards(rewards, gamma):
    """Compute discounted cumulative rewards (RL-style utility)."""
    discounted = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = rewards[t] + gamma * running_add
        discounted[t] = running_add
    return discounted

def normalize(x):
    """Z-score normalize an array."""
    x = np.array(x)
    return (x - np.mean(x)) / (np.std(x) + 1e-8) if np.std(x) > 0 else x - np.mean(x)

def softmax(x):
    """Stable softmax."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def to_tensor(np_array, device='cpu', dtype=torch.float32):
    """Convert numpy array to PyTorch tensor."""
    return torch.tensor(np_array, dtype=dtype, device=device)

def draw_uniform_particles(n_particles, n_balls, init_area=((0, 50), (0, 50)), speed_range=(5, 15)):
    """
    Generate uniformly drawn initial particles for n balls.
    Used in run.py if you want to modularize init_particles.
    """
    particles = []
    for _ in range(n_particles):
        particle = []
        for _ in range(n_balls):
            x = np.random.uniform(*init_area[0])
            y = np.random.uniform(*init_area[1])
            angle = np.random.uniform(0, np.pi / 2)
            speed = np.random.uniform(*speed_range)
            vx = speed * np.cos(angle)
            vy = speed * np.sin(angle)
            particle.extend([x, y, vx, vy])
        particles.append(particle)
    return np.array(particles)
