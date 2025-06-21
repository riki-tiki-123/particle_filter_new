import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

def animate_trajectories(time_list, true_positions, observations, estimates, filename="particle_filter_tracking.mp4"):
    """
    Create an animation showing true, observed, and estimated ball positions in 2D.
    Saves the animation to an mp4 file.
    """
    n_obs, n_balls, _ = true_positions.shape

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.arange(n_balls))

    # Initialize plots
    true_dots = [ax.plot([], [], 'o', color=colors[i], label=f"True Ball {i+1}")[0] for i in range(n_balls)]
    est_dots  = [ax.plot([], [], 'x', color=colors[i], label=f"Est Ball {i+1}")[0] for i in range(n_balls)]
    obs_dots  = [ax.plot([], [], '.', color=colors[i], alpha=0.4, label=f"Obs Ball {i+1}")[0] for i in range(n_balls)]

    ax.set_xlim(0, np.max(true_positions[..., 0]) + 10)
    ax.set_ylim(np.min(true_positions[..., 1]) - 20, np.max(true_positions[..., 1]) + 10)
    ax.set_title("Particle Filter Tracking (True / Est / Obs)")
    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.grid(True)
    ax.legend(loc="lower left", ncol=1)

    def update(frame):
        for b in range(n_balls):
            true_dots[b].set_data(true_positions[frame, b, 0], true_positions[frame, b, 1])
            est_dots[b].set_data(estimates[frame, b, 0], estimates[frame, b, 1])
            if observations[frame] is not None:
                obs_dots[b].set_data(observations[frame][b][0], observations[frame][b][1])
            else:
                obs_dots[b].set_data([], [])
        ax.set_title(f"t = {time_list[frame]:.2f} s")
        return true_dots + est_dots + obs_dots

    anim = FuncAnimation(fig, update, frames=n_obs, blit=True)

    writer = FFMpegWriter(fps=10, bitrate=1800)
    anim.save(filename, writer=writer)
    print(f"[INFO] Animation saved to {filename}")
