import numpy as np
import matplotlib.pyplot as plt
from simulator import Ball, simulate_trajectories, sample_noisy_observations
from particle_filter import ParticleFilter
from clustering import kmeans_positions
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' if you prefe

def run_tracking_example(
    n_balls=None, #This is defined later in the next if method. 
    t_end=5.0,


    #this manages the distance between time steps in the true trajectory.
    dt_sim=0.01,

    #these two variables will define how large the time steps are, obs_dt_mean goes into
    #the simulator through sample_noisy_observations
    obs_dt_mean=0.1,
    
    
    obs_noise_std=1.5,
    dropout_prob=0.2,
    n_particles=1000,
    Q_scale=1e-2,
    R_scale=1.0,
    init_area=((0, 50), (0, 50)),


    #over the next two options we can define the initial speed and angle that the balls take off at.
    init_speed_range=(5, 50),
    init_angle_range=(0, np.pi/2),
    ball_radius=0.5,
    random_seed=0
):
    rng = np.random.RandomState(random_seed)

    if n_balls is None:
        # This is where you can decide how many balls that you want in the experiment. 
        n_balls = rng.randint(2, 4)
        print(f"[INFO] Randomized number of balls: {n_balls}")

    # Simulate balls
    balls = []
    for _ in range(n_balls):
        init_pos = rng.uniform([init_area[0][0], init_area[1][0]],
                               [init_area[0][1], init_area[1][1]])
        speed = rng.uniform(*init_speed_range)
        angle = rng.uniform(*init_angle_range)
        init_vel = speed * np.array([np.cos(angle), np.sin(angle)])
        balls.append(Ball(init_pos, init_vel))

    ts, true_trajs = simulate_trajectories(balls, t_end=t_end, dt=dt_sim)

    # Generate observations
    obs_times, observations = sample_noisy_observations(
        ts, true_trajs,
        obs_dt_mean=obs_dt_mean,
        obs_noise_std=obs_noise_std,
        dropout_prob=dropout_prob
    )

        # ---- DEBUG: Log which times are dropouts and which are observations ----
    print("==== OBSERVATION DROP DEBUG ====")
    for i, obs in enumerate(observations):
        t = obs_times[i]
        if obs is None:
            print(f"[DROPOUT] t={t:.3f} s: DROPPED frame.")
        else:
            print(f"[OBSERVED] t={t:.3f} s: {obs}")
    print("================================\n")

    # Initialize particles
    init_particles = []
    for _ in range(n_particles):
        particle = []
        for _ in range(n_balls):
            x = rng.uniform(*init_area[0])
            y = rng.uniform(*init_area[1])
            speed = rng.uniform(*init_speed_range)
            angle = rng.uniform(*init_angle_range)
            vx = speed * np.cos(angle)
            vy = speed * np.sin(angle)
            particle.extend([x, y, vx, vy])
        init_particles.append(particle)
    init_particles = np.array(init_particles)

    Q = Q_scale * np.eye(4 * n_balls)
    R = (obs_noise_std ** 2) * np.eye(2)

    #here is where the run.py code interacts with the particle filter. 

    pf = ParticleFilter(
        n_particles=n_particles,
        n_balls=n_balls,
        process_noise_cov=Q,
        obs_noise_cov=R,
        init_particles=init_particles
    )

    est_positions_over_time = []
    true_positions_over_time = []
    time_list = []
    dropout_indices = []

    last_time = 0.0
    for idx, obs in enumerate(observations):
        t_obs = obs_times[idx]
        dt = t_obs - last_time
        last_time = t_obs

        pf.predict(dt, ball_radius=ball_radius)
        pf.update(obs)
        pf.resample()

        if obs is None:
            dropout_indices.append(idx)

        positions = []
        for b in range(n_balls):
            positions.append(pf.particles[:, 4*b:4*b+2])
        particle_positions = np.vstack(positions).reshape(-1, 2)
        weights_tiled = np.tile(pf.weights, n_balls)

        centers_kmeans = kmeans_positions(
            particles=particle_positions,
            weights=weights_tiled,
            n_clusters=n_balls,
            n_resample=None,
            random_state=random_seed
        )
        est_positions_over_time.append(centers_kmeans)

        idx_ts = np.searchsorted(ts, t_obs)
        if idx_ts >= len(ts):
            idx_ts = len(ts) - 1
        true_frame = np.vstack([traj[idx_ts] for traj in true_trajs])
        true_positions_over_time.append(true_frame)
        time_list.append(t_obs)

    time_list = np.array(time_list)
    true_positions_over_time = np.stack(true_positions_over_time, axis=0)
    est_positions_over_time  = np.stack(est_positions_over_time, axis=0)

    # === XY Trajectory Plot ===
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw initial area boundary (box from (0,0) to (50,50))
    init_x_range = init_area[0]
    init_y_range = init_area[1]
    ax.plot([init_x_range[0], init_x_range[1], init_x_range[1], init_x_range[0], init_x_range[0]],
            [init_y_range[0], init_y_range[0], init_y_range[1], init_y_range[1], init_y_range[0]],
            linestyle='-', color='gray', linewidth=1.5, label='Init Area')

    valid_y = true_positions_over_time[:, :, 1]
    valid_y = valid_y[valid_y > 0]
    y_min = 0
    y_max = np.max(valid_y) + 10  # Add margin if desired


    # Shaded dropout regions (at true X position during dropout)
    for idx in dropout_indices:
        for b in range(n_balls):
            x = true_positions_over_time[idx, b, 0]
            ax.fill_betweenx(
                y=[0, y_max], x1=x - 0.5, x2=x + 0.5,
                color='purple', alpha=0.3
            )


    for b in range(n_balls):
        # True path (solid), only above ground
        mask_true = true_positions_over_time[:, b, 1] >= 0 # do not have the graph show below 0
        ax.plot(true_positions_over_time[mask_true, b, 0],
                true_positions_over_time[mask_true, b, 1],
                linestyle='-', linewidth=2, label=f"True Ball {b+1}")

        # Particle filter prediction (dashed), only above ground
        mask_true = true_positions_over_time[:, b, 1] >= 0 # do not have the graph show below 0
        mask_est = est_positions_over_time[:, b, 1] >= 0 
        ax.plot(est_positions_over_time[mask_est, b, 0],
                est_positions_over_time[mask_est, b, 1],
                linestyle='--', color=ax.lines[-1].get_color(), linewidth=1.7, label=f"PF Est. {b+1}")

        # Observed path (faint dots), only above ground
        mask_true = true_positions_over_time[:, b, 1] >= 0 # do not have the graph show below 0
        obs_x = [observations[i][b][0] if observations[i] is not None and observations[i][b][1] >= 0 else np.nan for i in range(len(time_list))] 
        obs_y = [observations[i][b][1] if observations[i] is not None and observations[i][b][1] >= 0 else np.nan for i in range(len(time_list))]
        ax.plot(obs_x, obs_y, marker='o', linestyle='None', alpha=0.4, markersize=6, label=f"Obs Ball {b+1}")

    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_title(f"XY Trajectories: True, Observed, and Particle Filter Estimated ({n_balls} Balls)")
    all_x = true_positions_over_time[:, :, 0]
    x_min = np.min(all_x[all_x > 0]) - 10
    x_max = np.max(all_x) + 10
    ax.set_xlim(x_min, x_max)

    ax.set_ylim(init_y_range[0] - 10, init_y_range[1] + 30)
    ax.legend(loc="best", ncol=2)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("particle_filter_xy_trajectory.png", dpi=150)
    plt.show()
    print("[INFO] Saved XY trajectory plot with dropouts as particle_filter_xy_trajectory.png")

    # === 2D Time-Series Plot (Y and X vs Time) ===
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for b in range(n_balls):
        # Y vs Time
        axs[0].plot(time_list, true_positions_over_time[:, b, 1],
                    linestyle='--', label=f"True Y Ball {b+1}")
        est_y = [est_positions_over_time[i, b, 1]
                 for i in range(len(time_list)) if observations[i] is not None]
        est_t = [time_list[i]
                 for i in range(len(time_list)) if observations[i] is not None]
        axs[0].plot(est_t, est_y, marker='o', linestyle='None', label=f"Est Y Ball {b+1}")
        obs_y = [observations[i][b][1] if observations[i] is not None else np.nan
                 for i in range(len(time_list))]
        axs[0].plot(time_list, obs_y, linestyle='-', alpha=0.4, label=f"Obs Y Ball {b+1}")

        # X vs Time
        axs[1].plot(time_list, true_positions_over_time[:, b, 0],
                    linestyle='--', label=f"True X Ball {b+1}")
        est_x = [est_positions_over_time[i, b, 0]
                 for i in range(len(time_list)) if observations[i] is not None]
        axs[1].plot(est_t, est_x, marker='o', linestyle='None', label=f"Est X Ball {b+1}")
        obs_x = [observations[i][b][0] if observations[i] is not None else np.nan
                 for i in range(len(time_list))]
        axs[1].plot(time_list, obs_x, linestyle='-', alpha=0.4, label=f"Obs X Ball {b+1}")

    axs[0].set_ylabel("Y Position (m)")
    axs[1].set_ylabel("X Position (m)")
    axs[1].set_xlabel("Time (s)")
    axs[0].legend(loc="upper right", ncol=3)
    axs[0].set_title(f"True, Estimated, and Observed Trajectories of {n_balls} Balls")
    plt.tight_layout()
    plt.savefig("particle_filter_time_series.png", dpi=150)
    print("[INFO] Saved time series plot as particle_filter_time_series.png")

    # === 3D Trajectory Plot ===
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    for b in range(n_balls):
        ax.plot(true_positions_over_time[:, b, 0],
                true_positions_over_time[:, b, 1],
                time_list,
                linestyle='--', label=f"True Ball {b+1}")
        est_x = [est_positions_over_time[i, b, 0]
                 for i in range(len(time_list)) if observations[i] is not None]
        est_y = [est_positions_over_time[i, b, 1]
                 for i in range(len(time_list)) if observations[i] is not None]
        est_t = [time_list[i]
                 for i in range(len(time_list)) if observations[i] is not None]
        ax.plot(est_x, est_y, est_t, marker='o', linestyle='None', label=f"Est Ball {b+1}")
        obs_x = [observations[i][b][0] if observations[i] is not None else np.nan
                 for i in range(len(time_list))]
        obs_y = [observations[i][b][1] if observations[i] is not None else np.nan
                 for i in range(len(time_list))]
        ax.plot(obs_x, obs_y, time_list, linestyle='-', alpha=0.4, label=f"Obs Ball {b+1}")

    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_zlabel("Time (s)")
    ax.set_title(f"3D Trajectory View (X, Y, Time)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("particle_filter_3d_trajectory.png", dpi=150)
    print("[INFO] Saved 3D trajectory plot as particle_filter_3d_trajectory.png")

if __name__ == "__main__":
    run_tracking_example()


    rng = np.random.RandomState(random_seed)

    if n_balls is None:
        # This is where you can decide how many balls that you want in the experiment. 
        n_balls = rng.randint(2, 4)
        print(f"[INFO] Randomized number of balls: {n_balls}")

    # Simulate balls
    balls = []
    for _ in range(n_balls):
        init_pos = rng.uniform([init_area[0][0], init_area[1][0]],
                               [init_area[0][1], init_area[1][1]])
        speed = rng.uniform(*init_speed_range)
        angle = rng.uniform(*init_angle_range)
        init_vel = speed * np.array([np.cos(angle), np.sin(angle)])
        balls.append(Ball(init_pos, init_vel))

    ts, true_trajs = simulate_trajectories(balls, t_end=t_end, dt=dt_sim)

    # Generate observations
    obs_times, observations = sample_noisy_observations(
        ts, true_trajs,
        obs_dt_mean=obs_dt_mean,
        obs_noise_std=obs_noise_std,
        dropout_prob=dropout_prob
    )

        # ---- DEBUG: Log which times are dropouts and which are observations ----
    print("==== OBSERVATION DROP DEBUG ====")
    for i, obs in enumerate(observations):
        t = obs_times[i]
        if obs is None:
            print(f"[DROPOUT] t={t:.3f} s: DROPPED frame.")
        else:
            print(f"[OBSERVED] t={t:.3f} s: {obs}")
    print("================================\n")

    # Initialize particles
    init_particles = []
    for _ in range(n_particles):
        particle = []
        for _ in range(n_balls):
            x = rng.uniform(*init_area[0])
            y = rng.uniform(*init_area[1])
            speed = rng.uniform(*init_speed_range)
            angle = rng.uniform(*init_angle_range)
            vx = speed * np.cos(angle)
            vy = speed * np.sin(angle)
            particle.extend([x, y, vx, vy])
        init_particles.append(particle)
    init_particles = np.array(init_particles)

    Q = Q_scale * np.eye(4 * n_balls)
    R = (obs_noise_std ** 2) * np.eye(2)

    #here is where the run.py code interacts with the particle filter. 

    pf = ParticleFilter(
        n_particles=n_particles,
        n_balls=n_balls,
        process_noise_cov=Q,
        obs_noise_cov=R,
        init_particles=init_particles
    )

    est_positions_over_time = []
    true_positions_over_time = []
    time_list = []
    dropout_indices = []

    last_time = 0.0
    for idx, obs in enumerate(observations):
        t_obs = obs_times[idx]
        dt = t_obs - last_time
        last_time = t_obs

        pf.predict(dt, ball_radius=ball_radius)
        pf.update(obs)
        pf.resample()

        if obs is None:
            dropout_indices.append(idx)

        positions = []
        for b in range(n_balls):
            positions.append(pf.particles[:, 4*b:4*b+2])
        particle_positions = np.vstack(positions).reshape(-1, 2)
        weights_tiled = np.tile(pf.weights, n_balls)

        centers_kmeans = kmeans_positions(
            particles=particle_positions,
            weights=weights_tiled,
            n_clusters=n_balls,
            n_resample=None,
            random_state=random_seed
        )
        est_positions_over_time.append(centers_kmeans)

        idx_ts = np.searchsorted(ts, t_obs)
        if idx_ts >= len(ts):
            idx_ts = len(ts) - 1
        true_frame = np.vstack([traj[idx_ts] for traj in true_trajs])
        true_positions_over_time.append(true_frame)
        time_list.append(t_obs)

    time_list = np.array(time_list)
    true_positions_over_time = np.stack(true_positions_over_time, axis=0)
    est_positions_over_time  = np.stack(est_positions_over_time, axis=0)

    # === XY Trajectory Plot ===
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw initial area boundary (box from (0,0) to (50,50))
    init_x_range = init_area[0]
    init_y_range = init_area[1]
    ax.plot([init_x_range[0], init_x_range[1], init_x_range[1], init_x_range[0], init_x_range[0]],
            [init_y_range[0], init_y_range[0], init_y_range[1], init_y_range[1], init_y_range[0]],
            linestyle='-', color='gray', linewidth=1.5, label='Init Area')

    valid_y = true_positions_over_time[:, :, 1]
    valid_y = valid_y[valid_y > 0]
    y_min = 0
    y_max = np.max(valid_y) + 10  # Add margin if desired


    # Shaded dropout regions (at true X position during dropout)
    for idx in dropout_indices:
        for b in range(n_balls):
            x = true_positions_over_time[idx, b, 0]
            ax.fill_betweenx(
                y=[0, y_max], x1=x - 0.5, x2=x + 0.5,
                color='purple', alpha=0.3
            )


    for b in range(n_balls):
        # True path (solid), only above ground
        mask_true = true_positions_over_time[:, b, 1] >= 0 # do not have the graph show below 0
        ax.plot(true_positions_over_time[mask_true, b, 0],
                true_positions_over_time[mask_true, b, 1],
                linestyle='-', linewidth=2, label=f"True Ball {b+1}")

        # Particle filter prediction (dashed), only above ground
        mask_true = true_positions_over_time[:, b, 1] >= 0 # do not have the graph show below 0
        mask_est = est_positions_over_time[:, b, 1] >= 0 
        ax.plot(est_positions_over_time[mask_est, b, 0],
                est_positions_over_time[mask_est, b, 1],
                linestyle='--', color=ax.lines[-1].get_color(), linewidth=1.7, label=f"PF Est. {b+1}")

        # Observed path (faint dots), only above ground
        mask_true = true_positions_over_time[:, b, 1] >= 0 # do not have the graph show below 0
        obs_x = [observations[i][b][0] if observations[i] is not None and observations[i][b][1] >= 0 else np.nan for i in range(len(time_list))] 
        obs_y = [observations[i][b][1] if observations[i] is not None and observations[i][b][1] >= 0 else np.nan for i in range(len(time_list))]
        ax.plot(obs_x, obs_y, marker='o', linestyle='None', alpha=0.4, markersize=6, label=f"Obs Ball {b+1}")

    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_title(f"XY Trajectories: True, Observed, and Particle Filter Estimated ({n_balls} Balls)")
    all_x = true_positions_over_time[:, :, 0]
    x_min = np.min(all_x[all_x > 0]) - 10
    x_max = np.max(all_x) + 10
    ax.set_xlim(x_min, x_max)

    ax.set_ylim(init_y_range[0] - 10, init_y_range[1] + 30)
    ax.legend(loc="best", ncol=2)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("particle_filter_xy_trajectory.png", dpi=150)
    plt.show()
    print("[INFO] Saved XY trajectory plot with dropouts as particle_filter_xy_trajectory.png")

    # === 2D Time-Series Plot (Y and X vs Time) ===
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for b in range(n_balls):
        # Y vs Time
        axs[0].plot(time_list, true_positions_over_time[:, b, 1],
                    linestyle='--', label=f"True Y Ball {b+1}")
        est_y = [est_positions_over_time[i, b, 1]
                 for i in range(len(time_list)) if observations[i] is not None]
        est_t = [time_list[i]
                 for i in range(len(time_list)) if observations[i] is not None]
        axs[0].plot(est_t, est_y, marker='o', linestyle='None', label=f"Est Y Ball {b+1}")
        obs_y = [observations[i][b][1] if observations[i] is not None else np.nan
                 for i in range(len(time_list))]
        axs[0].plot(time_list, obs_y, linestyle='-', alpha=0.4, label=f"Obs Y Ball {b+1}")

        # X vs Time
        axs[1].plot(time_list, true_positions_over_time[:, b, 0],
                    linestyle='--', label=f"True X Ball {b+1}")
        est_x = [est_positions_over_time[i, b, 0]
                 for i in range(len(time_list)) if observations[i] is not None]
        axs[1].plot(est_t, est_x, marker='o', linestyle='None', label=f"Est X Ball {b+1}")
        obs_x = [observations[i][b][0] if observations[i] is not None else np.nan
                 for i in range(len(time_list))]
        axs[1].plot(time_list, obs_x, linestyle='-', alpha=0.4, label=f"Obs X Ball {b+1}")

    axs[0].set_ylabel("Y Position (m)")
    axs[1].set_ylabel("X Position (m)")
    axs[1].set_xlabel("Time (s)")
    axs[0].legend(loc="upper right", ncol=3)
    axs[0].set_title(f"True, Estimated, and Observed Trajectories of {n_balls} Balls")
    plt.tight_layout()
    plt.savefig("particle_filter_time_series.png", dpi=150)
    print("[INFO] Saved time series plot as particle_filter_time_series.png")

    # === 3D Trajectory Plot ===
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    for b in range(n_balls):
        ax.plot(true_positions_over_time[:, b, 0],
                true_positions_over_time[:, b, 1],
                time_list,
                linestyle='--', label=f"True Ball {b+1}")
        est_x = [est_positions_over_time[i, b, 0]
                 for i in range(len(time_list)) if observations[i] is not None]
        est_y = [est_positions_over_time[i, b, 1]
                 for i in range(len(time_list)) if observations[i] is not None]
        est_t = [time_list[i]
                 for i in range(len(time_list)) if observations[i] is not None]
        ax.plot(est_x, est_y, est_t, marker='o', linestyle='None', label=f"Est Ball {b+1}")
        obs_x = [observations[i][b][0] if observations[i] is not None else np.nan
                 for i in range(len(time_list))]
        obs_y = [observations[i][b][1] if observations[i] is not None else np.nan
                 for i in range(len(time_list))]
        ax.plot(obs_x, obs_y, time_list, linestyle='-', alpha=0.4, label=f"Obs Ball {b+1}")

    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_zlabel("Time (s)")
    ax.set_title(f"3D Trajectory View (X, Y, Time)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("particle_filter_3d_trajectory.png", dpi=150)
    print("[INFO] Saved 3D trajectory plot as particle_filter_3d_trajectory.png")

if __name__ == "__main__":
    run_tracking_example()
