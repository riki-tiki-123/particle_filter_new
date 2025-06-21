# clustering.py

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity

def kmeans_positions(particles: np.ndarray,
                     weights: np.ndarray,
                     n_clusters: int,
                     n_resample: int = None,
                     random_state: int = 0) -> np.ndarray:
    """
    Given a set of weighted particle (x, y) positions (possibly from all balls in all particles),
    use weighted K-Means clustering to extract the n_clusters most likely position centers.

    Args:
        particles   : (N, 2) array of [x, y] positions (can be stacked from all balls and all particles)
        weights     : (N,) array of normalized importance weights
        n_clusters  : How many ball positions (clusters) to extract
        n_resample  : How many points to resample for K-Means (default=N)
        random_state: Seed for reproducibility

    Returns:
        centers: (n_clusters, 2) array of estimated [x, y] positions (the cluster centers)
    
    Method:
        Since standard KMeans does not accept sample weights directly (for most sklearn versions),
        we resample particles according to their weights to approximate a weighted dataset.
        Then run KMeans to find the most likely cluster centers (blobs), which correspond to the
        estimated positions of the balls.

        This is a common and effective way to extract multi-modal (multi-target) state estimates
        from a particle filter when the targets are indistinguishable.
    """
    N = particles.shape[0]
    if n_resample is None:
        n_resample = N

    weights = weights / np.sum(weights)  # Ensure weights sum to 1


    # Resample indices with replacement according to particle weights
    idx = np.random.RandomState(random_state).choice(
        N, size=n_resample, p=weights
    )
    sampled = particles[idx, :2]  # take only x, y columns

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(sampled)
    return kmeans.cluster_centers_


def kde_peak_positions(particles: np.ndarray,
                       weights: np.ndarray,
                       n_peaks: int,
                       bandwidth: float = 1.0,
                       grid_size: int = 100) -> np.ndarray:
    """
    Use a 2D weighted kernel density estimate (KDE) over (x, y) positions and find the top-n_peaks
    (i.e., density maxima) as alternative cluster centers.

    Args:
        particles  : (N, 2) array of [x, y] positions
        weights    : (N,) array of normalized importance weights
        n_peaks    : How many density peaks to extract (typically n_balls)
        bandwidth  : KDE bandwidth (smoothing parameter)
        grid_size  : Resolution of the 2D grid for searching peaks

    Returns:
        peaks: (n_peaks, 2) array of estimated [x, y] positions at the top density peaks
    
    Method:
        Fit a weighted KDE on the particle cloud, evaluate density over a grid,
        then return the coordinates of the highest local maxima.

        This approach is more robust if clusters are non-Gaussian, but is slower for large grids.
        Useful for exam/portfolio: you can show both KMeans and KDE clustering as two alternative
        approaches to extract multi-target positions from a multi-modal distribution.
    """
    xy = particles[:, 0:2]
    kde = KernelDensity(bandwidth=bandwidth)
    kde.fit(xy, sample_weight=weights)

    # Make a grid over the spatial extent of the particles
    x_min, y_min = xy.min(axis=0) - 1.0
    x_max, y_max = xy.max(axis=0) + 1.0
    xs = np.linspace(x_min, x_max, grid_size)
    ys = np.linspace(y_min, y_max, grid_size)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.vstack([xx.ravel(), yy.ravel()]).T

    # Evaluate log-density on grid
    logdens = kde.score_samples(grid)
    idx = np.argsort(logdens)[-n_peaks:]  # indices of highest peaks
    peaks = grid[idx]
    return peaks

# ---- END FILE ----
