
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import imageio.v2 as imageio
from joblib import Parallel, delayed

def generate_gbm(N, T, mu, sigma, S0, r):
    dt = 1.0 / T
    Z = np.random.standard_normal((T, N))
    daily_returns = np.exp((mu - r * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
    paths = np.vstack([np.ones(N) * S0, S0 * daily_returns.cumprod(axis=0)])
    return paths

def plot_gbm(paths, t, output_dir, T, min_log_return, max_log_return, max_density):
    N = paths.shape[1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6), gridspec_kw={'width_ratios': [3, 1]})

    # Plot paths
    ax1.plot(paths[:t + 1, :], lw=0.5)
    ax1.set_title(f'Geometric Brownian Motion\n(t={t})')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Price')
    ax1.set_xlim(0, T)
    ax1.set_ylim(0, paths.max())

    # Plot log-returns distribution
    if t > 0:
        total_log_returns = np.log(paths[t, :] / paths[0, :])
        ax2.hist(total_log_returns, bins=int(np.sqrt(total_log_returns.size)), orientation='horizontal', density=True, color = "green")

    ax2.set_title('Log-Returns\nDistribution')
    ax2.set_xlabel('Density')
    ax2.set_ylabel('Log-Returns')
    ax2_ylim = max(abs(min_log_return), abs(max_log_return))
    ax2.set_ylim(-ax2_ylim, ax2_ylim)
    ax2.set_xlim(0, max_density * 1.1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'gbm_{t:03d}.png'), transparent=True)
    plt.close(fig)

if __name__ == '__main__':
    # Parameters
    N = 777  # Number of paths
    T = 252  # Number of time steps (e.g., trading days in a year)
    mu = 0  # Drift
    sigma = 20/99  # Volatility
    S0 = 100  # Initial stock price
    r = 0 # Risk-free rate

    # Create output directory
    output_dir = 'gbm_plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate GBM paths
    paths = generate_gbm(N, T, mu, sigma, S0, r)

    # Calculate log-returns and their limits for fixed axes
    log_returns = np.log(paths[1:, :] / S0)


    min_log_return = log_returns.min()
    max_log_return = log_returns.max()
    density, bins = np.histogram(log_returns.flatten(), bins=int(np.sqrt(log_returns.size)), density=True)
    max_density = density.max()

    # Generate plots for each time step
    print("Rendering frames in parallel...")
    max_jobs = os.cpu_count() // 4
    max_jobs = max(1, max_jobs)
    Parallel(n_jobs=max_jobs, prefer="processes", verbose=5)(
        delayed(plot_gbm)(paths, t, output_dir, T, min_log_return, max_log_return, max_density)
        for t in range(T + 1)
    )

    # Create GIF from plots
    images = []
    print("Collating GIF")
    for t in range(T + 1):
        images.append(imageio.imread(os.path.join(output_dir, f'gbm_{t:03d}.png')))
    imageio.mimsave('gbm_evolution.gif', images, fps=10, disposal=2, loop=0)

    print(f"Created gbm_evolution.gif")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Compress the GIF
    try:
        from pygifsicle import optimize
        print("Compressing GIF...")
        optimize('gbm_evolution.gif')
        print("GIF compressed successfully.")
    except Exception as e:
        print(f"Could not compress GIF: {e}")
        print("Please make sure gifsicle is installed and in your PATH.")
