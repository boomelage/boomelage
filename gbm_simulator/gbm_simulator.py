
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import os
import shutil
import imageio.v2 as imageio
from joblib import Parallel, delayed

# CRT terminal palette: phosphor colours for the paths, green for the "screen" chrome
PHOSPHOR_GREEN = '#33ff66'
PHOSPHOR_DIM = '#1a8033'
PHOSPHORS = ['#33ff66', '#ffb000', '#00e5ff', '#ff4fd8', '#b6ff00']

# Applied inside plot_gbm so it also takes effect in the joblib worker processes
CRT_STYLE = {
    'font.family': 'serif',
    'font.serif': ['cmr10'],
    'mathtext.fontset': 'cm',
    'axes.unicode_minus': False,  # cmr10 has no Unicode minus glyph
    'axes.formatter.use_mathtext': True,
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.linewidth': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'axes.grid': True,
    'grid.color': PHOSPHOR_DIM,
    'grid.linestyle': ':',
    'grid.linewidth': 0.6,
    'hatch.color': PHOSPHOR_GREEN,
    'hatch.linewidth': 0.6,
}

def generate_gbm(N, T, mu, sigma, S0, r):
    dt = 1.0 / T
    Z = np.random.standard_normal((T, N))
    daily_returns = np.exp((mu - r * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
    paths = np.vstack([np.ones(N) * S0, S0 * daily_returns.cumprod(axis=0)])
    return paths

def plot_gbm(paths, t, output_dir, T, min_log_return, max_log_return, max_density):
    with plt.rc_context(CRT_STYLE):
        _plot_gbm(paths, t, output_dir, T, min_log_return, max_log_return, max_density)

def _plot_gbm(paths, t, output_dir, T, min_log_return, max_log_return, max_density):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6), gridspec_kw={'width_ratios': [3, 1]})
    # Dim phosphor halo around a black outline for the axes frame.
    # Kept opaque: GIF transparency is 1-bit, so soft alpha glows turn into blobs.
    glow_effect = [
        pe.Stroke(linewidth=3.5, foreground=PHOSPHOR_DIM),
        pe.Stroke(linewidth=2, foreground='black'),
        pe.Normal(),
    ]
    # Thin dark edge on text to hide the jaggies from the GIF's 1-bit transparency
    text_outline = [pe.Stroke(linewidth=1.2, foreground='black'), pe.Normal()]

    # Plot paths
    ax1.set_prop_cycle(color=PHOSPHORS)
    ax1.plot(paths[:t + 1, :], lw=0.5)
    # Counter is a separate text on a fixed baseline: inside a two-line title, mathtext line
    # heights vary with the digits and nudge the title up and down between frames
    ax1.set_title('GEOMETRIC BROWNIAN MOTION', pad=18)
    counter = ax1.text(0.5, 1.012, f'$t = {t}$', transform=ax1.transAxes, ha='center', va='baseline',
                       color=PHOSPHOR_GREEN, fontsize=12)
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Price')
    ax1.set_xlim(0, T)
    ax1.set_ylim(0, 200)

    # Plot log-returns distribution
    if t > 0:
        total_log_returns = np.log(paths[t, :] / paths[0, :])
        ax2.hist(total_log_returns, bins=int(np.sqrt(total_log_returns.size)), orientation='horizontal', density=True,
                 facecolor='none', edgecolor=PHOSPHOR_GREEN, hatch='////', linewidth=0.8)

    ax2.set_title('LOG-RETURNS\nDISTRIBUTION')
    ax2.set_xlabel('Density')
    ax2.set_ylabel('Log-Return')
    ax2_ylim = max(abs(min_log_return), abs(max_log_return))
    ax2.set_ylim(-ax2_ylim, ax2_ylim)
    ax2.set_xlim(0, max_density * 1.1)

    for ax in (ax1, ax2):
        ax.tick_params(which='both', colors=PHOSPHOR_GREEN)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_path_effects(text_outline)
        for spine in ax.spines.values():
            spine.set_color(PHOSPHOR_GREEN)
            spine.set_path_effects(glow_effect)
        for text in (ax.title, ax.xaxis.label, ax.yaxis.label):
            text.set_color(PHOSPHOR_GREEN)
            text.set_path_effects(text_outline)
    counter.set_path_effects(text_outline)

    # Fixed margins rather than tight_layout, which re-fits per frame and makes the axes jitter
    fig.subplots_adjust(left=0.081, right=0.98, bottom=0.093, top=0.905, wspace=0.26)
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
