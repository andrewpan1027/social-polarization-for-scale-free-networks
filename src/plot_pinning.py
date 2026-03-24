import multiprocessing
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import core model and network generator
from model_core import BinaryOpinionModel
from network_generator import (
    generate_poissonian_small_world,
    generate_stochastic_holme_kim,
)

# Path settings
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "Fig_polarization_pinning"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    """Pinning experiment configuration parameters"""

    N: int  # Total number of nodes
    k_avg: float  # Average degree
    network_type: str  # Network type: 'small_world' or 'scale_free'
    epsilon: float  # Rewiring probability (small-world network)
    p_triangle: float  # Triangle probability (scale-free network)
    G_topics: int  # Topic dimension
    alpha: float  # Friend/enemy weight parameter
    beta: float  # Social pressure sensitivity (inverse temperature)
    burn_in_sweeps: int  # Evolution duration (MCS)
    n_trials: int  # Number of trials per sampling point
    pinned_fracs: np.ndarray  # Array of pinning fractions


def calculate_polarization_subset(model: BinaryOpinionModel, indices: np.ndarray):
    """
    Calculate polarization within specific node subset

    Parameters
    ----------
    model : BinaryOpinionModel
        Model instance
    indices : np.ndarray
        Index array of node subset

    Returns
    -------
    float
        Polarization within subset (variance)
    """
    if len(indices) <= 1:
        return 0.0

    # Get subset opinion matrix (n_subset, G)
    sub_opinions = model.opinions[indices]

    # Calculate alignment matrix (n_subset, n_subset)
    alignment_matrix = sub_opinions @ sub_opinions.T / model.G_topics

    # Get upper triangle (excluding diagonal)
    n_sub = len(indices)
    tri_indices = np.triu_indices(n_sub, k=1)
    vals = alignment_matrix[tri_indices]

    # Calculate variance
    return float(np.var(vals))


def run_single_trial(params: Config, f_val: float, show_progress: bool = False):
    """
    Run single complete pinning experiment (including both targeted and random strategies)

    Parameters
    ----------
    params : Config
        Experiment configuration
    f_val : float
        Pinning fraction
    show_progress : bool
        Whether to show progress bar

    Returns
    -------
    tuple[float, float]
        (psi_targeted, psi_random)
    """
    # 1. Generate network
    if params.network_type == "small_world":
        graph = generate_poissonian_small_world(
            N=params.N, k_avg=params.k_avg, epsilon=params.epsilon
        )
    else:
        graph = generate_stochastic_holme_kim(
            N=params.N, k_avg=params.k_avg, p_triangle=params.p_triangle
        )

    n_nodes = graph.number_of_nodes()
    nodes_all = np.arange(n_nodes)
    adj = [np.array(list(graph.neighbors(i))) for i in nodes_all]
    degrees = np.array([len(a) for a in adj])

    # Actual number of pinned nodes
    n_pin = int(np.round(f_val * n_nodes))

    # --- Strategy A: Targeted pinning (high-degree hub nodes) ---
    hubs = np.argsort(degrees)[-n_pin:] if n_pin > 0 else np.array([], dtype=int)

    model_t = BinaryOpinionModel(
        N=n_nodes,
        G_topics=params.G_topics,
        alpha=params.alpha,
        beta=params.beta,
        fixed_nodes=hubs,
    )

    total_steps = params.burn_in_sweeps * n_nodes * params.G_topics
    for _ in range(total_steps):
        model_t.step(nodes_all, adj)

    unpinned_t = nodes_all[~model_t.is_fixed]
    psi_target = calculate_polarization_subset(model_t, unpinned_t)

    # --- Strategy B: Random pinning ---
    random_nodes = (
        np.random.choice(n_nodes, size=n_pin, replace=False)
        if n_pin > 0
        else np.array([], dtype=int)
    )

    model_r = BinaryOpinionModel(
        N=n_nodes,
        G_topics=params.G_topics,
        alpha=params.alpha,
        beta=params.beta,
        fixed_nodes=random_nodes,
    )

    for _ in range(total_steps):
        model_r.step(nodes_all, adj)

    unpinned_r = nodes_all[~model_r.is_fixed]
    psi_random = calculate_polarization_subset(model_r, unpinned_r)

    return psi_target, psi_random


def run_single_trial_task(args):
    """Multiprocess execution helper function"""
    params, f_val, seed = args
    np.random.seed(seed)
    return f_val, run_single_trial(params, f_val, show_progress=False)


def run_experiment_parallel(params: Config):
    """Run multiple pinning experiments in parallel"""
    base_seed = np.random.randint(0, 1000000)
    tasks = []
    for f in params.pinned_fracs:
        for i in range(params.n_trials):
            tasks.append((params, f, base_seed + len(tasks)))

    n_cpu = max(1, cpu_count())
    pool_results = []

    print(f"Running multiprocess pinning experiments (CPU cores: {n_cpu})...")
    with Pool(processes=n_cpu) as pool:
        for res in tqdm(
            pool.imap_unordered(run_single_trial_task, tasks),
            total=len(tasks),
            desc="Total Progress",
        ):
            pool_results.append(res)

    # Aggregate results
    f_map_t = {f: [] for f in params.pinned_fracs}
    f_map_r = {f: [] for f in params.pinned_fracs}

    for f_val, (psi_t, psi_r) in pool_results:
        f_map_t[f_val].append(psi_t)
        f_map_r[f_val].append(psi_r)

    means_t = np.array([np.mean(f_map_t[f]) for f in params.pinned_fracs])
    stds_t = np.array([np.std(f_map_t[f]) for f in params.pinned_fracs])
    means_r = np.array([np.mean(f_map_r[f]) for f in params.pinned_fracs])
    stds_r = np.array([np.std(f_map_r[f]) for f in params.pinned_fracs])

    return means_t, stds_t, means_r, stds_r


def get_experiment_data(params: Config, replace: bool = False):
    """
    Get experiment data: load if file exists and replace=False; otherwise recompute and save.

    Parameters
    ----------
    params : Config
        Experiment configuration
    replace : bool
        Whether to force recomputation
    """
    save_fn = f"{params.network_type}_{params.N}nodes_{int(params.k_avg)}degrees_{params.n_trials}trials_{params.burn_in_sweeps}sweeps.npz"
    file_path = RESULTS_DIR / save_fn

    if file_path.exists() and not replace:
        print(f"Data file exists, loading directly: {save_fn}")
        return np.load(file_path)

    print(f"Starting experiment computation: {save_fn}")
    means_t, stds_t, means_r, stds_r = run_experiment_parallel(params)

    # Save results
    np.savez(
        file_path,
        pinned_fracs=params.pinned_fracs,
        means_targeted=means_t,
        stds_targeted=stds_t,
        means_random=means_r,
        stds_random=stds_r,
    )
    print(f"Experiment completed, results saved to: {file_path}")

    return np.load(file_path)


def plot_pinning_results(data, params: Config):
    """
    Plot pinning experiment results, strictly matching reference image style

    Parameters
    ----------
    data : dict-like
        Experiment data
    params : Config
        Experiment configuration
    """
    # Set plotting style
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 12,
            "axes.labelpad": 12,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "figure.titlesize": 18,
            "figure.titleweight": "bold",
            "axes.grid": True,
            "grid.alpha": 0.2,
            "grid.linestyle": "--",
            "grid.color": "#cccccc",
        }
    )

    f_values = data["pinned_fracs"]
    m_t = data["means_targeted"]
    m_r = data["means_random"]

    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)

    # 1. Plot core curves
    # Random pinning (Control): blue, square marker, hollow
    ax.plot(
        f_values,
        m_r,
        "s-",
        color="#1F77B4",
        label="Control (Random selection)",
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=1.5,
        linewidth=1.5,
        zorder=3,
    )

    # Targeted pinning: red, circle marker, solid
    ax.plot(
        f_values,
        m_t,
        "o-",
        color="#D62728",
        label="Targeted (High-degree hubs)",
        markersize=8,
        linewidth=2,
        zorder=4,
    )

    # 2. Plot fill area between two lines (polarization dividend)
    ax.fill_between(f_values, m_t, m_r, color="#D62728", alpha=0.08, zorder=2)

    # 3. Plot baseline (1/G) and its background
    baseline = 1.0 / params.G_topics
    # Add baseline background band
    ax.fill_between(
        [f_values[0], f_values[-1]],
        baseline - 0.02,
        baseline + 0.02,
        color="#bdc3c7",
        alpha=0.15,
        zorder=1,
    )
    ax.axhline(
        y=baseline, color="#7f8c8d", linestyle="--", linewidth=2, alpha=0.6, zorder=2
    )

    # Baseline text label
    # ax.text(
    #     0.005,
    #     baseline + 0.03,
    #     "Null model baseline (1/G)",
    #     color="#7f8c8d",
    #     fontsize=14,
    #     ha="left",
    #     va="bottom",
    # )

    # 4. Axis and optimization
    ax.set_xlabel("Pinned fraction ($f$)", fontsize=16)
    ax.set_ylabel("Polarization of unpinned nodes ($\Psi$)", fontsize=16)

    # Set tick range matching reference figure
    ax.set_xlim(-0.002, 0.105)
    ax.set_ylim(0.08, 0.52)

    # Remove top and right border
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)

    # Legend optimization
    ax.legend(frameon=False, loc="upper right", fontsize=12)

    # Grid line optimization (show horizontal lines only)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    save_path = RESULTS_DIR / f"pinning_{params.network_type}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Image saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    # Use higher density parameter configuration for smoother curves
    cfg = Config(
        N=500,
        k_avg=12,
        network_type="scale_free",
        epsilon=0.175,
        p_triangle=0.65,
        G_topics=9,
        alpha=0.5,
        beta=2.7,
        burn_in_sweeps=60,  # Slightly reduce sweeps for faster computation, 60 is sufficient for convergence
        n_trials=20,  # 20 trials achieves good averaging
        pinned_fracs=np.linspace(0.0, 0.1, 51),  # Increase sampling point density
    )

    # 1. Get data (compute or load from file)
    res_data = get_experiment_data(cfg, replace=False)

    # 2. Plot
    plot_pinning_results(res_data, cfg)
