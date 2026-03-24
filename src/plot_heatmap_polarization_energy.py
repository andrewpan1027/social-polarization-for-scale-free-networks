"""
Plot heatmaps of global polarization degree Ψ and normalized Hamiltonian under different alpha, beta

This script generates two heatmaps:
1. Global polarization degree Ψ (global_psi)
2. Normalized Hamiltonian quantity (normalized_energy)

Reference: plot_heatmap_balance_triads.py structure
"""

import multiprocessing
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from tqdm import tqdm

# Import core model and network generator
from model_core import BinaryOpinionModel
from network_generator import (
    generate_poissonian_small_world,
    generate_stochastic_holme_kim,
)

# Path settings
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "Fig_heatmap_polarization_energy"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    """
    Experiment configuration parameters
    """

    N: int  # Number of nodes
    k_avg: float  # Average degree
    G_topics: int  # Opinion dimension
    burn_in_sweeps: int  # Evolution steps (MCS)

    # Network parameters
    epsilon: float = 0.175  # Small-world rewiring probability (small_world only)
    p_triangle: float = 0.65  # Triangle probability (scale_free only)
    network_type: str = "scale_free"  # "small_world" or "scale_free"

    # Experiment parameters
    n_trials: int = 10  # Number of trials per parameter point

    # Parameter space scan configuration (Alpha, Beta)
    alpha_start: float = 0.0
    alpha_end: float = 1.0
    alpha_steps: int = 41

    beta_start: float = 0.0
    beta_end: float = 5.0
    beta_steps: int = 101


def run_single_simulation(args):
    """
    Run single simulation
    args: (config, alpha, beta, seed)

    Returns
    -------
    tuple : (alpha, beta, global_psi, normalized_energy_mean)
    """
    config, alpha, beta, seed = args
    # Set random seed
    np.random.seed(seed)

    # 1. Generate network
    if config.network_type == "small_world":
        graph = generate_poissonian_small_world(
            N=config.N, k_avg=config.k_avg, epsilon=config.epsilon
        )
    else:  # scale_free
        graph = generate_stochastic_holme_kim(
            N=config.N, k_avg=config.k_avg, p_triangle=config.p_triangle
        )

    n_nodes = graph.number_of_nodes()
    nodes_all = np.arange(n_nodes)
    neighbors = [np.array(list(graph.neighbors(i))) for i in nodes_all]

    # 2. Initialize model
    model = BinaryOpinionModel(
        N=n_nodes, G_topics=config.G_topics, alpha=alpha, beta=beta
    )

    # 3. Evolution
    total_steps = config.burn_in_sweeps * n_nodes * config.G_topics
    for _ in range(total_steps):
        model.step(nodes_all, neighbors)

    # 4. Calculate global polarization
    global_psi, _ = model.calculate_polarization()

    # 5. Calculate normalized energy: each node's Hamiltonian divided by degree, then mean
    _, energies = model.calculate_hamilton(neighbors)
    degrees = np.array([len(nb) for nb in neighbors])
    # Avoid division by zero: isolated nodes have zero energy, skip them
    mask = degrees > 0
    normalized_energies = np.zeros_like(energies)
    normalized_energies[mask] = energies[mask] / degrees[mask]
    energies_mean = float(normalized_energies.mean())

    return (alpha, beta, global_psi, energies_mean)


def run_experiment_parallel(config: Config):
    """
    Run experiment in parallel, scanning parameter space

    Returns
    -------
    dict : contains alphas, betas, grid_psi, grid_energy
    """
    alphas = np.linspace(config.alpha_start, config.alpha_end, config.alpha_steps)
    betas = np.linspace(config.beta_start, config.beta_end, config.beta_steps)

    tasks = []
    base_seed = np.random.randint(1000000)
    seed_counter = 0

    for a in alphas:
        for b in betas:
            # Generate independent tasks for each trial
            for _ in range(config.n_trials):
                tasks.append((config, a, b, base_seed + seed_counter))
                seed_counter += 1

    n_cpu = max(1, cpu_count())
    print(
        f"Starting parallel computation, tasks: {len(tasks)}, CPU cores: {n_cpu} ({config.n_trials} trials per parameter point)"
    )

    results_map_sum_psi = {}  # (idx_a, idx_b) -> sum_psi
    results_map_sum_energy = {}  # (idx_a, idx_b) -> sum_energy
    results_map_count = {}  # (idx_a, idx_b) -> count

    with Pool(processes=n_cpu) as pool:
        for res in tqdm(
            pool.imap_unordered(run_single_simulation, tasks), total=len(tasks)
        ):
            a, b, psi, energy = res
            idx_a = (np.abs(alphas - a)).argmin()
            idx_b = (np.abs(betas - b)).argmin()

            key = (idx_a, idx_b)
            if key not in results_map_sum_psi:
                results_map_sum_psi[key] = 0.0
                results_map_sum_energy[key] = 0.0
                results_map_count[key] = 0

            results_map_sum_psi[key] += psi
            results_map_sum_energy[key] += energy
            results_map_count[key] += 1

    # Organize results into matrices
    grid_psi = np.zeros((len(alphas), len(betas)))
    grid_energy = np.zeros((len(alphas), len(betas)))

    for idx_a in range(len(alphas)):
        for idx_b in range(len(betas)):
            if (idx_a, idx_b) in results_map_sum_psi:
                # Calculate average
                count = results_map_count[(idx_a, idx_b)]
                grid_psi[idx_a, idx_b] = results_map_sum_psi[(idx_a, idx_b)] / count
                grid_energy[idx_a, idx_b] = (
                    results_map_sum_energy[(idx_a, idx_b)] / count
                )

    return {
        "alphas": alphas,
        "betas": betas,
        "grid_psi": grid_psi,
        "grid_energy": grid_energy,
    }


def get_experiment_data(config: Config, replace: bool = False):
    """
    Get experiment data: load if file exists and not forced to replace; otherwise recompute.

    Parameters
    ----------
    config : Config
        Experiment configuration
    replace : bool
        Whether to force recomputation

    Returns
    -------
    np.lib.npyio.NpzFile : npz file containing experiment data
    """
    # Construct filename containing key parameters to distinguish different experiments
    save_fn = (
        f"polarization_energy_{config.network_type}_N{config.N}_k{int(config.k_avg)}_"
        f"G{config.G_topics}_{config.alpha_steps}x{config.beta_steps}_"
        f"{config.n_trials}trials.npz"
    )
    file_path = RESULTS_DIR / save_fn

    if file_path.exists() and not replace:
        print(f"Data file exists, loading directly: {file_path}")
        return np.load(file_path)

    print(f"Starting new computation: {save_fn}")
    data = run_experiment_parallel(config)

    np.savez(
        file_path,
        alphas=data["alphas"],
        betas=data["betas"],
        grid_psi=data["grid_psi"],
        grid_energy=data["grid_energy"],
    )
    print(f"Data saved to: {file_path}")
    return np.load(file_path)


def plot_polarization_heatmap(data, config: Config):
    """
    Plot global polarization degree Ψ heatmap

    Parameters
    ----------
    data : np.lib.npyio.NpzFile
        Contains experiment data
    config : Config
        Experiment configuration
    """
    alphas = data["alphas"]
    betas = data["betas"]
    grid_psi = data["grid_psi"]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
        }
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    extent = [betas[0], betas[-1], alphas[0], alphas[-1]]

    # Use TwoSlopeNorm with 0.11 as boundary
    # Ensure vmax > vcenter
    vmax = max(grid_psi.max(), 0.1101)
    norm = TwoSlopeNorm(vmin=0, vcenter=0.11, vmax=vmax)

    im = ax.imshow(
        grid_psi,
        extent=extent,
        origin="lower",
        cmap="RdYlBu_r",
        norm=norm,
        aspect="auto",
    )
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\alpha$")
    # ax.set_title(r"Global Polarization $\Psi$")

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, label=r"$\Psi$")

    plt.tight_layout()

    # Construct save filename with information
    save_fn = f"polarization_heatmap_{config.network_type}_N{config.N}_k{int(config.k_avg)}.png"
    save_path = RESULTS_DIR / save_fn

    plt.savefig(save_path, dpi=300)
    print(f"Global polarization heatmap saved to: {save_path}")
    plt.close()


def plot_energy_heatmap(data, config: Config):
    """
    Plot normalized Hamiltonian heatmap

    Parameters
    ----------
    data : np.lib.npyio.NpzFile
        Contains experiment data
    config : Config
        Experiment configuration
    """
    alphas = data["alphas"]
    betas = data["betas"]
    grid_energy = data["grid_energy"]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
        }
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    extent = [betas[0], betas[-1], alphas[0], alphas[-1]]

    im = ax.imshow(
        grid_energy,
        extent=extent,
        origin="lower",
        cmap="viridis_r",
        aspect="auto",
    )
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\alpha$")
    # ax.set_title(r"Normalized Hamiltonian")

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, label=r"$\langle H \rangle$")

    plt.tight_layout()

    # Construct save filename with information
    save_fn = (
        f"energy_heatmap_{config.network_type}_N{config.N}_k{int(config.k_avg)}.png"
    )
    save_path = RESULTS_DIR / save_fn

    plt.savefig(save_path, dpi=150)
    print(f"Normalized Hamiltonian heatmap saved to: {save_path}")
    plt.close()


if __name__ == "__main__":
    # Configure experiment
    config = Config(
        network_type="scale_free",  # stochastic holme kim
        N=500,
        k_avg=12,
        G_topics=9,
        burn_in_sweeps=35,
        p_triangle=0.65,
        epsilon=0.175,
        # Production parameters
        alpha_steps=41,
        beta_steps=101,
        n_trials=20,  # Number of trials
    )

    # Get data (auto cache)
    data = get_experiment_data(config, replace=False)

    # Plot two heatmaps
    plot_polarization_heatmap(data, config)
    plot_energy_heatmap(data, config)
