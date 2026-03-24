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
RESULTS_DIR = BASE_DIR / "results" / "Fig_individual_polarization_energy"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def calculate_energy_density_all(
    model: BinaryOpinionModel, neighbors: list[np.ndarray]
):
    """
    Calculate normalized energy for all nodes (social pressure, energy divided by degree k_i)

    Parameters
    ----------
    model : BinaryOpinionModel
        Model instance
    neighbors : list[np.ndarray]
        Adjacency list structure

    Returns
    -------
    np.ndarray
        Array of shape (N,) containing energy density for each node
    """
    N = model.N
    energy_densities = np.zeros(N)

    for i in range(N):
        nbs = neighbors[i]
        k_i = len(nbs)
        if k_i == 0:
            continue

        # Calculate local energy
        E_i = model.calculate_individual_energy(model.opinions[i], model.opinions[nbs])
        # Normalized energy (social pressure) = local energy / degree k_i
        energy_densities[i] = E_i / k_i

    return energy_densities


def aggregated_results(results, num_bins=31):
    """
    Aggregate multiple trial data (log-binning by degree)

    Parameters
    ----------
    results : list
        List where each element is (degrees, individual_polarization, individual_energy)
    num_bins : int
        Number of bins

    Returns
    -------
    tuple
        (bin_centers, bin_means_psi, bin_stds_psi, bin_means_energy, bin_stds_energy)
    """
    all_degrees = []
    all_polarizations = []
    all_energies = []

    # Collect data from all trials
    for degrees, individual_polarization, individual_energy in results:
        all_degrees.extend(degrees)
        all_polarizations.extend(individual_polarization)
        all_energies.extend(individual_energy)

    all_degrees = np.array(all_degrees)
    all_polarizations = np.array(all_polarizations)
    all_energies = np.array(all_energies)

    # Filter invalid values
    valid_mask = all_degrees > 0
    all_degrees = all_degrees[valid_mask]
    all_polarizations = all_polarizations[valid_mask]
    all_energies = all_energies[valid_mask]

    k_min, k_max = all_degrees.min(), all_degrees.max()
    if np.isclose(k_min, k_max):
        k_max = k_min * 2.0

    # Log-binning
    bins = np.geomspace(k_min, k_max, num_bins)
    bin_centers = np.sqrt(bins[:-1] * bins[1:])

    bin_means_psi = []
    bin_stds_psi = []
    bin_means_energy = []
    bin_stds_energy = []

    for i in range(len(bins) - 1):
        mask = (all_degrees >= bins[i]) & (all_degrees < bins[i + 1])
        if np.any(mask):
            bin_means_psi.append(np.mean(all_polarizations[mask]))
            bin_stds_psi.append(np.std(all_polarizations[mask]))
            bin_means_energy.append(np.mean(all_energies[mask]))
            bin_stds_energy.append(np.std(all_energies[mask]))
        else:
            bin_means_psi.append(np.nan)
            bin_stds_psi.append(np.nan)
            bin_means_energy.append(np.nan)
            bin_stds_energy.append(np.nan)

    # Return means and standard deviations
    return (
        np.array(bin_centers),
        np.array(bin_means_psi),
        np.array(bin_stds_psi),
        np.array(bin_means_energy),
        np.array(bin_stds_energy),
    )


@dataclass
class Config:
    """Pinning experiment configuration parameters"""

    N: int  # Total number of nodes
    k_avg: float  # Average degree
    network_type: str  # Network type
    epsilon: float  # Rewiring probability (small-world network)
    p_triangle: float  # Triangle probability (scale-free network)
    G_topics: int  # Issue dimension
    alpha: float  # Friend/enemy weight parameter
    beta: float  # Social pressure sensitivity (inverse temperature)
    burn_in_sweeps: int  # Evolution duration (MCS)
    n_trials: int  # Number of trials


def run_single_trial(params: Config, show_progress: bool = True):
    """
    Run single complete experiment, calculating polarization and energy for each node

    Parameters
    ----------
    params : Config
        Experiment parameters
    show_progress : bool
        Whether to show progress bar

    Returns
    -------
    tuple
        (degrees, individual_psi, individual_energy)
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

    # 2. Initialize model
    model = BinaryOpinionModel(
        N=n_nodes,
        G_topics=params.G_topics,
        alpha=params.alpha,
        beta=params.beta,
    )

    # 3. Run evolution (1 sweep = N * G updates)
    total_steps = params.burn_in_sweeps * n_nodes * params.G_topics
    step_range = range(total_steps)
    if show_progress:
        step_range = tqdm(step_range, desc="Evolution", leave=False)

    for _ in step_range:
        model.step(nodes_all, adj)

    # 4. Calculate metrics
    _, individual_psi = model.calculate_polarization()
    individual_energy = calculate_energy_density_all(model, adj)

    return degrees, individual_psi, individual_energy


def run_single_trial_task(args):
    """Multiprocess execution helper function"""
    params, seed = args
    np.random.seed(seed)
    return run_single_trial(params, show_progress=False)


def run_experiment_parallel(params: Config):
    """Run multiple trials in parallel"""
    base_seed = np.random.randint(0, 1000000)
    tasks = [(params, base_seed + i) for i in range(params.n_trials)]

    n_cpu = max(1, cpu_count())
    results = []

    print(f"Running multiprocess experiment (CPU cores: {n_cpu})...")
    with Pool(processes=n_cpu) as pool:
        for res in tqdm(
            pool.imap_unordered(run_single_trial_task, tasks),
            total=len(tasks),
            desc="Trials",
        ):
            results.append(res)

    return results


def get_experiment_data(params: Config, replace: bool = False):
    """
    Get experiment data: load if file exists and replace=False; otherwise recompute and save.

    Parameters
    ----------
    params : Config
        Experiment configuration parameters
    replace : bool
        Whether to force recomputation and overwrite existing file

    Returns
    -------
    dict-like
        Dictionary object containing experiment results (np.lib.npyio.NpzFile)
    """
    save_fn = f"{params.network_type}_{params.N}nodes_{int(params.k_avg)}degrees_{params.n_trials}trials_{params.burn_in_sweeps}sweeps.npz"
    file_path = RESULTS_DIR / save_fn

    if file_path.exists() and not replace:
        print(f"Data file exists, loading directly: {save_fn}")
        return np.load(file_path)

    print(f"Starting experiment computation: {save_fn}")
    raw_res = run_experiment_parallel(params)
    res_agg = aggregated_results(raw_res)

    # Save results
    np.savez(
        file_path,
        bin_centers=res_agg[0],
        bin_means=res_agg[1],
        bin_stds=res_agg[2],
        bin_means_energy=res_agg[3],
        bin_stds_energy=res_agg[4],
    )
    print(f"Experiment completed, results saved to: {file_path}")

    return np.load(file_path)


def plot_combined_results(configs):
    """
    Plot polarization and energy dual-panel figure based on uploaded image style

    Parameters
    ----------
    configs : list
        Configuration list containing data loading results and styles
    """
    # Set plotting style
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 14,
            "axes.labelpad": 10,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "figure.titlesize": 18,
            "figure.titleweight": "bold",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "grid.color": "#cccccc",
            "grid.linestyle": "-",
        }
    )

    # Create dual-panel figure
    fig = plt.figure(figsize=(18, 6), dpi=150)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.25)

    # ===============================
    # Panel A: Polarization Ψ(i)
    # ===============================
    ax1 = fig.add_subplot(gs[0])
    for cfg in configs:
        res = cfg["data"]
        centers = res["bin_centers"]
        means = res["bin_means"]
        stds = res["bin_stds"]

        # Sort by x-axis coordinates
        idx = np.argsort(centers)
        centers, means, stds = centers[idx], means[idx], stds[idx]

        # Remove NaN values
        valid = ~np.isnan(means)
        centers, means, stds = centers[valid], means[valid], stds[valid]

        if "6" in cfg["label"]:
            # Remove several points at end with large fluctuations (visual smoothing only, consistent with uploaded figure)
            centers, means, stds = centers[:-2], means[:-2], stds[:-2]
        else:
            # Similarly remove end points based on uploaded figure style
            centers, means, stds = centers[:-1], means[:-1], stds[:-1]

        # Extract degree value and construct precise legend label
        import re

        k_val = re.search(r"\d+", cfg["label"]).group()
        display_label = rf"Scale-free network ⟨$k$⟩ = {k_val}"

        ax1.fill_between(
            centers,
            means - stds,
            means + stds,
            color=cfg["color"],
            alpha=0.15,
        )
        ax1.plot(
            centers,
            means,
            color=cfg["color"],
            linewidth=2.5,
            marker=cfg["marker"],
            markersize=5,
            label=display_label,
        )

    ax1.set_xscale("log")
    ax1.set_xlabel("Degree $k$")
    ax1.set_ylabel(r"Polarization $\Psi(i)$")
    ax1.legend(loc="lower right", frameon=False, fontsize=13)
    # Fine-tune Y-axis ticks
    ax1.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    ax1.grid(True, which="major", alpha=0.3)
    ax1.grid(True, which="minor", alpha=0.1)

    # ===============================
    # Panel B: Social pressure H^(i)
    # ===============================
    ax2 = fig.add_subplot(gs[1])
    for cfg in configs:
        res = cfg["data"]
        centers = res["bin_centers"]
        means = res["bin_means_energy"]
        stds = res["bin_stds_energy"]

        # Sort by x-axis coordinates
        idx = np.argsort(centers)
        centers, means, stds = centers[idx], means[idx], stds[idx]

        # Remove NaN values
        valid = ~np.isnan(means)
        centers, means, stds = centers[valid], means[valid], stds[valid]

        if "6" in cfg["label"]:
            centers, means, stds = centers[:-2], means[:-2], stds[:-2]
        else:
            centers, means, stds = centers[:-1], means[:-1], stds[:-1]

        # Extract degree value and construct precise legend label
        import re

        k_val = re.search(r"\d+", cfg["label"]).group()
        display_label = rf"Scale-free network ⟨$k$⟩ = {k_val}"

        ax2.fill_between(
            centers,
            means - stds,
            means + stds,
            alpha=0.15,
            color=cfg["color"],
        )
        ax2.plot(
            centers,
            means,
            lw=2.5,
            color=cfg["color"],
            marker=cfg["marker"],
            markersize=5,
            label=display_label,
        )

    ax2.set_xscale("log")
    ax2.set_xlabel("Degree $k$")
    ax2.set_ylabel(r"Social Pressure $H^{(i)}$")
    ax2.legend(loc="upper right", frameon=False, fontsize=13)
    # Fine-tune Y-axis ticks to match uploaded figure (-0.45 to -0.15)
    ax2.set_yticks([-0.45, -0.40, -0.35, -0.30, -0.25, -0.20, -0.15])
    ax2.grid(True, which="major", alpha=0.3)
    ax2.grid(True, which="minor", alpha=0.1)

    # ===============================
    # Save and display
    # ===============================
    # Use subplots_adjust instead of tight_layout for better layout control
    fig.subplots_adjust(left=0.08, right=0.95, bottom=0.15, top=0.92, wspace=0.25)
    save_path = RESULTS_DIR / "individual_polarization_energy.png"
    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )
    print(f"Image saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    # 1. Base configuration (as template)
    base_config = Config(
        N=500,
        k_avg=9,  # Initial value, will be modified later
        network_type="scale_free",
        epsilon=0.175,
        p_triangle=0.65,
        G_topics=9,
        alpha=0.5,
        beta=2.7,
        burn_in_sweeps=80,
        n_trials=50,
    )

    # 2. Get data for different degrees
    k_list = [6, 9, 12]
    loaded_results = []

    for k in k_list:
        # Modify current degree configuration
        current_config = Config(
            N=base_config.N,
            k_avg=k,
            network_type=base_config.network_type,
            epsilon=base_config.epsilon,
            p_triangle=base_config.p_triangle,
            G_topics=base_config.G_topics,
            alpha=base_config.alpha,
            beta=base_config.beta,
            burn_in_sweeps=base_config.burn_in_sweeps,
            n_trials=base_config.n_trials,
        )
        data = get_experiment_data(current_config, replace=False)
        loaded_results.append(data)

    # 3. Organize plotting configuration
    configs_to_plot = [
        {
            "data": loaded_results[0],
            "color": "#708090",
            "marker": "v",
            "label": r"$\langle k \rangle = 6$",
        },
        {
            "data": loaded_results[1],
            "color": "#191970",
            "marker": "o",
            "label": r"$\langle k \rangle = 9$",
        },
        {
            "data": loaded_results[2],
            "color": "#B22222",
            "marker": "s",
            "label": r"$\langle k \rangle = 12$",
        },
    ]

    # 4. Run plotting
    plot_combined_results(configs_to_plot)
