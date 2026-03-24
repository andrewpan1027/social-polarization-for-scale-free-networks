import multiprocessing
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from model_core import BinaryOpinionModel
from network_generator import (
    generate_poissonian_small_world,
    generate_stochastic_holme_kim,
)

# Path settings
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "Fig_individual_energy_evolution"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    """Thermodynamic experiment configuration parameters"""

    N: int  # Total number of nodes
    k_avg: float  # Average degree
    network_type: str  # Network type
    epsilon: float  # Rewiring probability (small-world network)
    p_triangle: float  # Triangle probability (scale-free network)
    G_topics: int  # Topic dimension
    alpha: float  # Friend-enemy preference
    beta: float  # Social pressure sensitivity, larger means more rational
    hubs_pct: float  # Hub node percentage
    leaves_pct: float  # Leaf node percentage
    burn_in_sweeps: int  # Total sweep count (1 sweep = N*G updates)
    record: float  # Recording interval (MCS)
    n_trials: int  # Number of trials per parameter set


def calculate_energy_metrics(model, adj, degrees, hub_indices, leaf_indices):
    """
    Calculate energy density for core, periphery, and global
    """
    N = model.N
    energy_densities = np.zeros(N)

    for i in range(N):
        if degrees[i] > 0:
            nbs = adj[i]
            neighbor_ops = model.opinions[nbs]
            # Use method name from model_core.py
            e = model.calculate_individual_energy(model.opinions[i], neighbor_ops)
            energy_densities[i] = e / degrees[i]

    e_hub = np.mean(energy_densities[hub_indices]) if len(hub_indices) > 0 else 0.0
    e_leaf = np.mean(energy_densities[leaf_indices]) if len(leaf_indices) > 0 else 0.0
    e_global = np.mean(energy_densities)

    return e_hub, e_leaf, e_global, energy_densities


def run_single_trial(params: Config, show_progress: bool = True):
    """
    Run single complete experiment
    """
    # Initialize network
    if params.network_type == "small_world":
        graph = generate_poissonian_small_world(
            N=params.N, k_avg=params.k_avg, epsilon=params.epsilon
        )
    elif params.network_type == "scale_free":
        graph = generate_stochastic_holme_kim(
            N=params.N, k_avg=params.k_avg, p_triangle=params.p_triangle
        )
    else:
        raise ValueError(f"Unknown network_type: {params.network_type}")

    # Get neighbor list and degree for each node
    N_nodes = graph.number_of_nodes()
    nodes_all = np.arange(N_nodes)
    adj = [np.array(list(graph.neighbors(i))) for i in nodes_all]
    degrees = np.array([len(a) for a in adj])

    # Determine Hub and Leaf node indices
    # Sort by degree in descending order
    sorted_node_indices = np.argsort(degrees)[::-1]

    n_hubs = int(N_nodes * params.hubs_pct)
    n_leaves = int(N_nodes * params.leaves_pct)

    # Hubs: nodes with largest degree
    hub_indices = (
        sorted_node_indices[:n_hubs] if n_hubs > 0 else np.array([], dtype=int)
    )
    # Leaves: nodes with smallest degree (at end of descending array)
    leaf_indices = (
        sorted_node_indices[-n_leaves:] if n_leaves > 0 else np.array([], dtype=int)
    )

    # Initialize model
    model = BinaryOpinionModel(
        N=N_nodes,
        G_topics=params.G_topics,
        alpha=params.alpha,
        beta=params.beta,
    )

    # Calculate total steps and generate complete time series at once
    total_steps = int(params.burn_in_sweeps / params.record)
    time_points = [0.0] + [step * params.record for step in range(1, total_steps + 1)]

    # Record initial state (t = 0)
    e_hub, e_leaf, e_global, _ = calculate_energy_metrics(
        model, adj, degrees, hub_indices, leaf_indices
    )

    energy_hubs = [e_hub]
    energy_leaves = [e_leaf]
    energy_global = [e_global]

    # Evolve gradually and record
    step_range = range(1, total_steps + 1)
    if show_progress:
        step_range = tqdm(step_range, desc="Sweeps", leave=False)

    for _ in step_range:
        # run_sweeps is not a native method of BinaryOpinionModel, need to loop step manually
        # 1 MCS = N * G steps
        steps_per_sweep = int(N_nodes * params.G_topics * params.record)
        for _ in range(steps_per_sweep):
            model.step(nodes_all, adj)

        # Calculate current energy density
        e_hub, e_leaf, e_global, _ = calculate_energy_metrics(
            model, adj, degrees, hub_indices, leaf_indices
        )
        energy_hubs.append(e_hub)
        energy_leaves.append(e_leaf)
        energy_global.append(e_global)

    # Collect final state distribution
    _, _, _, final_densities = calculate_energy_metrics(
        model, adj, degrees, hub_indices, leaf_indices
    )

    dist_hubs_nodes = (
        final_densities[hub_indices]
        if hub_indices.size > 0
        else np.array([], dtype=float)
    )
    dist_leaves_nodes = (
        final_densities[leaf_indices]
        if leaf_indices.size > 0
        else np.array([], dtype=float)
    )
    dist_global_nodes = final_densities

    # Return time and sequence data, as well as final state distribution
    return (
        np.array(time_points),
        np.array(energy_hubs),
        np.array(energy_leaves),
        np.array(energy_global),
        dist_hubs_nodes,
        dist_leaves_nodes,
        dist_global_nodes,
    )


def run_single_trial_task(args):
    """
    Module-level helper function for multiprocess parallelization
    """
    params, seed = args
    np.random.seed(seed)
    return run_single_trial(params, show_progress=False)


def run_experiment_parallel(params: Config):
    """
    Thermodynamic experiment (multiprocess parallel version)
    """
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


def aggregate_results(raw_results):
    """
    Aggregate multiple simulations: time series take mean, scatter concatenate
    raw_results: List of (time, energy_hubs, energy_leaves, energy_global, dist_hubs, dist_leaves, dist_global)
    """
    # Assume all trials have the same time sequence
    time_points = raw_results[0][0]

    # Collect time series from all trials
    all_energy_hubs_series = np.array([r[1] for r in raw_results])
    all_energy_leaves_series = np.array([r[2] for r in raw_results])
    all_energy_global_series = np.array([r[3] for r in raw_results])

    # Calculate mean time series
    mean_energy_hubs = np.mean(all_energy_hubs_series, axis=0)
    mean_energy_leaves = np.mean(all_energy_leaves_series, axis=0)
    mean_energy_global = np.mean(all_energy_global_series, axis=0)

    # Calculate standard deviation time series
    std_energy_hubs = np.std(all_energy_hubs_series, axis=0)
    std_energy_leaves = np.std(all_energy_leaves_series, axis=0)
    std_energy_global = np.std(all_energy_global_series, axis=0)

    # Collect distribution data (concatenate final state node distribution from all trials)
    dist_hubs_all = np.concatenate([r[4] for r in raw_results])
    dist_leaves_all = np.concatenate([r[5] for r in raw_results])
    dist_global_all = np.concatenate([r[6] for r in raw_results])

    return {
        "time": time_points,
        "energy_hubs": mean_energy_hubs,
        "energy_leaves": mean_energy_leaves,
        "energy_global": mean_energy_global,
        "std_energy_hubs": std_energy_hubs,
        "std_energy_leaves": std_energy_leaves,
        "std_energy_global": std_energy_global,
        "dist_energy_hubs": dist_hubs_all,
        "dist_energy_leaves": dist_leaves_all,
        "dist_energy_global": dist_global_all,
    }


def get_experiment_data(params: Config, replace: bool = False):
    """
    Get experiment data: load if file exists and replace=False; otherwise recompute and save.
    """
    save_fn = f"{params.network_type}_{params.N}nodes_{int(params.k_avg)}degrees_{params.n_trials}trials_{params.burn_in_sweeps}sweeps.npz"
    file_path = RESULTS_DIR / save_fn

    if file_path.exists() and not replace:
        print(f"Data file exists, loading directly: {save_fn}")
        return np.load(file_path)

    print(f"Starting experiment computation: {save_fn}")
    raw_res = run_experiment_parallel(params)
    res_agg = aggregate_results(raw_res)

    # Save results
    np.savez(
        file_path,
        time=res_agg["time"],
        energy_hubs=res_agg["energy_hubs"],
        energy_leaves=res_agg["energy_leaves"],
        energy_global=res_agg["energy_global"],
        std_energy_hubs=res_agg["std_energy_hubs"],
        std_energy_leaves=res_agg["std_energy_leaves"],
        std_energy_global=res_agg["std_energy_global"],
        dist_energy_hubs=res_agg["dist_energy_hubs"],
        dist_energy_leaves=res_agg["dist_energy_leaves"],
        dist_energy_global=res_agg["dist_energy_global"],
    )
    print(f"Experiment completed, results saved to: {file_path}")

    return np.load(file_path)


def plot_results(result, params: Config):
    """
    Plot individual energy potential evolution (only ax1 content, x-axis changed to steps)
    """
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 14,
            "axes.labelpad": 12,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "figure.titlesize": 16,
            "figure.titleweight": "bold",
        }
    )

    fig = plt.figure(figsize=(10, 7), dpi=150)
    ax1 = fig.add_subplot(111)

    # Plot time series
    # Convert time to Steps (Total Updates = MCS * N * G)
    steps = result["time"] * params.N * params.G_topics

    ax1.plot(
        steps,
        result["energy_hubs"],
        color="#D62728",
        lw=3,
        label="Hubs (Core)",
    )
    ax1.fill_between(
        steps,
        result["energy_hubs"] - result["std_energy_hubs"],
        result["energy_hubs"] + result["std_energy_hubs"],
        color="#D62728",
        alpha=0.2,
        edgecolor="none",
    )

    ax1.plot(
        steps,
        result["energy_leaves"],
        color="#1F77B4",
        lw=3,
        label="Leaves (Periphery)",
    )
    ax1.fill_between(
        steps,
        result["energy_leaves"] - result["std_energy_leaves"],
        result["energy_leaves"] + result["std_energy_leaves"],
        color="#1F77B4",
        alpha=0.2,
        edgecolor="none",
    )

    ax1.plot(
        steps,
        result["energy_global"],
        color="#2CA02C",
        lw=2,
        linestyle="--",
        alpha=0.7,
        label="Global Average",
    )
    ax1.fill_between(
        steps,
        result["energy_global"] - result["std_energy_global"],
        result["energy_global"] + result["std_energy_global"],
        color="#2CA02C",
        alpha=0.15,
        edgecolor="none",
    )

    # Annotate energy potential difference (Frustration Transfer) - at last time point
    final_t = steps[-1]
    final_hub = result["energy_hubs"][-1]
    final_leaf = result["energy_leaves"][-1]

    # ax1.annotate(
    #     "",
    #     xy=(final_t, final_leaf),
    #     xytext=(final_t, final_hub),
    #     arrowprops=dict(arrowstyle="<->", color="black", lw=1.5),
    # )
    # ax1.text(
    #     final_t - 5,
    #     (final_hub + final_leaf) / 2,
    #     "Energy Gap",
    #     ha="right",
    #     va="center",
    #     fontsize=11,
    #     fontweight="bold",
    # )

    # ax1.set_xlabel("Total Steps", fontsize=14, fontweight="bold")
    # ax1.set_ylabel(
    #     "Avg Energy Density $\\langle H_i/k_i \\rangle$", fontsize=14, fontweight="bold"
    # )
    # ax1.set_title(
    #     "Kinetic Decoupling & Frustration Transfer",
    #     fontsize=15,
    #     fontweight="bold",
    #     loc="left",
    # )
    # ax1.set_ylim(-0.5, 0)
    ax1.legend(frameon=False, fontsize=14, loc="upper right")
    ax1.grid(alpha=0.3)

    # Save image
    plt.tight_layout()
    save_path = (
        RESULTS_DIR / f"energy_evolution_{params.network_type}_N{params.N}_steps.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Image saved to: {save_path}")
    # plt.show()


if __name__ == "__main__":
    # Create experiment configuration
    config = Config(
        # === Network parameters ===
        N=5000,  # Total number of nodes
        k_avg=12,  # Average degree
        network_type="scale_free",  # Network type: 'small_world' or 'scale_free'
        epsilon=0.175,  # Rewiring probability (small-world network)
        p_triangle=0.65,  # Triangle probability (scale-free network)
        # === Model parameters ===
        G_topics=9,  # Topic dimension
        alpha=0.5,  # Friend-enemy preference
        beta=2.7,  # Social pressure sensitivity, larger means more rational
        hubs_pct=0.1,  # Hub node percentage
        leaves_pct=0.5,  # Leaf node percentage
        # === Experiment parameters ===
        burn_in_sweeps=30,  # Total sweep count (1 sweep = N*G updates)
        record=1.0,  # Recording interval (MCS)
        n_trials=50,  # Number of trials per parameter set
    )

    # Get data
    data = get_experiment_data(config, replace=True)

    # Plot results
    plot_results(data, config)
