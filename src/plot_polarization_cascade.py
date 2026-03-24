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
RESULTS_DIR = BASE_DIR / "results" / "Fig_polarization_cascade"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    """Cascade experiment configuration parameters"""

    N: int  # Total number of nodes
    k_avg: float  # Average degree
    network_type: str  # Network type
    epsilon: float  # Rewiring probability (small-world network)
    p_triangle: float  # Triangle probability (scale-free network)
    G_topics: int  # Topic dimension
    alpha: float  # Friend-enemy preference
    beta: float  # Social pressure sensitivity, larger means more rational
    burn_in_sweeps: int  # Total duration (MCS)
    record_steps: int  # Recording interval (Steps)
    n_trials: int  # Number of trials per parameter set
    n_bins: int = 100  # Normalize to how many percentiles (e.g. 100)


def run_single_trial(params: Config, show_progress: bool = True):
    """
    Run single complete cascade experiment
    """
    # 1. Initialize network
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

    n_nodes = graph.number_of_nodes()
    nodes_all = np.arange(n_nodes)
    adj = [np.array(list(graph.neighbors(i))) for i in nodes_all]
    degrees = np.array([len(a) for a in adj])

    # 2. Get node indices sorted by degree
    # Sort from small to large
    sorted_indices = np.argsort(degrees)

    # 3. Initialize model
    model = BinaryOpinionModel(
        N=n_nodes,
        G_topics=params.G_topics,
        alpha=params.alpha,
        beta=params.beta,
    )

    # 4. Simulate and record
    records = []

    # Calculate time steps
    total_sweeps = params.burn_in_sweeps
    steps_per_sweep = n_nodes * params.G_topics
    record_steps = params.record_steps
    total_steps = int(total_sweeps * steps_per_sweep)

    current_step = 0

    # Record initial state (t=0)
    _, individual_psi = model.calculate_polarization()
    # Sort by degree
    sorted_psi = individual_psi[sorted_indices]

    # Bin to mean
    # Divide sorted psi into n_bins parts, each calculate mean
    # Use array_split to handle non-divisible cases
    chunks = np.array_split(sorted_psi, params.n_bins)
    binned_psi = np.array([np.mean(chunk) for chunk in chunks])
    records.append(binned_psi)

    pbar = None
    if show_progress:
        pbar = tqdm(total=total_steps, desc="Evolution", leave=False)

    while current_step < total_steps:
        # Run one recording interval
        # Note: run_sweeps may slightly exceed target step, control manually here
        steps_to_run = min(record_steps, total_steps - current_step)

        # For efficiency, could batch update, but model.step is single-step update
        # Better if model has run_sweeps method, but model_core.py doesn't
        # Loop calling model.step here
        for _ in range(steps_to_run):
            model.step(nodes_all, adj)

        current_step += steps_to_run

        if pbar:
            pbar.update(steps_to_run)

        # Record state
        _, individual_psi = model.calculate_polarization()
        sorted_psi = individual_psi[sorted_indices]
        chunks = np.array_split(sorted_psi, params.n_bins)
        binned_psi = np.array([np.mean(chunk) for chunk in chunks])
        records.append(binned_psi)

    if pbar:
        pbar.close()

    return np.array(records)  # Shape: (T, n_bins)


def run_single_trial_task(args):
    """Multiprocess helper function"""
    params, seed = args
    np.random.seed(seed)
    return run_single_trial(params, show_progress=False)


def run_experiment_parallel(params: Config):
    """Run multiple trials in parallel and take mean"""
    base_seed = np.random.randint(0, 1000000)
    tasks = [(params, base_seed + i) for i in range(params.n_trials)]

    n_cpu = max(1, cpu_count())
    print(f"Starting parallel experiment with {n_cpu} CPUs...")

    all_records = []
    # spawn startup method may have issues, but on Linux/Mac fork is default
    # Mac defaults to spawn, may need to check Pickling, Config dataclass should be picklable
    with Pool(processes=n_cpu) as pool:
        for res in tqdm(
            pool.imap_unordered(run_single_trial_task, tasks),
            total=len(tasks),
            desc="Trials",
        ):
            all_records.append(res)

    # Convert to numpy array: (n_trials, T, n_bins)
    # Truncate to minimum length in case
    min_len = min(len(r) for r in all_records)
    all_records_truncated = [r[:min_len] for r in all_records]

    all_records_array = np.array(all_records_truncated)

    # Take mean over trials -> (T, n_bins)
    mean_records = np.mean(all_records_array, axis=0)

    # Transpose to (n_bins, T) for imshow (rows are y-axis)
    heatmap_matrix = mean_records.T

    # Generate time point array (stored as steps, since record_steps is steps)
    time_points = np.arange(min_len) * params.record_steps

    return time_points, heatmap_matrix


def get_experiment_data(params: Config, replace: bool = False):
    """Get or compute experiment data"""
    save_fn = (
        f"{params.network_type}_{params.N}nodes_{int(params.k_avg)}degrees_"
        f"{params.n_trials}trials_{params.burn_in_sweeps}sweeps.npz"
    )
    file_path = RESULTS_DIR / save_fn

    if file_path.exists() and not replace:
        print(f"Loading existing data: {file_path}")
        data = np.load(file_path)
        return data["time_points"], data["heatmap_matrix"]

    print(f"Running new experiment: {save_fn}")
    time_points, heatmap_matrix = run_experiment_parallel(params)

    np.savez(file_path, time_points=time_points, heatmap_matrix=heatmap_matrix)
    print(f"Data saved to: {file_path}")

    return time_points, heatmap_matrix


def plot_result(time_points, heatmap_matrix, params: Config):
    """Plot heatmap"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # time_points now stores steps
    steps_points = time_points

    # Plot heatmap
    # origin='lower' means matrix row 0 at bottom
    # heatmap_matrix[0] corresponds to degree lowest bin (Bottom 1%)
    img = ax.imshow(
        heatmap_matrix,
        origin="lower",
        aspect="auto",
        interpolation="bilinear",  # Use bilinear interpolation for smoother image
        cmap="inferno",
        vmin=0.0,
        vmax=0.6,
        extent=[steps_points[0], steps_points[-1], 0, 100],  # X: steps, Y: 0-100%
    )

    ax.set_xlabel("Steps", fontsize=14)
    ax.set_ylabel("Node Degree Percentile (Lowest to Highest)", fontsize=14)

    # X-axis ticks
    total_steps = params.burn_in_sweeps * params.N * params.G_topics
    ax.set_xticks(np.linspace(0, total_steps, 6))

    # Y-axis ticks: show percentages
    yticks = [0, 25, 50, 75, 100]
    ytick_labels = ["0%", "25%", "50%", "75%", "100%"]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)
    ax.tick_params(axis="both", which="major", labelsize=12)

    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label(r"Individual Polarization $\psi_i$")

    # ax.set_title(
    #     f"Polarization Cascade ({params.network_type}, N={params.N}, <k>={params.k_avg})"
    # )

    # Add parameter text
    param_text = (
        f"N={params.N}, <k>={params.k_avg}\n"
        f"G={params.G_topics}, alpha={params.alpha}, beta={params.beta}\n"
        f"Trials={params.n_trials}"
    )
    # ax.text(
    #     0.02,
    #     0.98,
    #     param_text,
    #     transform=ax.transAxes,
    #     fontsize=10,
    #     verticalalignment="top",
    #     bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    # )

    plt.tight_layout()
    save_fn = (
        f"polarization_cascade_{params.network_type}_N{params.N}_k{params.k_avg}.png"
    )
    save_path = RESULTS_DIR / save_fn
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to: {save_path}")
    # plt.show() # In server/headless environment may not need show


if __name__ == "__main__":
    # Experiment configuration
    config = Config(
        N=500,
        k_avg=9,
        network_type="scale_free",
        epsilon=0.175,
        p_triangle=0.65,
        G_topics=9,
        alpha=0.5,
        beta=2.7,
        burn_in_sweeps=40,
        record_steps=200,  # Record every 200 steps
        n_trials=100,  # Increase trials for smoother heatmap
        n_bins=100,  # Y-axis resolution
    )

    # Get data
    # replace=False means load from file if exists, set to True to force rerun
    time_points, heatmap_matrix = get_experiment_data(config, replace=False)

    # Plot
    plot_result(time_points, heatmap_matrix, config)
