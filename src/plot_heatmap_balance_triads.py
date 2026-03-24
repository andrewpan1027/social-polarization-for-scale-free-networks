import multiprocessing
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

# Import core model and network generator
from model_core import BinaryOpinionModel
from network_generator import (
    generate_poissonian_small_world,
    generate_stochastic_holme_kim,
)

# Path settings
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "Fig_heatmap_balance_triads"
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
    # Grid definition stored here
    alpha_start: float = 0.0
    alpha_end: float = 1.0
    alpha_steps: int = 41

    beta_start: float = 0.0
    beta_end: float = 5.0
    beta_steps: int = 101


def calculate_triad_ratios(model: BinaryOpinionModel, adj_indices: list[np.ndarray]):
    """
    Calculate ratios of four triad types (+++, ++-, +--, ---)
    (Optimized version: using graph structure iteration)
    """
    N = model.N
    opinions = model.opinions  # (N, G)

    # Counter: 0: ---, 1: +--, 2: ++-, 3: +++
    counts = np.zeros(4, dtype=int)

    # For fast edge sign lookup, use cached dictionary
    adj_sets = [set(nbs) for nbs in adj_indices]
    edge_signs = {}

    # Iterate through all triangles
    for i in range(N):
        nbs_i = adj_indices[i]
        nbs_i_upper = nbs_i[nbs_i > i]

        for j in nbs_i_upper:
            # Calculate edge (i, j) sign
            if (i, j) not in edge_signs:
                dot_ij = np.dot(opinions[i], opinions[j])
                sign_ij = 1 if dot_ij > 0 else -1
                edge_signs[(i, j)] = sign_ij
            else:
                sign_ij = edge_signs[(i, j)]

            nbs_j = adj_indices[j]
            nbs_j_upper = nbs_j[nbs_j > j]

            for k in nbs_j_upper:
                if k in adj_sets[i]:
                    # Found triangle (i, j, k)
                    s1 = sign_ij

                    if (j, k) not in edge_signs:
                        dot_jk = np.dot(opinions[j], opinions[k])
                        s2 = 1 if dot_jk > 0 else -1
                        edge_signs[(j, k)] = s2
                    else:
                        s2 = edge_signs[(j, k)]

                    if (i, k) not in edge_signs:
                        dot_ik = np.dot(opinions[i], opinions[k])
                        s3 = 1 if dot_ik > 0 else -1
                        edge_signs[(i, k)] = s3
                    else:
                        s3 = edge_signs[(i, k)]

                    # Count positive edges
                    pos_count = 0
                    if s1 > 0:
                        pos_count += 1
                    if s2 > 0:
                        pos_count += 1
                    if s3 > 0:
                        pos_count += 1

                    counts[pos_count] += 1

    total_triads = counts.sum()
    if total_triads == 0:
        return (0.0, 0.0, 0.0, 0.0)

    return (
        counts[3] / total_triads,  # +++
        counts[2] / total_triads,  # ++-
        counts[1] / total_triads,  # +--
        counts[0] / total_triads,  # ---
    )


def run_single_simulation(args):
    """
    Run single simulation
    args: (config, alpha, beta, seed)
    """
    config, alpha, beta, seed = args
    # Set random seed only once
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
    adj = [np.array(list(graph.neighbors(i))) for i in nodes_all]

    # 2. Initialize model
    model = BinaryOpinionModel(
        N=n_nodes, G_topics=config.G_topics, alpha=alpha, beta=beta
    )

    # 3. Evolution
    total_steps = config.burn_in_sweeps * n_nodes * config.G_topics
    for _ in range(total_steps):
        model.step(nodes_all, adj)

    # 4. Calculate triad ratios
    ratios = calculate_triad_ratios(model, adj)

    return (alpha, beta, ratios)


def run_experiment_parallel(config: Config):
    """
    Run experiment in parallel, scanning parameter space
    """
    alphas = np.linspace(config.alpha_start, config.alpha_end, config.alpha_steps)
    betas = np.linspace(config.beta_start, config.beta_end, config.beta_steps)

    tasks = []
    base_seed = np.random.randint(1000000)
    seed_counter = 0

    for a in alphas:
        for b in betas:
            # Generate separate tasks for each trial
            for _ in range(config.n_trials):
                tasks.append((config, a, b, base_seed + seed_counter))
                seed_counter += 1

    n_cpu = max(1, cpu_count())
    print(
        f"Starting parallel computation, tasks: {len(tasks)}, CPU cores: {n_cpu} ({config.n_trials} trials per parameter point)"
    )

    results_map_sum = {}  # (idx_a, idx_b) -> sum_ratios
    results_map_count = {}  # (idx_a, idx_b) -> count

    with Pool(processes=n_cpu) as pool:
        for res in tqdm(
            pool.imap_unordered(run_single_simulation, tasks), total=len(tasks)
        ):
            a, b, ratios = res
            idx_a = (np.abs(alphas - a)).argmin()
            idx_b = (np.abs(betas - b)).argmin()

            key = (idx_a, idx_b)
            if key not in results_map_sum:
                results_map_sum[key] = np.zeros(4)
                results_map_count[key] = 0

            results_map_sum[key] += np.array(ratios)
            results_map_count[key] += 1

    # Organize results into matrices
    grid_3_pos = np.zeros((len(alphas), len(betas)))
    grid_2_pos = np.zeros((len(alphas), len(betas)))
    grid_1_pos = np.zeros((len(alphas), len(betas)))
    grid_0_pos = np.zeros((len(alphas), len(betas)))

    for idx_a in range(len(alphas)):
        for idx_b in range(len(betas)):
            if (idx_a, idx_b) in results_map_sum:
                # Average logic
                r = results_map_sum[(idx_a, idx_b)] / results_map_count[(idx_a, idx_b)]
                grid_3_pos[idx_a, idx_b] = r[0]
                grid_2_pos[idx_a, idx_b] = r[1]
                grid_1_pos[idx_a, idx_b] = r[2]
                grid_0_pos[idx_a, idx_b] = r[3]

    return {
        "alphas": alphas,
        "betas": betas,
        "grid_3_pos": grid_3_pos,
        "grid_2_pos": grid_2_pos,
        "grid_1_pos": grid_1_pos,
        "grid_0_pos": grid_0_pos,
    }


def get_experiment_data(config: Config, replace: bool = False):
    """
    Get experiment data: load if file exists and not forced to replace; otherwise recompute.
    """
    # Construct filename containing key parameters to distinguish different experiments
    save_fn = (
        f"triad_{config.network_type}_N{config.N}_k{int(config.k_avg)}_"
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
        grid_3_pos=data["grid_3_pos"],
        grid_2_pos=data["grid_2_pos"],
        grid_1_pos=data["grid_1_pos"],
        grid_0_pos=data["grid_0_pos"],
    )
    print(f"Data saved to: {file_path}")
    return np.load(file_path)


def plot_heatmaps(data, config: Config):
    """
    Plot 2x2 heatmaps
    """
    alphas = data["alphas"]
    betas = data["betas"]
    g3 = data["grid_3_pos"]
    g2 = data["grid_2_pos"]
    g1 = data["grid_1_pos"]
    g0 = data["grid_0_pos"]

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

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    extent = [betas[0], betas[-1], alphas[0], alphas[-1]]

    # Plot subplots
    # Subplot 1: +++
    ax = axes[0, 0]
    im = ax.imshow(
        g3, extent=extent, origin="lower", cmap="viridis", aspect="auto", vmin=0, vmax=1
    )
    ax.set_title("+++ (Balanced)")
    ax.set_ylabel(r"$\alpha$")

    # Subplot 2: +--
    ax = axes[0, 1]
    im = ax.imshow(
        g1, extent=extent, origin="lower", cmap="viridis", aspect="auto", vmin=0, vmax=1
    )
    ax.set_title("+-- (Balanced)")

    # Subplot 3: ++-
    ax = axes[1, 0]
    im = ax.imshow(
        g2, extent=extent, origin="lower", cmap="viridis", aspect="auto", vmin=0, vmax=1
    )
    ax.set_title("++- (Unbalanced)")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\alpha$")

    # Subplot 4: ---
    ax = axes[1, 1]
    im = ax.imshow(
        g0, extent=extent, origin="lower", cmap="viridis", aspect="auto", vmin=0, vmax=1
    )
    ax.set_title("--- (Unbalanced)")
    ax.set_xlabel(r"$\beta$")

    # Adjust layout first to leave space for colorbar
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    # Add colorbar
    fig.colorbar(im, ax=axes, label="ratio", fraction=0.046, pad=0.04)

    # Construct save filename with information
    save_fn = (
        f"triad_heatmaps_{config.network_type}_N{config.N}_k{int(config.k_avg)}.png"
    )
    save_path = RESULTS_DIR / save_fn

    plt.savefig(save_path, dpi=300)
    print(f"Image saved to: {save_path}")
    # plt.show()


if __name__ == "__main__":
    # Configure experiment
    config = Config(
        network_type="scale_free",  # stochastic holme kim
        N=500,
        k_avg=12,
        G_topics=9,
        burn_in_sweeps=35,
        p_triangle=0.65,
        # Production parameters
        alpha_steps=41,
        beta_steps=101,
        n_trials=10,  # Increase number of trials
    )

    # Get data (auto cache)
    data = get_experiment_data(config, replace=False)

    # Plot
    plot_heatmaps(data, config)
