from dataclasses import asdict, dataclass, field
from multiprocessing import Pool, cpu_count
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import model_core
import network_generator

# Path settings
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent
RESULTS_DIR = BASE_DIR / "results" / "Fig_polarization_phase_transition"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Set plotting style
sns.set_context("talk")
sns.set_style("ticks")


@dataclass
class Config:
    """
    Experiment parameter configuration
    """

    # === Global network parameters ===
    N: int = 500  # Number of nodes (test: 500, production: 5000)
    k_values: np.ndarray = field(default_factory=lambda: np.arange(3, 8, 1))

    # === Specific network parameters ===
    epsilon: float = 0.175  # Poisson small-world: rewiring probability
    p_triangle: float = 0.65  # Scale-free network: triangle formation probability

    # === Model dynamics parameters ===
    G_topics: int = 9  # Opinion dimension
    alpha: float = 0.5  # Friend/enemy weight
    beta: float = 2.7  # Social pressure sensitivity
    n_trials: int = 100  # Number of trials

    # === Runtime state (dynamically modified) ===
    network_type: str = "HK_configuration"  # Current network type being run
    burn_in_sweeps: int = 60  # Warmup steps (dynamically set by main function)


def run_single_simulation(config_dict, k_target, seed):
    """
    Run single simulation task
    config_dict: configuration dictionary
    k_target: target average degree
    seed: random seed
    """
    np.random.seed(seed)

    # Unpack common parameters
    N = config_dict["N"]
    G_topics = config_dict["G_topics"]
    alpha = config_dict["alpha"]
    beta = config_dict["beta"]
    burn_in_sweeps = config_dict["burn_in_sweeps"]
    network_type = config_dict["network_type"]

    # 1. Generate network
    if network_type == "small_world":
        epsilon = config_dict["epsilon"]
        G = network_generator.generate_poissonian_small_world(N, k_target, epsilon)
    elif network_type == "scale_free":
        p_triangle = config_dict["p_triangle"]
        G = network_generator.generate_stochastic_holme_kim(N, k_target, p_triangle)
    elif network_type == "HK_configuration":
        p_triangle = config_dict["p_triangle"]
        G_hk = network_generator.generate_stochastic_holme_kim(N, k_target, p_triangle)
        G = network_generator.get_configuration_null_model(G_hk)
    else:
        raise ValueError(f"Unknown network type: {network_type}")

    actual_k = 2 * G.number_of_edges() / G.number_of_nodes()

    # 2. Initialize model
    model = model_core.BinaryOpinionModel(
        N=G.number_of_nodes(), G_topics=G_topics, alpha=alpha, beta=beta
    )

    # 3. Run model (precompute neighbor list)
    nodes = np.arange(G.number_of_nodes())
    neighbors = [np.array(list(G.neighbors(i))) for i in nodes]

    total_steps = burn_in_sweeps * G.number_of_nodes() * G_topics

    for _ in range(total_steps):
        model.step(nodes, neighbors)

    # 4. Calculate polarization
    global_psi, _ = model.calculate_polarization()

    return global_psi, actual_k


def run_single_trial_multiprocess(args):
    """
    Multiprocess wrapper function to unpack parameters
    """
    return run_single_simulation(*args)


def run_experiment_multiprocess(config: Config):
    """
    Phase transition experiment (multiprocess parallel version): study relationship between polarization and average degree.
    Corresponds to Thurner_2025 Figure 4A
    Use multiprocess parallelization for speedup

    Parameters:
        config: experiment configuration object

    Returns:
        all_k_avg: array of average degrees for all experiments
        all_psi: array of polarization levels for all experiments
    """
    # Generate random base seed and assign different seed for each trial
    base_seed = 12345
    trial_idx = 0

    # Convert to dictionary for passing to subprocess
    config_dict = asdict(config)

    # Prepare parameters for all trials
    trial_args = []
    for k in config.k_values:
        for _ in range(config.n_trials):
            trial_args.append((config_dict, k, base_seed + trial_idx))
            trial_idx += 1

    # Use multiprocess parallelization
    n_processes = cpu_count()  # Use all available CPU cores
    print(f"Using {n_processes} processes for parallel computation...")

    all_k_avg = []
    all_psi = []

    with Pool(processes=n_processes) as pool:
        # Use imap for real-time progress bar
        results = pool.imap(run_single_trial_multiprocess, trial_args)

        for psi, k_avg in tqdm(results, total=len(trial_args), desc="Trials"):
            all_k_avg.append(k_avg)
            all_psi.append(psi)

    all_k_avg = np.array(all_k_avg)
    all_psi = np.array(all_psi)

    return pd.DataFrame({"k_avg": all_k_avg, "psi": all_psi})


def filter_phase_transition_data(data, bins=60, threshold=5, ratio=0.1):
    """
    Data preprocessing: binning and filtering for phase transition plot bistable data cleaning
    """
    if data.empty:
        return pd.DataFrame(columns=["k_center", "psi_mean"])

    data = data.copy()

    # Ensure consistent column names
    if "k" not in data.columns and "k_avg" in data.columns:
        data = data.rename(columns={"k_avg": "k"})

    # 1. Grouping
    bin_edges = np.linspace(data["k"].min(), data["k"].max(), bins + 1)
    data["bin"] = pd.cut(data["k"], bins=bin_edges, include_lowest=True)

    # 2. Filter anomalies (handle bistable branches in phase transition region)
    def filter_split_branches(group):
        if len(group) == 0:
            return group

        # Get bin center
        bin_interval = group.name
        k_center = (bin_interval.left + bin_interval.right) / 2

        # Based on threshold, choose to keep upper or lower branch
        if k_center < threshold:
            # Left of threshold: keep low polarization branch (lower)
            limit = group["psi"].quantile(ratio)
            return group[group["psi"] <= limit]
        else:
            # Right of threshold: keep high polarization branch (upper)
            limit = group["psi"].quantile(1 - ratio)
            return group[group["psi"] >= limit]

    try:
        data_filtered = data.groupby("bin", observed=True, group_keys=False).apply(
            filter_split_branches, include_groups=False
        )
    except TypeError:
        # Compatibility with older pandas versions (no include_groups parameter)
        data_filtered = data.groupby("bin", observed=True, group_keys=False).apply(
            filter_split_branches
        )

    # 3. Aggregate to mean
    def aggregate_bin(group):
        if len(group) == 0:
            return pd.Series({"k_center": np.nan, "psi_mean": np.nan})
        return pd.Series(
            {"k_center": group["k"].mean(), "psi_mean": group["psi"].mean()}
        )

    try:
        result = (
            data_filtered.groupby("bin", observed=True)
            .apply(aggregate_bin, include_groups=False)
            .reset_index(drop=True)
        )
    except TypeError:
        # Compatibility with older pandas versions
        result = (
            data_filtered.groupby("bin", observed=True)
            .apply(aggregate_bin)
            .reset_index(drop=True)
        )

    return result.dropna()


def plot_phase_transition_results(config: Config):
    """
    Plot phase transition graph
    Directly read saved CSV files in results directory, apply filtering and plot.

    Ensure config N and n_trials match parameters used when generating data.
    """
    print("Starting data reading and plotting...")
    fig, ax = plt.subplots(figsize=(8, 5))

    # Define three network type configurations (must match parameters in get_phase_transition_data)
    # These parameters determine filename for reading data and subsequent filtering
    scenarios = [
        {
            "network_type": "small_world",
            "burn_in_sweeps": 180,
            "label": "Poisson small-world model",
            "marker": "o",
            "color": "#4c92c3",
            "threshold": 9.65,
            "ratio": 0.2,
            "offset": 0.3,
        },
        {
            "network_type": "scale_free",
            "burn_in_sweeps": 60,
            "label": "HK model",
            "marker": "^",
            "color": "#ff9f43",
            "threshold": 10.1,
            "ratio": 1.0,
            "offset": 0.3,
        },
        {
            "network_type": "HK_configuration",
            "burn_in_sweeps": 60,
            "label": "Configuration model",
            "marker": "s",
            "color": "#54a654",
            "threshold": 9.85,
            "ratio": 1.0,
            "offset": 0.3,
        },
    ]

    for sc in scenarios:
        # Construct filename
        filename = f"{sc['network_type']}_{config.N}nodes_{config.n_trials}trials_{sc['burn_in_sweeps']}sweeps.csv"
        file_path = RESULTS_DIR / filename

        if not file_path.exists():
            print(f"Warning: file not found {file_path}, skipping plot.")
            continue

        # Read data (raw data)
        df_raw = pd.read_csv(file_path)

        # Filter data
        df_filtered = filter_phase_transition_data(
            df_raw,
            bins=80,  # Use bins consistent with main
            threshold=sc["threshold"],
            ratio=sc["ratio"],
        )

        if df_filtered.empty:
            print(f"Warning: {sc['label']} empty after filtering.")
            continue

        # Plot
        ax.scatter(
            df_filtered["k_center"] + sc["offset"],
            df_filtered["psi_mean"],
            marker=sc["marker"],
            label=sc["label"],
            alpha=0.8,
            edgecolors="none",
            s=30,
            color=sc["color"],
        )

    # Coordinate axis and style settings
    ax.set_xlim(3, 14.5)
    ax.set_ylim(0, 0.85)

    # Customize spine width
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    # Customize tick parameters
    ax.tick_params(direction="out", length=6, width=1.2, labelsize=12)

    # Legend settings
    ax.legend(frameon=False, fontsize=12, loc="upper left")

    plt.tight_layout()

    output_path = RESULTS_DIR / "polarization_kavg_phase_transition.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Chart saved to: {output_path}")

    # plt.show()


def get_phase_transition_data(config, network_type, burn_in_sweeps):
    """
    Unified data loading/computation and save logic
    """
    config.network_type = network_type
    config.burn_in_sweeps = burn_in_sweeps

    filename = f"{config.network_type}_{config.N}nodes_{config.n_trials}trials_{config.burn_in_sweeps}sweeps.csv"
    file_path = RESULTS_DIR / filename

    df = run_experiment_multiprocess(config)
    df.to_csv(file_path, index=False)

    return df


if __name__ == "__main__":
    # Initialize global configuration
    config = Config(
        # === Network parameters ===
        N=500,  # Total number of nodes
        k_values=np.arange(3, 14, 0.1),  # Average degree range
        network_type="HK_configuration",  # small_world, scale_free, HK_configuration
        epsilon=0.175,  # Rewiring probability (small-world network)
        p_triangle=0.65,  # Triangle probability (scale-free network)
        # === Model parameters ===
        G_topics=9,  # Topic dimension
        alpha=0.5,  # Friend-enemy preference
        beta=2.7,  # Social pressure sensitivity, larger means more rational
        # === Experiment parameters ===
        burn_in_sweeps=60,  # Total sweep count (1 sweep = N*G updates)
        n_trials=100,  # Number of trials per parameter set
    )

    # 1. Poisson small-world network (blue circles)
    df_pws = get_phase_transition_data(
        config,
        network_type="small_world",
        burn_in_sweeps=180,  # Small-world network needs longer warmup
    )

    # 2. HK model - scale-free network (orange triangles)
    df_hk = get_phase_transition_data(
        config,
        network_type="scale_free",
        burn_in_sweeps=60,
    )

    # 3. Null model (Configuration Model) (green squares)
    df_hkcm = get_phase_transition_data(
        config,
        network_type="HK_configuration",
        burn_in_sweeps=60,
    )

    # Plot
    plot_phase_transition_results(config)
