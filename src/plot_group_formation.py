from dataclasses import asdict, dataclass, field
from multiprocessing import Pool, cpu_count
from pathlib import Path

import igraph as ig
import leidenalg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import model_core
import network_generator

# Path setup
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent
RESULTS_DIR = BASE_DIR / "results" / "Fig_group_formation"
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
    N: int = 500  # Number of nodes (500 for testing, 5000 for production)
    k_values: np.ndarray = field(default_factory=lambda: np.arange(3, 8, 1))

    # === Network-specific parameters ===
    epsilon: float = 0.175  # Poisson small-world: rewiring probability
    p_triangle: float = 0.65  # Scale-free network: triangle formation probability

    # === Model dynamics parameters ===
    G_topics: int = 9  # Opinion dimension
    alpha: float = 0.5  # Friend/enemy weight
    beta: float = 2.7  # Social pressure sensitivity
    n_trials: int = 100  # Number of repeated experiments

    # === Runtime state (dynamically modified) ===
    network_type: str = "HK_configuration"  # Current network type
    burn_in_sweeps: int = 60  # Burn-in steps (dynamically set by main function)

    # === Algorithm selection ===
    community_detection_method: str = (
        "leiden"  # "leiden", "louvain", or "cc" (connected components)
    )
    leiden_partition_type: str = "CPM"  # "CPM" or "Modularity"
    resolution_parameter: float = 1e-6  # Resolution parameter for CPM partition


def networkx_to_igraph(nx_graph):
    """
    Convert networkx graph to igraph graph, preserving node mapping.

    Return:
    - ig_graph: igraph.Graph object
    """
    # Remap nodes to 0 to N-1
    nodes = list(nx_graph.nodes())
    node_map = {i: node for i, node in enumerate(nodes)}
    node_map_inv = {node: i for i, node in enumerate(nodes)}

    edges = []
    for u, v in nx_graph.edges():
        edges.append((node_map_inv[u], node_map_inv[v]))

    g = ig.Graph(len(nodes), edges)
    return g


def run_single_simulation(config_dict, k_target, seed):
    """
    Run a single simulation task
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

    # 3. Run model (pre-compute neighbor list)
    nodes = np.arange(G.number_of_nodes())
    neighbors = [np.array(list(G.neighbors(i))) for i in nodes]

    total_steps = burn_in_sweeps * G.number_of_nodes() * G_topics

    for _ in range(total_steps):
        model.step(nodes, neighbors)

    # 4. Calculate link weights and construct graph
    nodes = np.arange(G.number_of_nodes())
    edges = list(G.edges())

    # Calculate dot product weights for all existing edges
    weights = []
    for u, v in edges:
        dot_product = np.dot(model.opinions[u], model.opinions[v])
        # Normalize dot product or take sign directly, usually take sign
        # Here take sign to match weight {-1, 1}
        weights.append(1 if dot_product > 0 else -1)

    weights = np.array(weights)
    # 5. Community detection
    max_community_size = 0
    method = config_dict.get("community_detection_method", "leiden")

    if method == "cc":
        # Connected components method (only positive edges)
        G_plus = nx.Graph()
        G_plus.add_nodes_from(nodes)
        for i, (u, v) in enumerate(edges):
            if weights[i] > 0:
                G_plus.add_edge(u, v)

        if G_plus.number_of_edges() == 0:
            max_community_size = 1
        else:
            components = list(nx.connected_components(G_plus))
            max_community_size = max(len(c) for c in components)

    elif method == "leiden":
        # Use leidenalg
        ig_G = ig.Graph(len(nodes), edges)
        partition_type = config_dict.get("leiden_partition_type", "CPM")

        if partition_type == "CPM":
            # Use CPM partition, supports positive and negative weights
            res_param = config_dict.get("resolution_parameter", 1e-6)
            part = leidenalg.find_partition(
                ig_G,
                leidenalg.CPMVertexPartition,
                weights=weights.tolist(),
                resolution_parameter=res_param,
                seed=int(seed),
            )
        else:
            # Use Modularity (usually for positive graphs, be cautious with negative weights)
            # If Modularity, usually only pass positive edges
            G_plus_edges = [edges[i] for i, w in enumerate(weights) if w > 0]
            ig_G_plus = ig.Graph(len(nodes), G_plus_edges)
            part = leidenalg.find_partition(
                ig_G_plus, leidenalg.ModularityVertexPartition, seed=int(seed)
            )

        max_community_size = max(len(c) for c in part)

    elif method == "louvain":
        # Use networkx built-in Louvain (usually only supports positive weights or unweighted)
        G_plus = nx.Graph()
        G_plus.add_nodes_from(nodes)
        for i, (u, v) in enumerate(edges):
            if weights[i] > 0:
                G_plus.add_edge(u, v)

        communities = nx.community.louvain_communities(G_plus, seed=int(seed))
        if communities:
            max_community_size = max(len(c) for c in communities)
        else:
            max_community_size = 1

    # 6. Calculate maximum community ratio f
    f = max_community_size / N

    return f, actual_k


def run_single_trial_multiprocess(args):
    """
    Wrapper function for multiprocessing, unpacks parameters
    """
    return run_single_simulation(*args)


def run_experiment_multiprocess(config: Config):
    """
    Group formation experiment (multiprocess parallel version):
    Studies relationship between maximum community size f and average degree of graph.
    Uses multiprocessing for parallel acceleration
    """
    # Generate random base seed, allocate different seeds for each trial
    base_seed = 54321  # Different seed
    trial_idx = 0

    # Convert to dictionary for passing to child processes
    config_dict = asdict(config)

    # Prepare parameters for all trials
    trial_args = []
    for k in config.k_values:
        for _ in range(config.n_trials):
            trial_args.append((config_dict, k, base_seed + trial_idx))
            trial_idx += 1

    # Use multiprocessing for parallel computation
    n_processes = cpu_count()
    print(f"Using {n_processes} processes for parallel computation...")

    all_k_avg = []
    all_f = []

    with Pool(processes=n_processes) as pool:
        # Use imap to get real-time progress bar
        results = pool.imap(run_single_trial_multiprocess, trial_args)

        for f_val, k_avg in tqdm(results, total=len(trial_args), desc="Trials"):
            all_k_avg.append(k_avg)
            all_f.append(f_val)

    all_k_avg = np.array(all_k_avg)
    all_f = np.array(all_f)

    return pd.DataFrame({"k_avg": all_k_avg, "f": all_f})


def filter_group_formation_data(data, bins=60, threshold=5, ratio=0.1):
    """
    Data preprocessing: binning and filtering
    """
    if data.empty:
        return pd.DataFrame(columns=["k_center", "f_mean"])

    data = data.copy()

    # Ensure consistent column names
    if "k" not in data.columns:
        if "k_avg" in data.columns:
            data = data.rename(columns={"k_avg": "k"})
        elif "k_plus_avg" in data.columns:
            data = data.rename(columns={"k_plus_avg": "k"})

    # 1. Grouping
    bin_edges = np.linspace(data["k"].min(), data["k"].max(), bins + 1)
    data["bin"] = pd.cut(data["k"], bins=bin_edges, include_lowest=True)

    # 2. Filtering (handle bistability branches, reference plot_phase_transition.py)
    def filter_split_branches(group):
        if len(group) == 0:
            return group

        # Get bin center
        bin_interval = group.name
        k_center = (bin_interval.left + bin_interval.right) / 2

        # Select upper or lower branch based on threshold
        # For community size f, we assume it increases with degree (similar to psi)
        if k_center < threshold:
            # Left of threshold: keep low value branch (bottom)
            limit = group["f"].quantile(ratio)
            subset = group[group["f"] <= limit].copy()
        else:
            # Right of threshold: keep high value branch (top)
            limit = group["f"].quantile(1 - ratio)
            subset = group[group["f"] >= limit].copy()

        # Ensure bin column exists (if include_groups=False, bin column is removed)
        subset["bin"] = bin_interval
        return subset

    try:
        data_filtered = data.groupby("bin", observed=True, group_keys=False).apply(
            filter_split_branches, include_groups=False
        )
    except TypeError:
        # Compatible with older pandas versions
        data_filtered = data.groupby("bin", observed=True, group_keys=False).apply(
            filter_split_branches
        )

    # 3. Aggregate and calculate mean
    def aggregate_bin(group):
        if len(group) == 0:
            return pd.Series({"k_center": np.nan, "f_mean": np.nan})
        return pd.Series({"k_center": group["k"].mean(), "f_mean": group["f"].mean()})

    try:
        result = (
            data_filtered.groupby("bin", observed=True)
            .apply(aggregate_bin, include_groups=False)
            .reset_index(drop=True)
        )
    except TypeError:
        # Compatible with older pandas versions
        result = (
            data_filtered.groupby("bin", observed=True)
            .apply(aggregate_bin)
            .reset_index(drop=True)
        )

    return result.dropna()


def plot_group_formation_results(config: Config):
    """
    Plot group formation results (f vs k)
    """
    print("Starting data loading and plotting...")
    fig, ax = plt.subplots(figsize=(8, 5))

    # Define configuration parameters for three network types
    scenarios = [
        {
            "network_type": "small_world",
            "burn_in_sweeps": 180,
            "label": "Poisson small-world model",
            "marker": "o",
            "color": "#4c92c3",
            "threshold": 9.65,
            "ratio": 1,
            "offset": -0.3,
        },
        {
            "network_type": "scale_free",
            "burn_in_sweeps": 60,
            "label": "HK model",
            "marker": "^",
            "color": "#ff9f43",
            "offset": -0.3,
            "threshold": 10.0,  # Default (reference phase_transition)
            "ratio": 1,  # No filtering
        },
        {
            "network_type": "HK_configuration",
            "burn_in_sweeps": 60,
            "label": "Configuration model",
            "marker": "s",
            "color": "#54a654",
            "offset": -0.3,
            "threshold": 10.0,  # Default (reference phase_transition)
            "ratio": 1,  # No filtering
        },
    ]

    for sc in scenarios:
        # Construct filename
        filename = f"{sc['network_type']}_{config.N}nodes_{config.n_trials}trials_{sc['burn_in_sweeps']}sweeps.csv"
        file_path = RESULTS_DIR / filename

        if not file_path.exists():
            print(f"Warning: file not found {file_path}, skipping plot.")
            continue

        # Load data
        df_raw = pd.read_csv(file_path)

        # Filter k_avg greater than 3, less than 14
        df_raw = df_raw[(df_raw["k_avg"] > 4) & (df_raw["k_avg"] < 14)]

        # Filter data (mainly for binning smoothing)
        df_filtered = filter_group_formation_data(
            df_raw,
            bins=100,
            threshold=sc["threshold"],
            ratio=sc["ratio"],
        )

        if df_filtered.empty:
            print(f"Warning: {sc['label']} has no data after filtering.")
            continue

        # Plot
        ax.scatter(
            df_filtered["k_center"] + sc["offset"],
            df_filtered["f_mean"],
            marker=sc["marker"],
            label=sc["label"],
            alpha=0.9,
            edgecolors="none",
            s=30,
            color=sc["color"],
        )

    # Axis and style setup
    # ax.set_ylabel("% of nodes in largest community")
    # ax.set_xlabel("Average Degree $\\langle k \\rangle$")

    # Adjust based on data range (f is between 0 and 1)
    # Reference project range usually displays main features between 0 and 0.5
    ax.set_ylim(0.08, 0.6)
    ax.set_xlim(3, 14.2)

    # Customize spine width
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    # Customize tick parameters
    ax.tick_params(direction="out", length=6, width=1.2, labelsize=12)

    # Legend setup
    ax.legend(loc="upper left", frameon=False, fontsize=12)

    plt.tight_layout()

    # Use fixed filename format
    output_path = RESULTS_DIR / f"group_formation_N{config.N}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")


def get_group_formation_data(config, network_type, burn_in_sweeps):
    """
    Unified handling of data loading/computation and saving logic
    """
    config.network_type = network_type
    config.burn_in_sweeps = burn_in_sweeps

    filename = f"{config.network_type}_{config.N}nodes_{config.n_trials}trials_{config.burn_in_sweeps}sweeps.csv"
    file_path = RESULTS_DIR / filename

    # To force re-run, can comment out this check
    if file_path.exists():
        print(f"Data already exists: {file_path}")
        return pd.read_csv(file_path)

    print(f"Running experiment: {network_type} ...")
    df = run_experiment_multiprocess(config)
    df.to_csv(file_path, index=False)

    return df


if __name__ == "__main__":
    # Initialize global configuration
    config = Config(
        # === Network parameters ===
        N=500,  # Total number of nodes
        k_values=np.arange(3, 16, 0.1),  # Average degree range
        network_type="HK_configuration",
        epsilon=0.175,
        p_triangle=0.65,
        # === Model parameters ===
        G_topics=9,
        alpha=0.5,
        beta=2.7,
        # === Experiment parameters ===
        burn_in_sweeps=60,
        n_trials=100,
        community_detection_method="leiden",
        leiden_partition_type="CPM",
        resolution_parameter=1e-6,
    )

    # 1. Poisson small-world network
    get_group_formation_data(
        config,
        network_type="small_world",
        burn_in_sweeps=180,
    )

    # 2. HK model - scale-free network
    get_group_formation_data(
        config,
        network_type="scale_free",
        burn_in_sweeps=60,
    )

    # 3. Null model (Configuration Model)
    get_group_formation_data(
        config,
        network_type="HK_configuration",
        burn_in_sweeps=60,
    )

    # Plot
    plot_group_formation_results(config)
