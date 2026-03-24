"""
Parameter dependence comparison plots:
Study the effect of average degree k_avg on global_psi, energies.mean(), and maximum community ratio f
under different parameters (alpha, beta, G_topics, p_triangle) on scale-free networks.
Output 4 rows × 3 columns composite plot.
"""

from dataclasses import asdict, dataclass, field
from multiprocessing import Pool, cpu_count
from pathlib import Path

import igraph as ig
import leidenalg
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
RESULTS_DIR = BASE_DIR / "results" / "Fig_parameter_dependence"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Set plotting style
sns.set_context("talk")
sns.set_style("ticks")


@dataclass
class Config:
    """
    Experiment parameter configuration

    Parameters
    ----------
    N : int
        Number of nodes
    k_values : np.ndarray
        Average degree range
    p_triangle : float
        Triangle formation probability (scale-free network)
    G_topics : int
        Opinion dimension
    alpha : float
        Friend/enemy weight
    beta : float
        Social pressure sensitivity
    n_trials : int
        Number of trials
    burn_in_sweeps : int
        Warmup steps (1 sweep = N * G updates)
    community_detection_method : str
        Community detection method ("leiden", "louvain", "cc")
    leiden_partition_type : str
        Leiden partition type ("CPM" or "Modularity")
    resolution_parameter : float
        Resolution parameter for CPM partition
    """

    # === Global network parameters ===
    N: int = 5000
    k_values: np.ndarray = field(default_factory=lambda: np.arange(3, 14, 0.5))

    # === Scale-free network parameters ===
    p_triangle: float = 0.65

    # === Model dynamics parameters ===
    G_topics: int = 9
    alpha: float = 0.5
    beta: float = 2.7
    n_trials: int = 100

    # === Runtime state ===
    burn_in_sweeps: int = 60

    # === Community detection parameters ===
    community_detection_method: str = "leiden"
    leiden_partition_type: str = "CPM"
    resolution_parameter: float = 1e-6


def run_single_simulation(config_dict, k_target, seed):
    """
    Run single simulation task, simultaneously calculate polarization, energy, and maximum community ratio.

    Parameters
    ----------
    config_dict : dict
        Configuration dictionary (serialized from Config)
    k_target : float
        Target average degree
    seed : int
        Random seed

    Returns
    -------
    tuple : (global_psi, energies_mean, f, actual_k)
        - global_psi : global polarization
        - energies_mean : average energy
        - f : maximum community ratio
        - actual_k : actual average degree
    """
    np.random.seed(seed)

    # Unpack parameters
    N = config_dict["N"]
    G_topics = config_dict["G_topics"]
    alpha = config_dict["alpha"]
    beta = config_dict["beta"]
    burn_in_sweeps = config_dict["burn_in_sweeps"]
    p_triangle = config_dict["p_triangle"]

    # 1. Generate scale-free network
    G = network_generator.generate_stochastic_holme_kim(N, k_target, p_triangle)
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

    # 5. Calculate normalized energy: each node's Hamiltonian divided by degree, then mean
    _, energies = model.calculate_hamilton(neighbors)
    degrees = np.array([len(nb) for nb in neighbors])
    # Avoid division by zero: isolated nodes have zero energy, skip them
    mask = degrees > 0
    normalized_energies = np.zeros_like(energies)
    normalized_energies[mask] = energies[mask] / degrees[mask]
    energies_mean = float(normalized_energies.mean())

    # 6. Community detection — calculate maximum community ratio f
    edges = list(G.edges())
    weights = []
    for u, v in edges:
        dot_product = np.dot(model.opinions[u], model.opinions[v])
        weights.append(1 if dot_product > 0 else -1)
    weights = np.array(weights)

    method = config_dict.get("community_detection_method", "leiden")

    if method == "leiden":
        ig_G = ig.Graph(G.number_of_nodes(), edges)
        partition_type = config_dict.get("leiden_partition_type", "CPM")

        if partition_type == "CPM":
            res_param = config_dict.get("resolution_parameter", 1e-6)
            part = leidenalg.find_partition(
                ig_G,
                leidenalg.CPMVertexPartition,
                weights=weights.tolist(),
                resolution_parameter=res_param,
                seed=int(seed),
            )
        else:
            # Modularity mode: use positive edges only
            plus_edges = [edges[i] for i, w in enumerate(weights) if w > 0]
            ig_G_plus = ig.Graph(G.number_of_nodes(), plus_edges)
            part = leidenalg.find_partition(
                ig_G_plus, leidenalg.ModularityVertexPartition, seed=int(seed)
            )

        max_community_size = max(len(c) for c in part)
    else:
        # Connected components method (backup)
        import networkx as nx

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

    f = max_community_size / N

    return global_psi, energies_mean, f, actual_k


def run_single_trial_multiprocess(args):
    """
    Multiprocess wrapper function to unpack parameters
    """
    return run_single_simulation(*args)


def run_experiment_multiprocess(config: Config):
    """
    Parameter dependence experiment (multiprocess parallel version).

    Parameters
    ----------
    config : Config
        Experiment parameter configuration

    Returns
    -------
    pd.DataFrame
        DataFrame containing k_avg, global_psi, energies_mean, f columns
    """
    base_seed = 12345
    trial_idx = 0

    config_dict = asdict(config)

    # Prepare parameters for all trials
    trial_args = []
    for k in config.k_values:
        for _ in range(config.n_trials):
            trial_args.append((config_dict, k, base_seed + trial_idx))
            trial_idx += 1

    # Multiprocess parallel computation
    n_processes = cpu_count()
    print(f"Using {n_processes} processes for parallel computation...")

    all_k_avg = []
    all_psi = []
    all_energy = []
    all_f = []

    with Pool(processes=n_processes) as pool:
        results = pool.imap(run_single_trial_multiprocess, trial_args)

        for psi, energy, f, k_avg in tqdm(
            results, total=len(trial_args), desc="Trials"
        ):
            all_k_avg.append(k_avg)
            all_psi.append(psi)
            all_energy.append(energy)
            all_f.append(f)

    return pd.DataFrame(
        {
            "k_avg": np.array(all_k_avg),
            "global_psi": np.array(all_psi),
            "energies_mean": np.array(all_energy),
            "f": np.array(all_f),
        }
    )


def get_experiment_data(config: Config, label: str, replace: bool = False):
    """
    Get experiment data: load if file exists and not forced to replace; otherwise recompute.

    Parameters
    ----------
    config : Config
        Experiment parameter configuration
    label : str
        Label name (for file naming, e.g. "alpha_0.4")
    replace : bool
        Whether to force recomputation

    Returns
    -------
    pd.DataFrame
        Experiment data
    """
    filename = (
        f"{label}_N{config.N}_"
        f"trials{config.n_trials}_"
        f"sweeps{config.burn_in_sweeps}.csv"
    )
    file_path = RESULTS_DIR / filename

    if file_path.exists() and not replace:
        print(f"Load existing data: {file_path}")
        return pd.read_csv(file_path)

    print(f"Run experiment: {label} ...")
    df = run_experiment_multiprocess(config)
    df.to_csv(file_path, index=False)
    print(f"Data saved to: {file_path}")

    return df


def filter_data(data, column, bins=60, threshold=5, ratio=0.1):
    """
    Data preprocessing: bin smoothing

    Parameters
    ----------
    data : pd.DataFrame
        Raw data, must contain "k_avg" column and specified column
    column : str
        Column name to process (e.g. "global_psi", "energies_mean", "f")
    bins : int
        Number of bins
    threshold : float
        Branch switching threshold (k value)
    ratio : float
        Quantile filtering ratio (1.0 = no filtering)

    Returns
    -------
    pd.DataFrame
        Filtered data with "k_center" and "{column}_mean" columns
    """
    if data.empty:
        return pd.DataFrame(columns=["k_center", f"{column}_mean"])

    df = data[["k_avg", column]].copy()
    df = df.rename(columns={"k_avg": "k"})

    # 1. Binning
    bin_edges = np.linspace(df["k"].min(), df["k"].max(), bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    df["bin_idx"] = np.digitize(df["k"], bin_edges) - 1
    df["bin_idx"] = df["bin_idx"].clip(0, bins - 1)

    # 2. Filter and aggregate by bin
    results_k = []
    results_val = []

    for b in range(bins):
        group = df[df["bin_idx"] == b]
        if len(group) == 0:
            continue

        k_center = bin_centers[b]

        # Filter bistable branches
        if ratio < 1.0:
            if k_center < threshold:
                limit = group[column].quantile(ratio)
                group = group[group[column] <= limit]
            else:
                limit = group[column].quantile(1 - ratio)
                group = group[group[column] >= limit]

        if len(group) == 0:
            continue

        results_k.append(group["k"].mean())
        results_val.append(group[column].mean())

    return pd.DataFrame(
        {
            "k_center": results_k,
            f"{column}_mean": results_val,
        }
    )


def plot_parameter_dependence(replace: bool = False):
    """
    Plot 4 rows × 3 columns parameter dependence comparison plots.

    Rows: alpha, beta, G_topics, p_triangle scans
    Columns: global_psi, energies.mean(), maximum community ratio f

    Parameters
    ----------
    replace : bool
        Whether to force recomputation of all data
    """
    # ========== Parameter scan definitions ==========
    # Default base parameters for each scan group
    base = dict(N=500, alpha=0.5, beta=2.7, G_topics=9, p_triangle=0.65, n_trials=20)

    sweep_definitions = [
        {
            "param_name": "alpha",
            "display_name": r"$\alpha$",
            "values": [0.4, 0.5, 0.6],
            "threshold": 10,
        },
        {
            "param_name": "beta",
            "display_name": r"$\beta$",
            "values": [2.5, 2.7, 3.0],
            "threshold": 10,
        },
        {
            "param_name": "G_topics",
            "display_name": r"$G$",
            "values": [6, 9, 12],
            "threshold": 10,
        },
        {
            "param_name": "p_triangle",
            "display_name": r"$p_\Delta$",
            "values": [0.05, 0.25, 0.45, 0.65],
            "threshold": 10,
        },
        {
            "param_name": "N",
            "display_name": r"$N$",
            "values": [500, 1000, 1500],
            "threshold": 10,
        },
    ]

    # Marker and color style lists
    markers = ["^", "o", "s", "D"]
    colors = ["#ff9f43", "#4c92c3", "#2d8659", "#c0392b"]

    # Y-axis column definitions
    y_columns = [
        {"col": "global_psi", "label": r"$\psi$"},
        {"col": "energies_mean", "label": r"$\langle E \rangle$"},
        {"col": "f", "label": r"$f_{max}$"},
    ]

    # ========== Data generation and collection ==========
    # sweep_data[row_idx] = [(label, value, DataFrame), ...]
    sweep_data = []
    for sweep in sweep_definitions:
        param_name = sweep["param_name"]
        row_data = []
        for val in sweep["values"]:
            # Build configuration: override scan parameter
            params = base.copy()
            params[param_name] = val
            config = Config(
                N=params["N"],
                alpha=params["alpha"],
                beta=params["beta"],
                G_topics=params["G_topics"],
                p_triangle=params["p_triangle"],
                n_trials=params["n_trials"],
                k_values=np.arange(3, 14, 0.5),
                burn_in_sweeps=60,
            )
            label = f"{param_name}_{val}"
            df = get_experiment_data(config, label, replace=replace)
            row_data.append((label, val, df))
        sweep_data.append(row_data)

    # ========== Plotting ==========
    n_rows = len(sweep_definitions)
    n_cols = len(y_columns)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 3.2 * n_rows),
        sharex=True,
    )

    for row_idx, sweep in enumerate(sweep_definitions):
        threshold = sweep["threshold"]

        for col_idx, y_def in enumerate(y_columns):
            ax = axes[row_idx, col_idx]
            col_name = y_def["col"]

            for data_idx, (label, val, df) in enumerate(sweep_data[row_idx]):
                # Filter and smooth
                df_filtered = filter_data(
                    df, col_name, bins=80, threshold=threshold, ratio=1.0
                )

                if df_filtered.empty:
                    continue

                marker = markers[data_idx % len(markers)]
                color = colors[data_idx % len(colors)]

                ax.scatter(
                    df_filtered["k_center"],
                    df_filtered[f"{col_name}_mean"],
                    marker=marker,
                    color=color,
                    alpha=0.85,
                    edgecolors="none",
                    s=25,
                    label=f"{val}" if col_idx == 0 else None,
                )

            # Axis settings
            ax.set_xlim(3, 14)
            ax.set_xticks([4, 6, 8, 10, 12, 14])

            # Border style
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
            ax.tick_params(direction="out", length=5, width=1.2, labelsize=14)

            # X-axis label only on last row
            if row_idx == n_rows - 1:
                ax.set_xlabel(r"$\langle k \rangle$", fontsize=14)

            # Y-axis label
            ax.set_ylabel(y_def["label"], fontsize=14)

            # Column title (only on first row)
            # if row_idx == 0:
            # ax.set_title(y_def["label"], fontsize=13)

        # Row title: show parameter name in legend of first column subplot
        ax_legend = axes[row_idx, 0]
        legend = ax_legend.legend(
            title=sweep["display_name"],
            frameon=False,
            fontsize=14,
            title_fontsize=14,
            loc="upper left",
        )

    plt.tight_layout()

    output_path = RESULTS_DIR / f"parameter_dependence_N{base['N']}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Chart saved to: {output_path}")


if __name__ == "__main__":
    plot_parameter_dependence(replace=False)
