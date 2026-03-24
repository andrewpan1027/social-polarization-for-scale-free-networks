from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm

# Import core model and network generator
from model_core import BinaryOpinionModel
from network_generator import (
    generate_poissonian_small_world,
    generate_stochastic_holme_kim,
)

# Path settings
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "Fig_opinion_network"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    """Network opinion visualization experiment configuration parameters"""

    N: int  # Total number of nodes
    k_avg: float  # Average degree
    network_type: str  # Network type
    epsilon: float  # Rewiring probability (small-world network)
    p_triangle: float  # Triangle probability (scale-free network)
    G_topics: int  # Issue dimension
    alpha: float  # Friend/enemy weight parameter
    beta: float  # Social pressure sensitivity (inverse temperature)
    burn_in_sweeps: int  # Evolution duration (MCS)
    dynamic_layout_type: str = "spring"  # Layout type: "spring" or "circular"
    retention: float = 0.6  # Inertia parameter (for layout)


def run_simulation_to_equilibrium(params: Config):
    """
    Run single complete simulation, evolve to equilibrium

    Parameters
    ----------
    params : Config
        Experiment parameters

    Returns
    -------
    tuple
        (graph, final_opinions, node_degrees)
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
    step_range = tqdm(range(total_steps), desc="Evolve to equilibrium")

    for _ in step_range:
        model.step(nodes_all, adj)

    return graph, model.opinions.copy(), degrees


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
    dict
        Dictionary containing graph, opinions, degrees
    """
    save_fn = f"{params.network_type}_{params.N}nodes_{int(params.k_avg)}degrees_{params.burn_in_sweeps}sweeps.npz"
    file_path = RESULTS_DIR / save_fn

    if file_path.exists() and not replace:
        print(f"Data file exists, loading directly: {save_fn}")
        data = np.load(file_path, allow_pickle=True)

        # Reconstruct graph structure
        edges = data["edges"]
        graph = nx.Graph()
        graph.add_nodes_from(range(params.N))
        graph.add_edges_from(edges)

        return {
            "graph": graph,
            "opinions": data["opinions"],
            "degrees": data["degrees"],
        }

    print(f"Starting experiment computation: {save_fn}")
    graph, opinions, degrees = run_simulation_to_equilibrium(params)

    # Save results
    edges = np.array(list(graph.edges()))
    np.savez(
        file_path,
        edges=edges,
        opinions=opinions,
        degrees=degrees,
    )
    print(f"Experiment completed, results saved to: {file_path}")

    return {
        "graph": graph,
        "opinions": opinions,
        "degrees": degrees,
    }


def plot_dynamic_network(graph, opinions, degrees, params: Config, save_path: Path):
    """
    Plot dynamic opinion network (mimicking _plot_dynamic_network from visualize_polarization_evolution.py)

    Parameters
    ----------
    graph : nx.Graph
        Network graph
    opinions : np.ndarray
        Opinion matrix (N, G_topics)
    degrees : np.ndarray
        Node degrees
    params : Config
        Configuration parameters
    save_path : Path
        Save path
    """
    # Set plotting style
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
        }
    )

    # Create canvas
    fig, ax = plt.subplots(figsize=(3, 3), dpi=120)
    # ax.set_title("Dynamic Opinion Network", fontsize=14, fontweight="bold")

    N_nodes = graph.number_of_nodes()
    nodes = np.arange(N_nodes)

    # 1. Calculate edge weights and colors
    weights = {}
    edge_vals = []
    for u, v in graph.edges():
        dot = np.dot(opinions[u], opinions[v]) / params.G_topics
        weights[(u, v)] = 1.0 + dot
        edge_vals.append(dot)

    nx.set_edge_attributes(graph, weights, "weight")

    # Calculate mean opinion (for node color and layout force)
    mean_opinions = np.mean(opinions, axis=1)

    # 2. Layout calculation
    if params.dynamic_layout_type == "circular":
        new_pos = nx.circular_layout(graph)
    else:
        # Spring layout (snapshot mode, high iterations)
        new_pos = nx.spring_layout(
            graph,
            pos=None,
            iterations=100,
            k=1.2 / np.sqrt(N_nodes),
            weight="weight",
            threshold=0.0001,
        )

        # 3. Apply left-right separation force (visualization enhancement)
        # Snapshot mode has lower inertia (0.8 retention)
        force = 1.0 - params.retention

        for i in nodes:
            target_x = 0.6 if mean_opinions[i] > 0 else -0.6
            new_pos[i][0] = new_pos[i][0] * params.retention + target_x * force
            # Also add slight centripetal force to prevent overall drift
            new_pos[i][1] *= 0.95

        # 4. Position normalization: prevent outliers from escaping and shrinking overall view
        # Convert all coordinates to array for batch operations
        pos_array = np.array([new_pos[i] for i in nodes])

        # Center adjustment: subtract mean to keep network at canvas center
        pos_array -= pos_array.mean(axis=0)

        # Scale control: ensure 95% of points are in [-0.9, 0.9] range, avoid extreme outliers affecting scale
        curr_scale = np.percentile(np.abs(pos_array), 95)
        if curr_scale > 0:
            pos_array = pos_array * (0.8 / curr_scale)

        # Force clipping: force rare escaping points back to boundary
        pos_array = np.clip(pos_array, -1.0, 1.0)

        # Write back to layout dictionary
        for i in nodes:
            new_pos[i] = pos_array[i]

    # 5. Calculate node size (based on degree)
    base_node_size = 400 / N_nodes
    node_sizes = base_node_size + degrees * (20 / np.sqrt(N_nodes))

    # 6. Plot edges (use fixed color: green for positive correlation, red for negative)
    edge_colors = ["lightgreen" if dot > 0 else "lightcoral" for dot in edge_vals]
    nx.draw_networkx_edges(
        graph,
        new_pos,
        ax=ax,
        edge_color=edge_colors,
        alpha=0.2,
        width=0.2,
    )

    # Plot nodes (use fixed color: red for positive opinion, blue for negative)
    node_colors = ["#e74c3c" if m > 0 else "#3498db" for m in mean_opinions]
    nx.draw_networkx_nodes(
        graph,
        new_pos,
        ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="black",
        linewidths=0.2,
    )

    ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)

    # Save image
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Image saved to: {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    # 1. Base configuration
    config = Config(
        N=500,
        k_avg=6,
        network_type="scale_free",
        epsilon=0.175,
        p_triangle=0.65,
        G_topics=9,
        alpha=0.5,
        beta=2.7,
        burn_in_sweeps=80,
        dynamic_layout_type="spring",  # or "circular"
        retention=0.9,
    )

    # 2. Get data (generate or load)
    data = get_experiment_data(config, replace=True)

    # 3. Plot network
    save_path = (
        RESULTS_DIR
        / f"opinion_network_{config.network_type}_N{config.N}_k{int(config.k_avg)}.png"
    )
    plot_dynamic_network(
        graph=data["graph"],
        opinions=data["opinions"],
        degrees=data["degrees"],
        params=config,
        save_path=save_path,
    )
