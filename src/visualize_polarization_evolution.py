from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm

from model_core import BinaryOpinionModel
from network_generator import (
    generate_barabasi_albert,
    generate_erdos_renyi,
    generate_poissonian_small_world,
    generate_random_tree,
    generate_regular_lattice,
    generate_stochastic_holme_kim,
    remove_triangles_rewire,
)

# Set matplotlib backend
matplotlib.use("Agg")

# Path settings
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results" / "polarization_evolution"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    """
    Experiment and visualization configuration parameters
    """

    # Network parameters
    network_type: str = "scale_free"
    N: int = 500
    k_avg: int = 10
    epsilon: float = 0.175  # Rewiring probability (small-world network)
    p_triangle: float = 0.65  # Triangle probability (scale-free network)
    hub_fraction: float = 0.05  # Hub node fraction (default Top 5%)

    # Model core parameters
    G_topics: int = 9
    alpha: float = 0.5
    beta: float = 2.7

    # Simulation parameters
    total_steps: int = int(1e5)
    steps_per_record: int = int(1e3)  # Record data every how many steps

    # Visualization parameters
    fps: int = 5
    dynamic_layout_type: str = (
        "force_directed"  # "force_directed" (force-directed) or "circular" (circular)
    )
    ham_mode: str = (
        "per_edge"  # "per_edge" (H_i/d_i, edge average pressure) or "total" (H_i, node total pressure)
    )


class PolarizationVisualizer:
    """
    Polarization evolution visualization manager
    Responsible for running simulation, recording data, and generating visualizations
    """

    def __init__(self, config: Config):
        self.config = config
        self.results_dir = RESULTS_DIR

        # Data recording
        self.steps_history = []
        self.global_psi_history = []
        self.global_ham_history = []
        self.hub_psi_history = []
        self.leaf_psi_history = []
        self.hub_ham_history = []
        self.leaf_ham_history = []

        # Detailed history recording (for video generation)
        # Format: (step, opinions_copy, acceptance_counts_copy, local_ham_copy, local_psi_copy)
        self.simulation_history = []

        # Experiment event points (e.g. phase switching time), for drawing vertical lines
        self.event_steps = []

    def _init_network(self):
        """Initialize network structure"""
        if self.config.network_type == "regular_lattice":
            self.graph = generate_regular_lattice(
                N=self.config.N,
                k=self.config.k_avg,
            )
        elif self.config.network_type == "small_world":
            self.graph = generate_poissonian_small_world(
                N=self.config.N, k_avg=self.config.k_avg, epsilon=self.config.epsilon
            )
        elif self.config.network_type == "scale_free":
            self.graph = generate_stochastic_holme_kim(
                N=self.config.N,
                k_avg=self.config.k_avg,
                p_triangle=self.config.p_triangle,
            )
        elif self.config.network_type == "erdos_renyi":
            self.graph = generate_erdos_renyi(
                N=self.config.N,
                k_avg=self.config.k_avg,
            )
        elif self.config.network_type == "barabasi_albert":
            self.graph = generate_barabasi_albert(
                N=self.config.N,
                k_avg=self.config.k_avg,
            )
        elif self.config.network_type == "ba_no_triangles":
            # First generate original BA network
            G_raw = generate_barabasi_albert(
                N=self.config.N,
                k_avg=self.config.k_avg,
            )
            # Count original triangles
            tri_count_before = sum(nx.triangles(G_raw).values()) // 3
            print(f"Original BA network triangle count: {tri_count_before}")

            # Remove triangles
            print("Removing triangles via rewiring...")
            self.graph = remove_triangles_rewire(G_raw)

            # Count triangles after removal
            tri_count_after = sum(nx.triangles(self.graph).values()) // 3
            print(f"Triangle count after removal: {tri_count_after}")
        elif self.config.network_type == "tree":
            self.graph = generate_random_tree(N=self.config.N)
        else:
            raise ValueError(f"Unknown network type: {self.config.network_type}")

        # Initialize node and edge arrays
        self.N_nodes = self.graph.number_of_nodes()
        self.nodes = np.arange(self.N_nodes)
        self.neighbors = [np.array(list(self.graph.neighbors(i))) for i in self.nodes]

        # Calculate initial layout
        self.pos = nx.spring_layout(self.graph, seed=42, k=1.0 / np.sqrt(self.N_nodes))
        # self.pos = nx.circular_layout(self.graph)

        # Determine Hub and Leaf indices
        self.node_degrees = np.array([self.graph.degree(i) for i in self.nodes])
        # Set threshold, nodes with top hub_fraction degrees are Hubs
        threshold = np.percentile(
            self.node_degrees, 100 * (1 - self.config.hub_fraction)
        )
        # If all nodes have same degree (e.g. some regular graphs), threshold also max, may result in empty hubs
        # Ensure at least some hubs, or if degrees all same then all leaves (or proportional selection)
        if threshold == np.min(self.node_degrees):
            # If network degree distribution extremely concentrated, take last X% of nodes as hubs
            num_hubs = max(1, int(self.N_nodes * self.config.hub_fraction))
            self.hub_indices = np.argsort(self.node_degrees)[-num_hubs:]
        else:
            self.hub_indices = np.where(self.node_degrees >= threshold)[0]

        self.leaf_indices = np.setdiff1d(self.nodes, self.hub_indices)

    def _init_model(self):
        """Initialize binary opinion model"""
        # Use actual node count to initialize model
        self.model = BinaryOpinionModel(
            N=self.N_nodes,
            G_topics=self.config.G_topics,
            alpha=self.config.alpha,
            beta=self.config.beta,
        )

    def run(self):
        # Initialize network and model
        self._init_network()
        self._init_model()

        # Record initial state
        self._record_state(0, self.nodes, self.neighbors)

        for step in tqdm(range(1, self.config.total_steps + 1)):
            # Run one opinion evolution
            self.model.step(self.nodes, self.neighbors)

            # Record current model state
            if step % self.config.steps_per_record == 0:
                self._record_state(step, self.nodes, self.neighbors)

    def _record_state(self, step, nodes, neighbors):
        """Record current model state"""
        # Global and local polarization
        global_psi, local_psi = self.model.calculate_polarization()
        _, ind_ham = self.model.calculate_hamilton(neighbors)

        # Calculate instantaneous degree (handle time-series network snapshot degree zero)
        current_degrees = np.array([len(n) for n in neighbors])

        # Calculate Hamilton-related metrics
        ham_node_vals = np.zeros(self.N_nodes)
        if self.config.ham_mode == "per_edge":
            sum_h_normalized = 0.0
            active_count = 0
            for i in range(self.N_nodes):
                deg = current_degrees[i]
                if deg > 0:
                    h_val = ind_ham[i] / deg
                    ham_node_vals[i] = h_val
                    sum_h_normalized += h_val
                    active_count += 1
            global_ham = sum_h_normalized / active_count if active_count > 0 else 0.0
        else:
            # "total" mode: use ind_ham directly
            ham_node_vals = ind_ham
            global_ham = ind_ham.mean()

        # Save history data
        self.steps_history.append(step)
        self.global_psi_history.append(global_psi)
        self.global_ham_history.append(global_ham)

        # Calculate average metrics for Hubs and Leaves
        hub_psi = local_psi[self.hub_indices].mean()
        leaf_psi = local_psi[self.leaf_indices].mean()
        hub_ham = ham_node_vals[self.hub_indices].mean()
        leaf_ham = ham_node_vals[self.leaf_indices].mean()

        self.hub_psi_history.append(hub_psi)
        self.leaf_psi_history.append(leaf_psi)
        self.hub_ham_history.append(hub_ham)
        self.leaf_ham_history.append(leaf_ham)

        # Save detailed history for visualization
        self.simulation_history.append(
            (
                step,
                self.model.opinions.copy(),
                self.model.acceptance_counts.copy(),
                ham_node_vals.copy(),
                local_psi,
                current_degrees,
            )
        )

    def _get_file_signature(self) -> str:
        """Generate filename signature based on parameters"""
        return f"{self.config.network_type}_N{self.config.N}_k{self.config.k_avg}"

    def _calculate_node_sizes(self, node_degrees):
        """Calculate node sizes for visualization"""
        base_node_size = 3000 / self.N_nodes
        return base_node_size + node_degrees * (30 / np.sqrt(self.N_nodes))

    def _get_global_limits(self):
        """Get fixed color map ranges for polarization (Psi) and pressure (Hamilton)"""
        # Requirement: psi 0-1
        psi_range = (0.0, 1.0)

        # Hamilton range
        if self.config.ham_mode == "per_edge":
            ham_range = (-1.0, 0.0)
        else:
            # "total" mode, Hamilton roughly around [-k_avg*2, 0] or even lower,
            # For meaningful color mapping, take full period quantiles or minimum
            all_ham = np.concatenate([h[3] for h in self.simulation_history])
            ham_range = (np.min(all_ham), 0.0)

        return psi_range, ham_range

    def _get_ham_labels(self):
        """Get corresponding chart titles and Y-axis labels based on ham_mode"""
        if self.config.ham_mode == "per_edge":
            return {
                "net_title": "Avg Edge Social Pressure (H_i/d_i)",
                "curve_title": "Avg Edge Social Pressure Evolution",
                "curve_ylabel": "<H_edge>",
                "bar_title": "Degree vs Avg Social Pressure",
                "bar_ylabel": "Avg H_i/d_i",
                "ylim": (-0.5, 0.0),
            }
        else:
            return {
                "net_title": "Total Node Social Pressure (H_i)",
                "curve_title": "Total Social Pressure Evolution",
                "curve_ylabel": "<H_total>",
                "bar_title": "Degree vs Social Pressure",
                "bar_ylabel": "Avg H_i",
                "ylim": None,
            }

    def _setup_figure(self):
        """Initialize canvas layout"""
        show_network = self.config.N <= 500
        # Reduce figsize for more compact view, avoid elements appearing too small
        figsize = (18, 16) if show_network else (12, 14)

        if show_network:
            fig, axes = plt.subplots(3, 3, figsize=figsize, constrained_layout=True)
        else:
            fig, axes = plt.subplots(3, 2, figsize=figsize, constrained_layout=True)

        def get_ax(row, col_idx_with_net, col_idx_no_net):
            if show_network:
                return axes[row, col_idx_with_net]
            else:
                return axes[row, col_idx_no_net]

        return fig, axes, show_network, get_ax

    def _plot_network_static(self, ax, node_vals, title, cmap, vmin, vmax, node_sizes):
        """
        Plot static network graph with academic quality

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Plot axis
        node_vals : np.ndarray
            Node state values
        title : str
            Figure title
        cmap : str
            Color mapping
        vmin, vmax : float
            Color mapping range
        node_sizes : np.ndarray
            Node sizes
        """
        if ax is None:
            return
        ax.clear()
        ax.set_title(title, fontsize=12, fontweight="bold")
        nx.draw_networkx_edges(
            self.graph, self.pos, ax=ax, alpha=0.1, edge_color="lightgray"
        )
        nx.draw_networkx_nodes(
            self.graph,
            self.pos,
            ax=ax,
            node_size=node_sizes,
            node_color=node_vals,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            edgecolors="black",
            linewidths=0.2,
        )
        ax.axis("off")
        # Compact layout, reduce white space around edges
        ax.set_xlim(-1.02, 1.02)
        ax.set_ylim(-1.02, 1.02)

    def _plot_evolution_curve(self, ax, history_dict, current_step, title, ylabel):
        """Plot evolution curve, supporting multiple lines"""
        ax.clear()
        ax.set_title(title, fontsize=12, fontweight="bold")

        # Find index of current step
        try:
            idx = self.steps_history.index(current_step)
        except ValueError:
            idx = 0

        for label, val_tuple in history_dict.items():
            # Support (data, color) or (data, color, linestyle)
            if len(val_tuple) == 3:
                data, color, linestyle = val_tuple
            else:
                data, color = val_tuple
                linestyle = "-"

            # Background light line (full period)
            ax.plot(
                self.steps_history,
                data,
                color=color,
                linestyle=linestyle,
                alpha=0.1,
                label="_nolegend_",
            )
            # Evolution line up to current step
            ax.plot(
                self.steps_history[: idx + 1],
                data[: idx + 1],
                color=color,
                linestyle=linestyle,
                linewidth=2,
                label=label,
            )
            # Current position point
            current_val = data[idx]
            ax.scatter([current_step], [current_val], c=color, s=30, zorder=5)

        # Draw event vertical lines
        if hasattr(self, "event_steps"):
            for ev_step in self.event_steps:
                ax.axvline(
                    x=ev_step, color="gray", linestyle="--", linewidth=1.5, alpha=0.6
                )

        ax.set_xlim(0, self.config.total_steps)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.8)
        ax.grid(True, alpha=0.3)

    def _plot_degree_bar(self, ax, vals, node_degrees, color, title, ylabel, ylim=None):
        """Plot bar chart based on degree (supports log-axis visual equal width)"""
        ax.clear()
        unique_degs = np.unique(node_degrees)
        avg_vals = np.array([vals[node_degrees == d].mean() for d in unique_degs])

        # In log coordinate, to make bar visually equal width, width must scale with x
        # width = 0.08 * plot_x represents width about 8% of current scale
        plot_x = unique_degs
        widths = 0.08 * plot_x

        ax.bar(
            plot_x,
            avg_vals,
            width=widths,
            color=color,
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_xscale("log")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)

        if ylim is not None:
            ax.set_ylim(ylim)

        ax.grid(True, alpha=0.2, which="both")  # Show grid for log scale specifically

    def _plot_opinion_distribution(self, ax, opinions, title):
        """
        Plot distribution histogram of all 2^G opinion vectors, distinguishing Hub and Leaf node contributions
        opinions: (N, G)
        """
        ax.clear()

        N, G = opinions.shape
        num_states = 2**G

        # Convert (-1, 1) to (0, 1), then view as binary number to convert to decimal index
        op_binary = (opinions > 0).astype(int)

        # Calculate integer index for each row (binary to decimal)
        powers = 1 << np.arange(G - 1, -1, -1)
        indices = op_binary @ powers

        # Count frequencies separately for Hub and Leaf
        hub_indices = self.hub_indices
        leaf_indices = self.leaf_indices

        counts_hub = np.bincount(indices[hub_indices], minlength=num_states)
        counts_leaf = np.bincount(indices[leaf_indices], minlength=num_states)

        xs = np.arange(num_states)
        # Use original counts, no longer normalize to frequency
        ys_hub = counts_hub
        ys_leaf = counts_leaf

        # Plot stacked bar chart: bottom layer Leaf, top layer Hub
        # Use blue/red color scheme: Leaf=blue, Hub=red
        ax.bar(
            xs,
            ys_leaf,
            color="#3498db",
            alpha=0.7,
            edgecolor="none",
            width=1.0,
            label="Leaf Nodes",
        )
        ax.bar(
            xs,
            ys_hub,
            bottom=ys_leaf,
            color="#e74c3c",
            alpha=0.9,
            edgecolor="none",
            width=1.0,
            label="Hub Nodes",
        )

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(
            f"Opinion State (Total {num_states} states, sorted by binary value)",
            fontsize=10,
        )
        ax.set_ylabel("Count", fontsize=10)

        # Hide X-axis ticks, too dense
        ax.set_xticks([])

        # Set X-axis range
        ax.set_xlim(0, num_states)
        # ax.set_ylim(0, 1)  # Remove fixed range for automatic scaling based on count

        # Add legend to distinguish Hub and Leaf
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    def _plot_dynamic_network(self, ax, opinions, node_sizes, current_pos=None):
        """Plot dynamic opinion network (if current_pos provided mainly for video, else final frame)"""
        if ax is None:
            return

        ax.clear()
        ax.set_title("Dynamic Opinion Network", fontsize=12, fontweight="bold")

        # 1. Calculate edge weights and colors
        weights = {}
        edge_vals = []
        for u, v in self.graph.edges():
            dot = np.dot(opinions[u], opinions[v]) / self.config.G_topics
            weights[(u, v)] = 1.0 + dot
            edge_vals.append(dot)

        nx.set_edge_attributes(self.graph, weights, "weight")

        # Calculate mean opinions for node colors and layout forces
        mean_opinions = np.mean(opinions, axis=1)

        if self.config.dynamic_layout_type == "circular":
            new_pos = nx.circular_layout(self.graph)
        else:
            # 2. Layout update
            # If current_pos provided, do incremental update based on it (video mode)
            # Otherwise compute from scratch (snapshot mode)
            iterations = 5 if current_pos is not None else 100
            pos_base = current_pos if current_pos is not None else None

            new_pos = nx.spring_layout(
                self.graph,
                pos=pos_base,
                iterations=iterations,
                k=1.2 / np.sqrt(self.N_nodes),
                weight="weight",
                threshold=0.0001,
            )

            # 3. Apply left-right separation force (visualization enhancement)
            # Video mode higher inertia (0.9 retention), snapshot mode lower (0.8 retention)
            retention = 0.9 if current_pos is not None else 0.8
            force = 1.0 - retention

            for i in self.nodes:
                target_x = 0.6 if mean_opinions[i] > 0 else -0.6
                new_pos[i][0] = new_pos[i][0] * retention + target_x * force
                # Also add slight centripetal force to prevent overall drift
                new_pos[i][1] *= 0.95

            # 4. Position normalization: prevent outliers from escaping and shrinking overall view
            # Convert all coordinates to array for batch operations
            pos_array = np.array([new_pos[i] for i in self.nodes])

            # Center adjustment: subtract mean to keep network at canvas center
            pos_array -= pos_array.mean(axis=0)

            # Scale control: ensure 95% of points in [-0.9, 0.9], avoid extreme outliers affecting scale
            curr_scale = np.percentile(np.abs(pos_array), 95)
            if curr_scale > 0:
                pos_array = pos_array * (0.8 / curr_scale)

            # Force clipping: force rare escaping points back to boundary
            pos_array = np.clip(pos_array, -1.0, 1.0)

            # Write back to layout dictionary
            for i in self.nodes:
                new_pos[i] = pos_array[i]

        # 5. Plot
        nx.draw_networkx_edges(
            self.graph,
            new_pos,
            ax=ax,
            edge_color=edge_vals,
            edge_cmap=plt.cm.RdYlGn,
            edge_vmin=-1.0,
            edge_vmax=1.0,
            alpha=0.6,
            width=0.8,
        )
        nx.draw_networkx_nodes(
            self.graph,
            new_pos,
            ax=ax,
            node_size=node_sizes,
            node_color=mean_opinions,
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
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

        return new_pos

    def generate_video_full(self, video=True, frame=True):
        """
        Generate complete visualization results (original generate_video 3x3 layout)
        Replace Alignment Hist with Opinion Vector Distribution
        """
        if not (video or frame):
            return

        print(f"Starting Full Visualization generation (Video={video}, Frame={frame}) ...")

        # Prepare data (limits)
        (psi_vmin, psi_vmax), (ham_vmin, ham_vmax) = self._get_global_limits()

        # Initialize canvas
        fig, axes, show_network, get_ax = self._setup_figure()

        # Map subplots
        # Row 1
        ax_net_psi = axes[0, 0] if show_network else None
        ax_curve_psi = get_ax(0, 1, 0)
        ax_hist_psi = get_ax(0, 2, 1)

        # Row 2
        ax_net_ham = axes[1, 0] if show_network else None
        ax_curve_ham = get_ax(1, 1, 0)
        ax_hist_ham = get_ax(1, 2, 1)

        # Row 3
        ax_dyn_net = axes[2, 0] if show_network else None
        ax_op_dist = get_ax(2, 1, 0)  # Replace original Alignment Hist with Opinion Dist
        ax_accept = get_ax(2, 2, 1)

        # Dynamic layout state preservation (video only)
        current_pos = self.pos.copy()

        # Get Hamilton-related parameters
        ham_labels = self._get_ham_labels()
        ham_ylim = ham_labels["ylim"]

        # Acceptance Count range
        acc_ylim = (0, 10)
        if self.simulation_history:
            last_degrees = self.simulation_history[-1][5]
            last_acceptance = self.simulation_history[-1][2]
            unique_degs = np.unique(last_degrees)
            max_avg_acc = 0
            for d in unique_degs:
                avg = last_acceptance[last_degrees == d].mean()
                if avg > max_avg_acc:
                    max_avg_acc = avg
            acc_ylim = (0, max_avg_acc * 1.1)

        def update(frame_idx):
            nonlocal current_pos

            # Unpack data
            step, opinions, acceptance, ham_vals, local_psi, current_degrees = (
                self.simulation_history[frame_idx]
            )

            # Dynamically calculate node sizes
            node_sizes = self._calculate_node_sizes(current_degrees)

            # --- Row 1 & 2 ---
            self._plot_network_static(
                ax_net_psi,
                local_psi,
                f"Local Polarization (Step {step})",
                "Reds",
                psi_vmin,
                psi_vmax,
                node_sizes,
            )
            self._plot_network_static(
                ax_net_ham,
                ham_vals,
                ham_labels["net_title"],
                "Blues_r",
                ham_vmin,
                ham_vmax,
                node_sizes,
            )

            # Polarization evolution
            hub_pct = int(self.config.hub_fraction * 100)
            leaf_pct = 100 - hub_pct
            psi_data_dict = {
                "Global": (self.global_psi_history, "black"),
                "Hubs ({}%)".format(hub_pct): (self.hub_psi_history, "red"),
                "Leaves ({}%)".format(leaf_pct): (self.leaf_psi_history, "blue"),
            }
            self._plot_evolution_curve(
                ax_curve_psi,
                psi_data_dict,
                step,
                (
                    "Polarization Evolution"
                    if show_network
                    else f"Polarization Evolution (Step {step})"
                ),
                "Psi",
            )

            # Hamilton evolution
            ham_data_dict = {
                "Global": (self.global_ham_history, "black"),
                "Hubs ({}%)".format(hub_pct): (self.hub_ham_history, "red"),
                "Leaves ({}%)".format(leaf_pct): (self.leaf_ham_history, "blue"),
            }
            self._plot_evolution_curve(
                ax_curve_ham,
                ham_data_dict,
                step,
                ham_labels["curve_title"],
                ham_labels["curve_ylabel"],
            )

            self._plot_degree_bar(
                ax_hist_psi,
                local_psi,
                current_degrees,
                "blue",
                "Degree vs Pol",
                "Avg Pol",
            )
            self._plot_degree_bar(
                ax_hist_ham,
                ham_vals,
                current_degrees,
                "green",
                ham_labels["bar_title"],
                ham_labels["bar_ylabel"],
                ylim=ham_ylim,
            )

            # --- Row 3 ---
            if show_network and ax_dyn_net:
                # Video mode incremental position update
                current_pos = self._plot_dynamic_network(
                    ax_dyn_net, opinions, node_sizes, current_pos
                )

            # Replace with Opinion Distribution
            self._plot_opinion_distribution(
                ax_op_dist,
                opinions,
                "Opinion Vector Distribution",
            )

            self._plot_degree_bar(
                ax_accept,
                acceptance,
                current_degrees,
                "orange",
                "Acceptance vs Degree",
                "Avg Acceptances",
                ylim=acc_ylim,
            )

            return axes.flatten()

        file_sig = self._get_file_signature()

        # 1. Generate final frame static plot
        if frame and self.simulation_history:
            # Re-plot Dynamic Network separately for better layout (iter=100)
            update(len(self.simulation_history) - 1)

            if show_network and ax_dyn_net:
                self._plot_dynamic_network(
                    ax_dyn_net,
                    self.simulation_history[-1][1],
                    self._calculate_node_sizes(self.simulation_history[-1][5]),
                    current_pos=None,
                )

            output_path = self.results_dir / f"last_frame_{file_sig}.png"
            try:
                plt.savefig(output_path, dpi=150)
                print(f"Full Last Frame saved to: {output_path}")
            except Exception as e:
                print(f"Error saving Frame: {e}")

        # 2. Generate video
        if video and self.simulation_history:
            current_pos = self.pos.copy()

            anim = animation.FuncAnimation(
                fig,
                update,
                frames=len(self.simulation_history),
                interval=1000 / self.config.fps,
                blit=False,
            )

            writer = animation.FFMpegWriter(
                fps=self.config.fps,
                metadata=dict(artist="Polarization Visualizer Full"),
                bitrate=3000,
                codec="libx264",
                extra_args=["-preset", "ultrafast", "-crf", "28"],
            )

            output_path = self.results_dir / f"evolution_{file_sig}.mp4"
            try:
                print(f"Saving Full Video to {output_path} ...")
                anim.save(output_path, writer=writer, dpi=100)
                print("Full Video saved successfully.")
            except Exception as e:
                print(f"Error saving Video: {e}")

        plt.close(fig)

    def generate_video_core(self, video=True, frame=True):
        """
        Generate core visualization results (2x2)
        Corresponds to original generate_video2 (2x2 layout), Opinion Distribution replaces Triangle Counts
        """
        if not (video or frame):
            return

        print(f"Starting Core Visualization generation (Video={video}, Frame={frame}) ...")

        # Set up 2x2 canvas
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
        ax_ham_time = axes[0, 0]
        ax_ham_deg = axes[0, 1]
        ax_op_dist = axes[1, 0]
        ax_acc_deg = axes[1, 1]

        # Get Hamilton-related parameters
        ham_labels = self._get_ham_labels()
        ham_ylim = ham_labels["ylim"]

        # Acceptance Count range
        acc_ylim = (0, 10)
        if self.simulation_history:
            last_degrees = self.simulation_history[-1][5]
            last_acceptance = self.simulation_history[-1][2]
            unique_degs = np.unique(last_degrees)
            max_avg_acc = 0
            for d in unique_degs:
                avg = last_acceptance[last_degrees == d].mean()
                if avg > max_avg_acc:
                    max_avg_acc = avg
            acc_ylim = (0, max_avg_acc * 1.1)

        def update(frame_idx):
            # Get data
            step, opinions, acceptance, ham_vals, local_psi, current_degrees = (
                self.simulation_history[frame_idx]
            )

            # 1. System Hamilton changes over time (Top-Left)
            hub_pct = int(self.config.hub_fraction * 100)
            leaf_pct = 100 - hub_pct

            ham_data = {
                "Global": (self.global_ham_history, "black"),
                "Hubs ({}%)".format(hub_pct): (self.hub_ham_history, "red"),
                "Leaves ({}%)".format(leaf_pct): (self.leaf_ham_history, "blue"),
            }
            self._plot_evolution_curve(
                ax_ham_time,
                ham_data,
                step,
                ham_labels["curve_title"],
                ham_labels["curve_ylabel"],
            )

            # 2. Hamilton for different degree nodes (Top-Right)
            self._plot_degree_bar(
                ax_ham_deg,
                ham_vals,
                current_degrees,
                "green",
                ham_labels["bar_title"],
                ham_labels["bar_ylabel"],
                ylim=ham_ylim,
            )

            # 3. Opinion Vector Distribution (Bottom-Left)
            self._plot_opinion_distribution(
                ax_op_dist, opinions, "Opinion Vector Distribution"
            )

            # 4. Acceptance count for different degree nodes (Bottom-Right)
            self._plot_degree_bar(
                ax_acc_deg,
                acceptance,
                current_degrees,
                "orange",
                "Acceptance Count by Degree",
                "Avg Accept Count",
                ylim=acc_ylim,
            )

            return axes.flatten()

        file_sig = self._get_file_signature()

        # 1. Generate final frame static plot
        if frame and self.simulation_history:
            update(len(self.simulation_history) - 1)
            output_path = self.results_dir / f"last_frame_core_{file_sig}.png"
            try:
                plt.savefig(output_path, dpi=150)
                print(f"Core Last Frame saved to: {output_path}")
            except Exception as e:
                print(f"Error saving Core Frame: {e}")

        # 2. Generate video
        if video and self.simulation_history:
            anim = animation.FuncAnimation(
                fig,
                update,
                frames=len(self.simulation_history),
                interval=1000 / self.config.fps,
                blit=False,
            )

            writer = animation.FFMpegWriter(
                fps=self.config.fps,
                metadata=dict(artist="Polarization Visualizer Core"),
                bitrate=3000,
                codec="libx264",
                extra_args=["-preset", "ultrafast", "-crf", "28"],
            )

            output_path = self.results_dir / f"video_core_{file_sig}.mp4"
            try:
                print(f"Saving Core Video to {output_path} ...")
                anim.save(output_path, writer=writer, dpi=100)
                print("Core Video saved successfully.")
            except Exception as e:
                print(f"Error saving Core Video: {e}")

        plt.close(fig)


if __name__ == "__main__":
    # Example running configuration
    config = Config(
        # Network parameters
        network_type="ba_no_triangles",  # "small_world", "scale_free", "erdos_renyi", "barabasi_albert", "regular_lattice", "tree", "ba_no_triangles"
        N=500,
        k_avg=12,
        epsilon=0.175,  # Rewiring probability (small-world network)
        p_triangle=0.65,  # Triangle probability (scale-free network)
        hub_fraction=0.1,  # Hub node fraction
        G_topics=9,
        # Simulation parameters
        total_steps=int(7e5),
        steps_per_record=int(5e3),
        # Visualization parameters
        fps=20,
        ham_mode="per_edge",  # Individual Hamilton, "total" or "per_edge"
        # dynamic_layout_type="circular",  # Opinion similarity network layout force_directed, circular
    )

    viz = PolarizationVisualizer(config)

    # Run simulation
    viz.run()

    # Generate complete results (Video + Frame)
    viz.generate_video_full(video=False, frame=True)

    # Generate core results (Video + Frame)
    # viz.generate_video_core(video=True, frame=True)
