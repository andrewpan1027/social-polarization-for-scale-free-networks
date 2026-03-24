import random

import networkx as nx
import numpy as np


def generate_poissonian_small_world(N: int, k_avg: float, epsilon: float = 0.175):
    """
    Generate Poissonian small-world network using ring construction combined with random rewiring.

    Parameters
    ----------
    N : int
        Number of nodes
    k_avg : float
        Expected average degree (mean of Poisson distribution)
    epsilon : float
        Small-world rewiring probability for each edge

    Returns
    -------
    G : nx.Graph
        Generated undirected graph
    """
    # 1. Generate target degrees (Poisson distribution)
    desired_degrees = np.random.poisson(k_avg, N)

    # 2. Build initial ring network (Poissonian Ring)
    G = nx.Graph()
    G.add_nodes_from(range(N))

    # Shuffle node order to avoid systematic bias
    nodes_order = np.random.permutation(N)

    for i in nodes_order:
        d = int(desired_degrees[i])
        # If target degree is 0, skip directly
        if d <= 0:
            continue

        k = 1
        # Use while loop to extend distance on ring
        while G.degree[i] < d and k <= N / 2:
            neighbor_right = (i + k) % N
            neighbor_left = (i - k) % N

            # Try to connect right neighbor
            if (
                G.degree[i] < d
                and G.degree[neighbor_right] < desired_degrees[neighbor_right]
                and not G.has_edge(i, neighbor_right)
            ):
                G.add_edge(i, neighbor_right)

            # Try to connect left neighbor
            if (
                G.degree[i] < d
                and G.degree[neighbor_left] < desired_degrees[neighbor_left]
                and not G.has_edge(i, neighbor_left)
            ):
                G.add_edge(i, neighbor_left)

            k += 1

    # 3. Small-world rewiring
    edges = list(G.edges())
    for u, v in edges:
        if np.random.random() < epsilon:
            # Try to select a new endpoint w
            # To avoid infinite loops, attempt at most a certain number of times
            for _ in range(10 * N):
                w = np.random.randint(0, N)
                if w != u and not G.has_edge(u, w):
                    # Perform rewiring: remove (u, v), add (u, w)
                    G.remove_edge(u, v)
                    G.add_edge(u, w)
                    break

    # 4. Return largest connected component and relabel nodes as 0..n-1 to avoid isolated node index mismatch
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")

    return G


def generate_holme_kim(N: int, k_avg: float, p_triangle: float = 0.5):
    """
    Generate scale-free network based on Holme-Kim algorithm (Scale-free network with high clustering).
    Uses networkx's powerlaw_cluster_graph which implements the Holme-Kim model.

    Parameters
    ----------
    N : int
        Number of nodes
    k_avg : float
        Expected average degree (target value)
    p_triangle : float
        Triangle formation probability (default 0.5)

    Returns
    -------
    G : nx.Graph
        Generated undirected simple graph (largest connected component)
    """
    # Calculate parameter m: for Holme-Kim model, average degree ⟨k⟩ ≈ 2m
    m = int(round(k_avg / 2))
    if m < 1:
        m = 1  # Ensure m >= 1

    # Use powerlaw_cluster_graph to generate network (Holme-Kim model)
    G = nx.powerlaw_cluster_graph(n=N, m=m, p=p_triangle)

    # Convert to undirected simple graph (remove multi-edges and self-loops)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))

    # Return largest connected component (LCC) and relabel nodes as 0..n-1
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")

    return G


def generate_stochastic_holme_kim(N: int, k_avg: float, p_triangle: float = 0.5):
    """
    Generate Stochastic Holme-Kim network that supports continuous average degree.

    Implements continuous average degree through mixing parameters: the number of edges added per step
    is randomly chosen between m_low and m_low+1, such that the expected average degree equals k_avg.

    Parameters
    ----------
    N : int
        Number of nodes
    k_avg : float
        Target average degree (can be a continuous value)
    p_triangle : float
        Triangle formation probability

    Returns
    -------
    G : nx.Graph
        Generated undirected graph (largest connected component)
    """
    # Calculate mixing parameters
    mu = k_avg / 2  # Target expected edges added per step
    m_low = int(np.floor(mu))
    p_extra = mu - m_low  # Probability of adding extra edge

    # Ensure m_low >= 1
    if m_low < 1:
        m_low = 1
        p_extra = mu - 1

    # Initialization: create complete graph with m0 nodes (seed graph)
    # m0 should be at least m_low + 1 to ensure triangle formation is possible
    m0 = max(m_low + 1, 3)  # At least 3 nodes to form a triangle

    if N <= m0:
        # If N is too small, directly return complete graph
        G = nx.complete_graph(N)
        return G

    G = nx.complete_graph(m0)

    # Performance optimization: maintain degree array and total degree to avoid repeated computation
    # Use array indices corresponding to node numbers (node numbers from 0 to N-1)
    degrees = np.zeros(N)
    for node in range(m0):
        degrees[node] = m0 - 1  # Each node in complete graph has degree m0-1
    total_degree = m0 * (m0 - 1)  # Total degree is 2 * number of edges

    # Growth loop: from m0 to N-1
    for i in range(m0, N):
        # Determine number of edges m_curr for current node
        if np.random.random() < p_extra:
            m_curr = m_low + 1
        else:
            m_curr = m_low

        # Ensure m_curr does not exceed number of existing nodes
        m_curr = min(m_curr, i)

        if m_curr <= 0:
            continue

        # Use set to track connected nodes, avoid repeated checks
        connected_nodes = set()

        # First edge: select neighbor u by degree probability (preferential attachment)
        if total_degree == 0:
            # If all nodes have degree 0, select randomly
            u = np.random.randint(0, i)
        else:
            # Use cumulative weights for weighted random selection (more efficient)
            # Only consider existing nodes (0 to i-1)
            existing_degrees = degrees[:i]
            cumsum = np.cumsum(existing_degrees)
            if cumsum[-1] == 0:
                u = np.random.randint(0, i)
            else:
                r = np.random.random() * cumsum[-1]
                u = np.searchsorted(cumsum, r)

        # Add first edge
        G.add_edge(i, u)
        degrees[i] += 1
        degrees[u] += 1
        total_degree += 2
        connected_nodes.add(u)
        edges_added = 1

        # Subsequent edges (m_curr - 1 edges)
        while edges_added < m_curr:
            if np.random.random() < p_triangle:
                # Try triangle closure: connect to a neighbor of u
                # Use graph adjacency list, but only check once
                neighbors_of_u = list(G.neighbors(u))
                # Exclude already connected nodes and self
                available_neighbors = [
                    n for n in neighbors_of_u if n != i and n not in connected_nodes
                ]

                if len(available_neighbors) > 0:
                    # Randomly select an available neighbor
                    v = np.random.choice(available_neighbors)
                    G.add_edge(i, v)
                    degrees[i] += 1
                    degrees[v] += 1
                    total_degree += 2
                    connected_nodes.add(v)
                    edges_added += 1
                    # u remains unchanged for possible subsequent triangle closures
                    continue

            # Otherwise, select other nodes by degree probability (preferential attachment)
            # Get all unconnected nodes (use set difference, more efficient)
            unconnected_mask = np.ones(i, dtype=bool)
            unconnected_mask[list(connected_nodes)] = False

            if not np.any(unconnected_mask):
                # If all nodes are already connected, stop
                break

            # Calculate degree weights for unconnected nodes
            unconnected_indices = np.where(unconnected_mask)[0]
            unconnected_degrees = degrees[unconnected_indices]
            total_unconnected_degree = np.sum(unconnected_degrees)

            if total_unconnected_degree == 0:
                # If all unconnected nodes have degree 0, select randomly
                v = np.random.choice(unconnected_indices)
            else:
                # Use cumulative weights for weighted random selection
                cumsum = np.cumsum(unconnected_degrees)
                r = np.random.random() * total_unconnected_degree
                idx = np.searchsorted(cumsum, r)
                v = unconnected_indices[idx]

            G.add_edge(i, v)
            degrees[i] += 1
            degrees[v] += 1
            total_degree += 2
            connected_nodes.add(v)
            edges_added += 1
            # Update u to newly connected node for subsequent triangle closures
            u = v

    # Return largest connected component (LCC) and relabel nodes as 0..n-1
    if not nx.is_connected(G) and G.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")

    return G


def generate_regular_lattice(N: int, k: int):
    """
    Generate regular lattice network (Regular Ring Lattice).
    This is the baseline state (p=0) of the Watts-Strogatz model.
    Each node connects to its k nearest neighbors (k must be even).

    Parameters
    ----------
    N : int
        Number of nodes
    k : int
        Degree of each node (if odd, will be forced to even to maintain symmetry)

    Returns
    -------
    G : nx.Graph
        Generated undirected lattice graph (largest connected component)
    """
    if k % 2 != 0:
        k = k - 1 if k > 1 else 2
        print(f"Warning: Regular lattice requires even k. Adjusted to {k}.")

    # Use Watts-Strogatz model with rewiring probability p=0
    G = nx.watts_strogatz_graph(n=N, k=k, p=0)

    # Return largest connected component and relabel nodes as 0..n-1
    if not nx.is_connected(G) and G.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")

    return G


def generate_erdos_renyi(N: int, k_avg: float):
    """
    Generate Erdős-Rényi random network.

    Parameters
    ----------
    N : int
        Number of nodes
    k_avg : float
        Expected average degree

    Returns
    -------
    G : nx.Graph
        Generated undirected graph (largest connected component)
    """
    p = k_avg / (N - 1) if N > 1 else 0
    G = nx.erdos_renyi_graph(n=N, p=p)

    # Extract LCC and relabel nodes
    if not nx.is_connected(G) and G.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")

    return G


def generate_barabasi_albert(N: int, k_avg: float):
    """
    Generate BA scale-free network (Barabási-Albert).

    Parameters
    ----------
    N : int
        Number of nodes
    k_avg : float
        Expected average degree (k_avg ≈ 2m)

    Returns
    -------
    G : nx.Graph
        Generated undirected graph (largest connected component)
    """
    m = int(round(k_avg / 2))
    if m < 1:
        m = 1

    if N <= m:
        return nx.complete_graph(N)

    G = nx.barabasi_albert_graph(n=N, m=m)

    # Extract LCC and relabel nodes
    if not nx.is_connected(G) and G.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")

    return G


def generate_random_tree(N: int):
    """
    Generate random tree network (no triangles).
    Note: Tree average degree ⟨k⟩ = 2(N-1)/N ≈ 2, cannot be adjusted by parameters.

    Parameters
    ----------
    N : int
        Number of nodes

    Returns
    -------
    G : nx.Graph
        Generated tree (undirected graph)
    """
    G = nx.random_labeled_tree(n=N, seed=None)
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    return G


class ActivityDrivenNetwork:
    def __init__(self, N, m, gamma, epsilon=0.001, seed=None):
        """
        Initialize Activity Driven Network model parameters.
        Snapshot network active nodes = eta * <x> * N
        Snapshot network average degree = 2 * eta * <x> * m
        Static network average degree = 2 * eta * <x> * m * T

        Parameters
        ----------
        N : int
            Number of nodes.
        m : int
            Number of links generated by each active node.
        gamma : float
            Power-law exponent of activity potential distribution.
        epsilon : float
            Lower bound of activity potential to avoid divergence.
        seed : int
            Random seed for reproducibility.
        """
        self.N = N
        self.m = m
        self.gamma = gamma
        self.epsilon = epsilon

        if seed is not None:
            np.random.seed(seed)

        # 1. Allocate activity potentials x_i according to F(x) ~ x^(-gamma)
        # Use inverse transform sampling to generate power-law distribution.
        # Range: [epsilon, 1]
        # PDF: P(x) = C * x^(-gamma)
        # Inverse CDF derived from P(x).

        # Generate uniform random numbers u in [0, 1]
        u = np.random.uniform(0, 1, N)

        # Inverse CDF formula for power-law distribution restricted to [epsilon, 1]:
        # x = [ (1^(1-gamma) - eps^(1-gamma)) * u + eps^(1-gamma) ] ^ (1/(1-gamma))
        exp = 1.0 - self.gamma
        term1 = 1.0**exp
        term2 = self.epsilon**exp

        self.activity_potentials = ((term1 - term2) * u + term2) ** (1.0 / exp)

    def generate_snapshot(self, eta=None, n_active=None, p_active=None, dt=1):
        """
        Generate snapshot network according to different input modes.

        Parameters
        ----------
        eta : float
            Mode 1, activity rate scaling factor. If provided, sample by probability distribution.
        n_active : int
            Mode 2, fixed number of active nodes. If provided, extract fixed number by potential weights.
        p_active : float
            Mode 3, fraction of active nodes. If provided, automatically convert to n_active = p_active * N.
        dt : float
            Time step, only valid when eta is provided.

        Returns
        -------
        nx.Graph
            Snapshot undirected network G_t.
        """
        G_t = nx.Graph()
        G_t.add_nodes_from(range(self.N))

        if p_active is not None:
            n_active = int(p_active * self.N)

        if eta is not None:
            # Mode 1: probability sampling based on eta
            activation_probs = eta * self.activity_potentials * dt
            active_mask = np.random.rand(self.N) < activation_probs
            active_nodes = np.where(active_mask)[0]
        elif n_active is not None:
            # Mode 2 & 3: extract fixed number of nodes by potential weights
            probs = self.activity_potentials / np.sum(self.activity_potentials)
            active_nodes = np.random.choice(
                range(self.N), size=n_active, replace=False, p=probs
            )
        else:
            raise ValueError("Must provide one of eta, n_active, or p_active to generate snapshot.")

        self.active_nodes = active_nodes

        edges = []
        for node in active_nodes:
            # Active node generates m links
            possible_targets = np.random.randint(0, self.N, size=self.m)
            for target in possible_targets:
                if node != target:  # Avoid self-loops
                    edges.append((node, target))

        G_t.add_edges_from(edges)
        return G_t


def get_configuration_null_model(G_original):
    """
    Generate Configuration Model (null model) based on original network.
    Perfectly preserves degree distribution but eliminates clustering.

    Parameters
    ----------
    G_original : nx.Graph
        Original network (e.g., HK network)

    Returns
    -------
    G_config : nx.Graph
        Configuration model network
    """
    # 1. Extract degree sequence of original network
    degree_seq = [d for n, d in G_original.degree()]

    # 2. Generate configuration model
    # nx.configuration_model generates a multigraph (with self-loops and multi-edges)
    # create_using=nx.Graph() automatically merges multi-edges, but does not remove self-loops
    G_config = nx.configuration_model(degree_seq, create_using=nx.Graph())
    G_config.remove_edges_from(nx.selfloop_edges(G_config))

    # Note: After conversion to simple graph, due to removal of multi-edges and self-loops,
    # average degree <k> slightly decreases. This decrease is usually negligible in sparse networks (loss < 1-2%).
    # To preserve both degree distribution and edge count strictly, consider using "Double Edge Swap" algorithm

    # Return largest connected component and relabel nodes as 0..n-1
    if not nx.is_connected(G_config) and G_config.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(G_config), key=len)
        G_config = G_config.subgraph(largest_cc).copy()
    G_config = nx.convert_node_labels_to_integers(G_config, ordering="sorted")

    return G_config


def remove_triangles_rewire(G, max_iter=100000):
    """
    Remove triangles from network through rewiring while trying to preserve node degrees.

    Parameters
    ----------
    G : nx.Graph
        Input graph
    max_iter : int
        Maximum number of iterations

    Returns
    -------
    G : nx.Graph
        Graph with triangles removed
    """
    G = G.copy()

    for _ in range(max_iter):
        tri = nx.triangles(G)
        tri_nodes = [n for n, t in tri.items() if t > 0]

        if not tri_nodes:
            break

        u = random.choice(tri_nodes)
        neighbors = list(G.neighbors(u))

        if len(neighbors) < 2:
            continue

        v, w = random.sample(neighbors, 2)

        if not G.has_edge(v, w):
            continue

        # Remove triangle edge
        G.remove_edge(v, w)

        # Find new connection point
        candidates = list(set(G.nodes()) - set(G.neighbors(v)) - {v})

        if candidates:
            x = random.choice(candidates)
            G.add_edge(v, x)
        else:
            G.add_edge(v, w)

    return G


def _compute_powerlaw_exponent(degrees: np.ndarray, k_min: int = 5):
    """
    Compute power-law exponent of degree distribution (maximum likelihood estimation based on tail).

    Parameters
    ----------
    degrees : np.ndarray
        Array of node degrees
    k_min : int
        Minimum degree threshold for estimation (default 5)

    Returns
    -------
    float | None
        Power-law exponent, or None if insufficient data
    """
    tail = degrees[degrees >= k_min]
    if len(tail) > 0:
        alpha = 1 + len(tail) / np.sum(np.log(tail / (k_min - 0.5)))
        return float(alpha)
    return None


def _print_network_statistics(
    graph: nx.Graph,
    network_name: str,
    target_k_avg: float,
    epsilon: float | None = None,
    p_triangle: float | None = None,
):
    """
    Print network statistics in unified format.

    Parameters
    ----------
    graph : nx.Graph
        Network graph object
    network_name : str
        Name of network type
    target_k_avg : float
        Target average degree
    epsilon : float | None
        Small-world rewiring probability (only for Poissonian small-world network)
    p_triangle : float | None
        Triangle formation probability (for Holme-Kim network)
    """
    degrees = np.array([d for _, d in graph.degree()])
    actual_k_avg = np.mean(degrees) if len(degrees) > 0 else 0.0
    clustering = nx.average_clustering(graph)
    powerlaw_exp = _compute_powerlaw_exponent(degrees)

    print(f"\n{'=' * 60}")
    print(f"Testing {network_name}")
    print(f"{'=' * 60}")
    print(f"Parameters:")
    print(f"  N = {graph.number_of_nodes()}")
    print(f"  Target ⟨k⟩ = {target_k_avg:.2f}")
    if epsilon is not None:
        print(f"  ε (rewiring probability) = {epsilon:.3f}")
    if p_triangle is not None:
        print(f"  p_triangle = {p_triangle:.3f}")
    print(f"\nNetwork Statistics:")
    print(f"  Number of nodes: {graph.number_of_nodes()}")
    print(f"  Number of edges: {graph.number_of_edges()}")
    print(f"  Is connected: {nx.is_connected(graph)}")
    print(f"  Actual ⟨k⟩: {actual_k_avg:.3f}")
    print(f"  Average clustering coefficient C: {clustering:.4f}")
    if powerlaw_exp is not None:
        print(f"  Power-law exponent (estimated): {powerlaw_exp:.3f}")
    else:
        print(f"  Power-law exponent (estimated): N/A (insufficient data)")


def visualize_degree_distribution(G, title="Degree Distribution", gamma=None):
    """
    Analyze and display degree distribution of arbitrary network (linear and logarithmic scales).

    Parameters
    ----------
    G : nx.Graph
        Input network graph.
    title : str
        Figure title.
    gamma : float, optional
        Exponent for drawing power-law reference line.
    """
    degrees = [d for n, d in G.degree()]
    N = G.number_of_nodes()

    # Plot layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Left plot: linear scale histogram ---
    ax1.hist(degrees, bins=30, alpha=0.6, color="skyblue", density=True)
    ax1.set_title(f"{title} (Linear Scale)")
    ax1.set_xlabel("Degree")
    ax1.set_ylabel("Density")

    # --- Right plot: log-log scale (verify power-law) ---
    def get_pdf(degs):
        values, counts = np.unique(degs, return_counts=True)
        mask = values > 0
        return values[mask], counts[mask] / len(degs)

    v, p = get_pdf(degrees)
    ax2.scatter(v, p, alpha=0.7, color="skyblue", s=25)

    # If gamma provided, draw reference line
    if gamma is not None:
        x_ref = np.exp(np.linspace(np.log(min(v)), np.log(max(v)), 100))
        # Find a middle point to align reference line
        mid_idx = len(p) // 2
        y_ref = x_ref ** (-gamma) * (p[mid_idx] / (v[mid_idx] ** (-gamma)))
        ax2.plot(x_ref, y_ref, "r--", label=rf"Power-law Ref ($\gamma={gamma}$)")
        ax2.legend()

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_title(f"{title} (Log-Log Scale)")
    ax2.set_xlabel("Degree (log)")
    ax2.set_ylabel("Probability (log)")
    ax2.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Unified test parameters
    N = 5000
    k_avg = 6
    epsilon = 0.175
    p_triangle = 0.65

    # 1. Test Poissonian small-world network
    G_psw = generate_poissonian_small_world(N=N, k_avg=k_avg, epsilon=epsilon)
    _print_network_statistics(
        graph=G_psw,
        network_name="Poissonian Small World Network",
        target_k_avg=k_avg,
        epsilon=epsilon,
    )

    # 2. Test scale-free network (Holme-Kim)
    G_hk = generate_holme_kim(N=N, k_avg=k_avg, p_triangle=p_triangle)
    _print_network_statistics(
        graph=G_hk,
        network_name="Scale-free Network (Holme-Kim)",
        target_k_avg=k_avg,
        p_triangle=p_triangle,
    )

    # 3. Test Stochastic Holme-Kim network
    G_shk = generate_stochastic_holme_kim(N=N, k_avg=k_avg, p_triangle=p_triangle)
    _print_network_statistics(
        graph=G_shk,
        network_name="Stochastic Holme-Kim Network",
        target_k_avg=k_avg,
        p_triangle=p_triangle,
    )

    # 4. Test configuration model
    G_config = get_configuration_null_model(G_shk)
    _print_network_statistics(
        graph=G_config,
        network_name="Configuration Model",
        target_k_avg=k_avg,
    )

    # Test regular lattice network
    G_lattice = generate_regular_lattice(N=N, k=k_avg)
    _print_network_statistics(
        graph=G_lattice,
        network_name="Regular Ring Lattice",
        target_k_avg=k_avg,
    )

    # Test Erdos-Renyi network
    G_er = generate_erdos_renyi(N=N, k_avg=k_avg)
    _print_network_statistics(
        graph=G_er,
        network_name="Erdos-Renyi Network",
        target_k_avg=k_avg,
    )

    # Test Barabasi-Albert network
    G_ba = generate_barabasi_albert(N=N, k_avg=k_avg)
    _print_network_statistics(
        graph=G_ba,
        network_name="Barabasi-Albert Network",
        target_k_avg=k_avg,
    )

    # 5. Test Activity Driven Network (snapshot vs aggregated)
    N = 500
    m = 1
    T = 12
    p_active = 0.02
    gamma = 2.7

    adn_model = ActivityDrivenNetwork(N=N, m=m, gamma=gamma)

    G_aggregated = nx.Graph()
    G_aggregated.add_nodes_from(range(N))
    for _ in range(T):
        G_t = adn_model.generate_snapshot(p_active=p_active)
        G_aggregated.add_edges_from(G_t.edges())

    non_iso_nodes = [n for n, d in G_aggregated.degree() if d > 0]
    avg_k_non_iso = 2 * G_aggregated.number_of_edges() / len(non_iso_nodes)
    print(f"Aggregated - Non-isolated nodes: {len(non_iso_nodes)}")
    print(f"Aggregated - Avg degree (non-isolated): {avg_k_non_iso:.4f}")

    # visualize_degree_distribution(
    #     G_t, title="Activity Driven Network (Single Snapshot)", gamma=2.7
    # )
    # visualize_degree_distribution(
    #     G_aggregated, title="Activity Driven Network (Aggregated)", gamma=2.7
    # )

    _print_network_statistics(
        graph=G_t,
        network_name="Activity Driven Network (Single Snapshot)",
        target_k_avg=0,
    )
    # _print_network_statistics(
    #     graph=G_aggregated,
    #     network_name="Activity Driven Network (Aggregated)",
    #     target_k_avg=0,
    # )
