import numpy as np

import network_generator


class BinaryOpinionModel:
    """
    Binary opinion model
    - Nodes: social individuals
    - Opinion vector: binary opinion (+1 / -1) for each node on G topics
    - Edge weight: friend (+1) / enemy (-1), determined by the sign of opinion vector dot product
    """

    def __init__(
        self,
        N: int,
        G_topics: int,
        alpha: float,
        beta: float,
        fixed_nodes=None,
    ):
        """
        Initialize binary opinion model

        Parameters
        ----------
        N : int
            Number of individuals
        G_topics : int
            Number of topics/opinions
        alpha : float
            Friend/enemy weight parameter
        beta : float
            Social sensitivity
        fixed_nodes : Iterable[int] | None
            If provided, these nodes will be "frozen" and their opinions will not be updated by dynamics
        """
        self.N = int(N)
        self.G_topics = int(G_topics)
        self.alpha = float(alpha)
        self.beta = float(beta)

        # Opinion matrix: shape (N, G), elements are -1 or +1
        self.opinions = np.random.choice([-1, 1], size=(self.N, self.G_topics))

        # Frozen node flags: do not participate in opinion evolution
        self.is_fixed = np.zeros(self.N, dtype=bool)
        if fixed_nodes is not None:
            self.is_fixed[np.array(list(fixed_nodes), dtype=int)] = True

        # Cumulative flip acceptance counts
        self.acceptance_counts = np.zeros(self.N)

    def step(self, nodes: np.ndarray, neighbors: list[np.ndarray]):
        """
        Execute one Metropolis Monte Carlo update:
        - Randomly select node i and an opinion dimension k
        - Attempt to flip s_i^k → -s_i^k
        - Accept or reject according to Metropolis criterion
        - If accepted, update the node's opinion

        Parameters
        ----------
        nodes : np.ndarray
            List of active node indices (nodes eligible for opinion updates in current time step)
        neighbors : list[np.ndarray]
            Adjacency list structure, neighbors[i] stores array of neighbor indices of node i
        """
        # 1. Randomly select an active node and dimension
        i = nodes[np.random.randint(len(nodes))]
        k = np.random.randint(0, self.G_topics)

        # Frozen nodes: maintain initial random state, skip update
        if self.is_fixed[i]:
            return

        # Get neighbors in current network
        nbs = neighbors[i]

        # Isolated nodes: no neighbors in current network, skip update
        if nbs.size == 0:
            return

        # Get neighbor opinion matrix (deg(i), G)
        neighbor_ops = self.opinions[nbs]

        # 2. Propose flip
        old_op = self.opinions[i].copy()
        new_op = old_op.copy()
        new_op[k] *= -1

        # 3. Calculate energy difference ΔE = E_new - E_old
        E_old = self.calculate_individual_energy(old_op, neighbor_ops)
        E_new = self.calculate_individual_energy(new_op, neighbor_ops)
        delta_E = E_new - E_old

        # 4. Metropolis acceptance criterion
        if delta_E <= 0:
            accept = True
        else:
            p = np.exp(-self.beta * delta_E)
            accept = bool(np.random.random() < p)

        if not accept:
            return

        # 5. Accept update: update opinion vector
        self.opinions[i] = new_op
        self.acceptance_counts[i] += 1

    def calculate_individual_energy(
        self,
        opinion_vec: np.ndarray,
        neighbor_opinions: np.ndarray,
    ):
        """
        Calculate local energy of node (with opinion opinion_vec)
        H^i = (1/G) * [ -alpha * sum_{(i,j) are friends} (s_i · s_j)
                      + (1 - alpha) * sum_{(i,j) are enemies} (s_i · s_j) ]

        Parameters
        ----------
        opinion_vec : np.ndarray
            Opinion vector of this node (G,)
        neighbor_opinions : np.ndarray
            Opinion matrix of neighbors (deg(i), G)

        Returns
        -------
        float
            Local energy H^i of the node
        """
        # Opinion similarity alignment_j = (s_i · s_j) / G
        alignments = (neighbor_opinions @ opinion_vec) / self.G_topics

        # Calculate energy contributions based on friends (alignments > 0), enemies (alignments <= 0)
        energy_contributions = np.where(
            alignments > 0,
            -self.alpha * alignments,
            (1.0 - self.alpha) * alignments,
        )

        return float(energy_contributions.sum())

    def calculate_polarization1(self):
        """
        Calculate global polarization ψ and individual polarization for each individual.

        Returns
        -------
        global_psi : float
            System global polarization
        individual_psi : np.ndarray
            Shape (N,), individual polarization degree for each node
        """
        # 1. Opinion similarity matrix (N, N)
        alignment_matrix = self.opinions @ self.opinions.T / self.G_topics

        # 2. Global polarization: calculate variance on upper triangle (k=1)
        indices = np.triu_indices(self.N, k=1)
        alignments_upper = alignment_matrix[indices]
        global_psi = float(np.var(alignments_upper))

        # 3. Individual polarization: get row i and remove self-similarity, calculate variance
        mask = ~np.eye(self.N, dtype=bool)
        individual_psi = np.var(
            alignment_matrix[mask].reshape(self.N, self.N - 1), axis=1
        )

        return global_psi, individual_psi

    def calculate_polarization(self):
        """
        Calculate global polarization ψ and individual polarization (optimized O(N * G^2) algorithm).

        Returns
        -------
        global_psi : float
            System global polarization
        individual_psi : np.ndarray
            Shape (N,), individual polarization degree for each node
        """
        # To calculate variance Var(X) = E[X^2] - (E[X])^2, we need to calculate first and second moments of a_ij
        # a_ij = (1/G) * \sum_k s_i^k * s_j^k

        # 1. Precompute statistics
        # S_k = \sum_i s_i^k (shape G,)
        S_k = np.sum(self.opinions, axis=0)
        # M_km = \sum_i s_i^k * s_i^m (shape G x G)
        M_km = self.opinions.T @ self.opinions

        # 2. Global first moment E[a_ij] (for all i < j pairs)
        sum_a = 0.5 * (np.sum(S_k**2) - self.N) / self.G_topics
        num_pairs = self.N * (self.N - 1) / 2
        E_a = sum_a / num_pairs

        # 3. Global second moment E[a_ij^2]
        # \sum_{k,m} (M_km^2 - \sum_i (s_i^k * s_i^m)^2) / (2 * G^2)
        # Since (s_i^k * s_i^m)^2 always equals 1, its sum is N
        sum_a_sq = (
            0.5 * (np.sum(M_km**2) - self.N * self.G_topics**2) / (self.G_topics**2)
        )
        E_a_sq = sum_a_sq / num_pairs

        global_psi = float(E_a_sq - E_a**2)

        # 4. Individual polarization individual_psi[i] (for node i, calculate variance of alignment with all j!=i)
        # E_j[a_ij] = (\sum_k s_i^k * (S_k - s_i^k)) / (G * (N-1))
        S_ik_Sk = np.sum(self.opinions * S_k, axis=1)
        E_j_a = (S_ik_Sk - self.G_topics) / (self.G_topics * (self.N - 1))

        # E_j[a_ij^2] = (\sum_{k,m} (s_i^k * s_i^m) * (M_km - s_i^k * s_i^m)) / (G^2 * (N-1))
        # Fast matrix multiplication implementation: \sum_{k,m} (s_i^k * s_i^m) * M_km = s_i^T @ M_km @ s_i
        # Using (opinions @ M_km) * opinions then sum across rows
        matrix_part = np.sum((self.opinions @ M_km) * self.opinions, axis=1)
        E_j_a_sq = (matrix_part - self.G_topics**2) / (self.G_topics**2 * (self.N - 1))

        individual_psi = E_j_a_sq - E_j_a**2

        return global_psi, individual_psi

    def calculate_hamilton(self, neighbors: list[np.ndarray]):
        """
        Calculate Hamiltonian for each node and system average

        H^i is local energy of node i:
        H^i = (1/G) * [ -alpha * sum_{(i,j) are friends} (s_i · s_j)
                      + (1 - alpha) * sum_{(i,j) are enemies} (s_i · s_j) ]

        System average:
        H = (1/N) * sum_i H^i

        Parameters
        ----------
        neighbors : list[np.ndarray]
            Neighbor structure must be provided to calculate energy.

        Returns
        -------
        - global_hamilton: average Hamiltonian
        - energies: shape (N,), local Hamiltonian for each node
        """
        # Initialize energy for each node
        energies = np.zeros(self.N)

        # Iterate over all nodes, calculate local energy for each
        for i in range(self.N):
            nbs = neighbors[i]
            if nbs.size == 0:
                continue

            energies[i] = self.calculate_individual_energy(
                self.opinions[i], self.opinions[nbs]
            )

        # Return average energy and individual energies
        return energies.mean(), energies


if __name__ == "__main__":
    # Create network
    N = 500
    k_avg = 6
    epsilon = 0.1
    G_topics = 9
    alpha = 0.5
    beta = 2.7
    total_steps = 100  # Total evolution steps

    # Generate base network
    graph = network_generator.generate_poissonian_small_world(
        N=N, k_avg=k_avg, epsilon=epsilon
    )
    N_nodes = graph.number_of_nodes()

    model = BinaryOpinionModel(
        N=N_nodes,
        G_topics=G_topics,
        alpha=alpha,
        beta=beta,
    )

    # Generate neighbor structure for testing
    nodes = np.arange(N_nodes)
    neighbors = [np.array(list(graph.neighbors(i))) for i in nodes]

    # Run opinion evolution model
    for _ in range(total_steps):
        model.step(nodes, neighbors)

    global_polarization, individual_polarization = model.calculate_polarization()
    global_energy, individual_energies = model.calculate_hamilton(neighbors)

    print(f"Global Polarization (psi): {global_polarization:.4f}")
    print(f"Global Hamilton (Energy): {global_energy:.4f}")
