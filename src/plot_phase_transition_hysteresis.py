import random
from dataclasses import asdict, dataclass, field
from multiprocessing import Pool, cpu_count
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import model_core
import network_generator

# Path settings
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent
RESULTS_DIR = BASE_DIR / "results" / "Fig_polarization_hysteresis"
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
    N: int = 500  # Number of nodes

    # === Forward process parameters ===
    k_forward_values: np.ndarray = field(default_factory=lambda: np.arange(3, 16, 0.5))

    # === Reverse (hysteresis) process parameters ===
    k_start: float = 16.0  # Initial high connectivity
    k_stop: float = 3.0  # Final low connectivity
    k_step_down: float = 0.5  # Average degree reduction per edge deletion

    # === Specific network parameters ===
    # === Specific network parameters ===
    epsilon: float = 0.3  # Poisson small-world: rewiring probability (Julia: 0.3)
    p_triangle: float = 0.65  # Scale-free network: triangle formation probability

    # === Model dynamics parameters ===
    G_topics: int = 9  # Opinion dimension
    alpha: float = 0.5  # Friend/enemy weight
    beta: float = 4.0  # Social pressure sensitivity (Julia: 4.0)
    gamma: float = 0.001  # Instigator fraction (Julia: 0.001)

    n_trials: int = 20  # Number of trials

    # === Sampling parameters ===
    n_samples: int = 30  # Steady-state sampling count (Julia: 30)
    sample_interval: int = 1  # Sampling interval (sweeps)

    # === Runtime state ===
    network_type: str = "HK_configuration"
    burn_in_sweeps: int = 100  # Initial warmup (Julia: 100)
    relaxation_sweeps: int = 100  # Re-equilibration time after edge deletion (Julia: 100 for hysteresis)


def run_forward_simulation_single(config_dict, k_target, seed):
    """
    Forward simulation: initialize at fixed k and run to steady state (same as plot_phase_transition.py)
    """
    np.random.seed(seed)
    # Unpack parameters
    N = config_dict["N"]
    G_topics = config_dict["G_topics"]
    alpha = config_dict["alpha"]
    beta = config_dict["beta"]
    gamma = config_dict["gamma"]
    burn_in_sweeps = config_dict["burn_in_sweeps"]
    network_type = config_dict["network_type"]

    n_samples = config_dict["n_samples"]
    sample_interval = config_dict["sample_interval"]

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

    # 2. Select Instigators
    # Following Julia logic: round(gamma * N) count
    n_instigators = int(round(gamma * N))
    if n_instigators == 0 and gamma > 0:
        # Ensure at least one if gamma > 0 (though Julia actually allows 0)
        # N=500, gamma=0.001 -> 0.5 -> round -> 0 -> I=0?
        # Julia: I = Int(round(gamma*N)). If N=500, gamma=0.001, I=0.
        # But if it's 0, there are no instigators at all.
        # User didn't specifically say N must be 500.
        # If N=500, then it's truly 0.
        # We respect the math.
        pass

    fixed_nodes = None
    if n_instigators > 0:
        all_nodes = list(range(G.number_of_nodes()))
        fixed_nodes = np.random.choice(all_nodes, size=n_instigators, replace=False)

    # 3. Initialize model
    model = model_core.BinaryOpinionModel(
        N=G.number_of_nodes(),
        G_topics=G_topics,
        alpha=alpha,
        beta=beta,
        fixed_nodes=fixed_nodes,
    )

    # 4. Set instigator opinions to all +1 (matching Julia: ones(Int8, G, I))
    if fixed_nodes is not None:
        model.opinions[fixed_nodes] = 1

    # 5. Run model (Burn-in)
    nodes = np.arange(G.number_of_nodes())
    neighbors = [np.array(list(G.neighbors(i))) for i in nodes]
    total_steps = burn_in_sweeps * G.number_of_nodes() * G_topics

    for _ in range(total_steps):
        model.step(nodes, neighbors)

    # 6. Sampling average
    psi_samples = []
    # If n_samples > 0, continue running and sampling
    # Note Julia: simulate!(SN, 1) -> sample -> repeat 30 times
    # Julia's simulate!(SN, t) is MCS step. t=1 is 1 MCS (N * G steps).
    steps_per_sample = sample_interval * G.number_of_nodes() * G_topics

    for _ in range(n_samples):
        # Run interval
        for _ in range(steps_per_sample):
            model.step(nodes, neighbors)
        # Record
        psi, _ = model.calculate_polarization()
        psi_samples.append(psi)

    global_psi_mean = np.mean(psi_samples)
    return global_psi_mean, actual_k


def run_hysteresis_simulation_single(config_dict, seed):
    """
    Reverse (hysteresis) simulation:
    1. Initialize high connectivity network (k_start)
    2. Run to steady state
    3. Gradually delete edges to lower k, run relaxation_sweeps after each deletion, record psi
    """
    np.random.seed(seed)
    # Unpack parameters
    N = config_dict["N"]
    G_topics = config_dict["G_topics"]
    alpha = config_dict["alpha"]
    beta = config_dict["beta"]
    gamma = config_dict["gamma"]
    burn_in_sweeps = config_dict["burn_in_sweeps"]
    relaxation_sweeps = config_dict["relaxation_sweeps"]
    network_type = config_dict["network_type"]
    n_samples = config_dict["n_samples"]
    sample_interval = config_dict["sample_interval"]

    k_curr = config_dict["k_start"]
    k_stop = config_dict["k_stop"]
    k_step = config_dict["k_step_down"]

    # 1. Generate initial high connectivity network
    if network_type == "small_world":
        epsilon = config_dict["epsilon"]
        G = network_generator.generate_poissonian_small_world(N, k_curr, epsilon)
    elif network_type == "scale_free":
        p_triangle = config_dict["p_triangle"]
        G = network_generator.generate_stochastic_holme_kim(N, k_curr, p_triangle)
    elif network_type == "HK_configuration":
        p_triangle = config_dict["p_triangle"]
        G_hk = network_generator.generate_stochastic_holme_kim(N, k_curr, p_triangle)
        G = network_generator.get_configuration_null_model(G_hk)
    else:
        raise ValueError(f"Unknown network type: {network_type}")

    # Convert to networkx graph for edge manipulation (network_generator returns nx.Graph)
    # Ensure it's modifiable
    G = G.copy()

    # 2. Select Instigators
    n_instigators = int(round(gamma * N))
    fixed_nodes = None
    if n_instigators > 0:
        all_nodes = list(range(G.number_of_nodes()))
        fixed_nodes = np.random.choice(all_nodes, size=n_instigators, replace=False)

    # 3. Initialize model
    model = model_core.BinaryOpinionModel(
        N=G.number_of_nodes(),
        G_topics=G_topics,
        alpha=alpha,
        beta=beta,
        fixed_nodes=fixed_nodes,
    )

    # 4. Set instigator opinions to all +1
    if fixed_nodes is not None:
        model.opinions[fixed_nodes] = 1

    results_k = []
    results_psi = []

    # 5. Initial warmup
    nodes = np.arange(G.number_of_nodes())
    neighbors = [np.array(list(G.neighbors(i))) for i in nodes]
    total_steps_burn_in = burn_in_sweeps * N * G_topics

    for _ in range(total_steps_burn_in):
        model.step(nodes, neighbors)

    # Record initial state (does it need sampling?)
    # Julia code: high-to-low is first simulate!(SN, 100) -> this is a burn-in
    # then pmap(k -> hysteresis_experiment(SN, k, 100), k_values)
    # Note Julia's logic is slightly different:
    # It first runs 100 steps burn-in on a single `SN`.
    # Then passes it to `hysteresis_experiment`.
    # In `hysteresis_experiment`, it copy's SN to call `SN_thinned`.
    # Then since target_k is definitely less than current, it immediately thins.
    # **Key point**: Julia's hysteresis is actually parallel!
    # `pmap(k -> hysteresis_experiment(SN, k, 100), k_values)`
    # Each k is independently derived from original High-K steady state SN, thin to k, then relax.
    # Not serial k -> k-step -> k-2step.
    # PYTHON code line 169 comment said "independent edge deletion mode: each time from High-K steady state".
    # So logic is aligned.

    # Do we need to record High-K point result?
    # Julia's k_values includes High-K point?
    # Julia: k_values = collect(4.0:0.1:8.5). Maximum is 8.5.
    # Initial SN created with K = maximum(k_values)+1 = 9.5.
    # So k_values doesn't include initial point. All are after edge deletion.
    # Python's k_start = 16.0, k_values is arange(..., k_stop, ...).
    # Our target_ks contain points after reduction.

    # Save High-K steady state (Deep Copy)
    G_base = G.copy()
    opinions_base = model.opinions.copy()

    # 6. Gradual edge deletion loop (independent edge deletion mode)
    target_ks = np.arange(k_curr - k_step, k_stop - 0.1, -k_step)

    for target_k in target_ks:
        # Reset to High-K state
        G_temp = G_base.copy()
        model.opinions = opinions_base.copy()

        # Calculate edges to delete
        current_m = G_temp.number_of_edges()
        target_m = int(target_k * N / 2)
        edges_to_remove = current_m - target_m

        if edges_to_remove > 0:
            current_edges = list(G_temp.edges())
            random.shuffle(current_edges)
            edges_removed = current_edges[:edges_to_remove]
            G_temp.remove_edges_from(edges_removed)

        # Update neighbor list
        neighbors_temp = [np.array(list(G_temp.neighbors(i))) for i in nodes]

        # 7. Re-equilibration (Relaxation)
        total_steps_relax = relaxation_sweeps * N * G_topics
        for _ in range(total_steps_relax):
            model.step(nodes, neighbors_temp)

        # 8. Sampling and recording
        # Similar to forward, perform multiple samplings and take average
        psi_samples = []
        steps_per_sample = sample_interval * N * G_topics

        for _ in range(n_samples):
            # Run
            for _ in range(steps_per_sample):
                model.step(nodes, neighbors_temp)
            # Record
            psi, _ = model.calculate_polarization()
            psi_samples.append(psi)

        mean_psi = np.mean(psi_samples)
        actual_k = 2 * G_temp.number_of_edges() / N

        results_k.append(actual_k)
        results_psi.append(mean_psi)

    return results_k, results_psi


def run_forward_wrapper(args):
    return run_forward_simulation_single(*args)


def run_hysteresis_wrapper(args):
    return run_hysteresis_simulation_single(*args)


def run_experiment(config: Config):
    """
    Run complete forward and reverse experiments
    """
    config_dict = asdict(config)
    n_processes = cpu_count()
    print(f"Starting experiments with {n_processes} processes...")

    # --- 1. Forward Simulation ---
    print("\n[Forward Simulation] Initializing at various <k>...")
    forward_args = []
    base_seed = 10000
    idx = 0
    for k in config.k_forward_values:
        for _ in range(config.n_trials):
            forward_args.append((config_dict, k, base_seed + idx))
            idx += 1

    forward_results = []
    with Pool(processes=n_processes) as pool:
        results = pool.imap(run_forward_wrapper, forward_args)
        for psi, k_avg in tqdm(results, total=len(forward_args)):
            forward_results.append({"k_avg": k_avg, "psi": psi, "type": "forward"})

    df_forward = pd.DataFrame(forward_results)

    # --- 2. Hysteresis Simulation ---
    print("\n[Hysteresis Simulation] Reducing <k> from equilibrium...")
    hysteresis_args = []
    base_seed_h = 20000
    for i in range(config.n_trials):
        hysteresis_args.append((config_dict, base_seed_h + i))

    hysteresis_results_list = []
    with Pool(processes=n_processes) as pool:
        results = pool.imap(run_hysteresis_wrapper, hysteresis_args)
        for ks, psis in tqdm(results, total=len(hysteresis_args)):
            # Expand trajectory data
            for k_val, psi_val in zip(ks, psis):
                hysteresis_results_list.append(
                    {"k_avg": k_val, "psi": psi_val, "type": "hysteresis"}
                )

    df_hysteresis = pd.DataFrame(hysteresis_results_list)

    return df_forward, df_hysteresis


def save_data(df_forward, df_hysteresis, config):
    filename_base = f"{config.network_type}_{config.N}nodes"

    path_forward = RESULTS_DIR / f"{filename_base}_forward.csv"
    df_forward.to_csv(path_forward, index=False)

    path_hysteresis = RESULTS_DIR / f"{filename_base}_hysteresis.csv"
    df_hysteresis.to_csv(path_hysteresis, index=False)

    print(f"Data saved to:\n  {path_forward}\n  {path_hysteresis}")


def plot_results(df_forward, df_hysteresis, config):
    """
    Plot hysteresis graph
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # 1. Forward Data (Blue Circles)
    # Aggregate mean for clearer plot, or scatter points?
    # Original S4 looks like scatter or mean.
    # Let's do binned mean first or direct scatter
    # For this plot beauty, we calculate mean near each k_avg

    # Simplification: direct scatter, alpha transparency
    # But original very clean, should be binned mean

    def get_binned_mean(df, bins=40):
        if df.empty:
            return pd.DataFrame()
        df = df.copy()
        df["bin"] = pd.cut(df["k_avg"], bins)
        # Must specify observed=True for compatibility with new pandas
        # and explicitly select numeric columns for mean() to avoid TypeError
        return df.groupby("bin", observed=True)[["k_avg", "psi"]].mean().dropna()

    # Forward
    df_f_mean = get_binned_mean(df_forward)
    ax.scatter(
        df_f_mean["k_avg"],
        df_f_mean["psi"],
        c="#4c92c3",
        marker="o",
        s=60,
        label=r"initialized at $\langle k \rangle$",
        alpha=0.9,
        edgecolors="none",
    )

    # Hysteresis
    df_h_mean = get_binned_mean(df_hysteresis)
    ax.scatter(
        df_h_mean["k_avg"],
        df_h_mean["psi"],
        c="#d64541",
        marker="v",
        s=60,
        label=r"reduced to $\langle k \rangle$ after polarization",
        alpha=0.9,
        edgecolors="none",
    )

    # Add arrow indication
    # Find two points in hysteresis loop middle to draw arrow?
    # For simplicity, manually draw approximate curved arrow
    # Specific coordinates need to adjust based on data range, skip for now or draw approximately

    ax.set_xlabel(r"Average Degree $\langle k \rangle$")
    ax.set_ylabel(r"Polarization $\psi$")
    ax.set_xlim(3, 16.5)
    ax.set_ylim(0, 0.7)

    # Set spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.tick_params(direction="out", length=6, width=1.2)

    ax.legend(frameon=False, fontsize=12)
    ax.set_title(f"Hysteresis in Polarization ({config.network_type})")

    plt.tight_layout()
    output_path = RESULTS_DIR / f"hysteresis_plot_{config.network_type}.png"
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    # Configuration
    # User mainly wants Fig S4, usually Poisson Small World or HK Configuration
    # Default here to Poisson Small World to match "blue circles" description

    config = Config(
        N=500,
        network_type="scale_free",  # Match Julia (Poisson Small World)
        k_forward_values=np.arange(3, 14, 1),
        k_start=16.0,
        k_stop=3.0,
        k_step_down=0.5,
        n_trials=2,  # Trial count
        burn_in_sweeps=80,  # Warmup (Julia: 100)
        relaxation_sweeps=50,  # Recovery after edge deletion (Julia: 100)
        # Other params take default from Config class:
        # beta=4.0, epsilon=0.3, gamma=0.001, n_samples=30
    )

    # 1. Run experiment
    # To use existing data directly, comment out this line and modify loading logic
    df_f, df_h = run_experiment(config)

    # 2. Save data
    save_data(df_f, df_h, config)

    # 3. Plot
    plot_results(df_f, df_h, config)
