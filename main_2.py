import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, List

# ------------------------------------------------------------
# Basic grid / environment
# ------------------------------------------------------------

ACTIONS = [
    (0, 1),   # right
    (1, 0),   # down
    (0, -1),  # left
    (-1, 0),  # up
]
N_ACTIONS = len(ACTIONS)


@dataclass
class PEGEnv:
    grid: np.ndarray  # 0 free, 1 obstacle, 2 evasion region
    capture_radius: int = 0  # capture if on same cell (0) or within manhattan radius
    wind_prob: float = 0.1   # probability that wind perturbs the move

    def __post_init__(self):
        self.H, self.W = self.grid.shape
        self.n_cells = self.H * self.W
        # Precompute indices of evasion cells
        self.evasion_cells = np.where(self.grid.ravel() == 2)[0]
        self.obstacle_cells = np.where(self.grid.ravel() == 1)[0]

    # ---------- helpers for indexing ----------

    def pos_to_xy(self, pos: int) -> Tuple[int, int]:
        return divmod(pos, self.W)

    def xy_to_pos(self, i: int, j: int) -> int:
        return i * self.W + j

    def in_bounds(self, i: int, j: int) -> bool:
        return 0 <= i < self.H and 0 <= j < self.W

    # ---------- terminal conditions + reward ----------

    def is_crash(self, pos: int) -> bool:
        # crash if obstacle or out of domain (out-of-domain handled via invalid moves)
        return self.grid.ravel()[pos] == 1

    def is_evasion(self, evader_pos: int) -> bool:
        return self.grid.ravel()[evader_pos] == 2

    def is_capture(self, p_pos: int, e_pos: int) -> bool:
        ip, jp = self.pos_to_xy(p_pos)
        ie, je = self.pos_to_xy(e_pos)
        manhattan = abs(ip - ie) + abs(jp - je)
        return manhattan <= self.capture_radius

    def classify_terminal(self, p_pos: int, e_pos: int) -> Tuple[bool, float]:
        """
        Returns (done, reward_for_pursuer)
        Rewards follow the paper's terminal reward G_h (zero-sum). 1/-1/0.
        """
        # out-of-bounds shouldn't happen if we clamp, but just in case:
        if not (0 <= p_pos < self.n_cells and 0 <= e_pos < self.n_cells):
            return True, 0.0

        p_crash = self.is_crash(p_pos)
        e_crash = self.is_crash(e_pos)
        evasion = self.is_evasion(e_pos)
        capture = self.is_capture(p_pos, e_pos)

        # If both crash
        if p_crash and e_crash:
            return True, 0.0

        # Single crash
        if p_crash and not e_crash:
            return True, -1.0  # pursuer crashes
        if e_crash and not p_crash:
            return True, 1.0   # evader crashes

        # Capture vs evasion
        if capture:
            return True, 1.0
        if evasion:
            return True, -1.0

        return False, 0.0

    # ---------- wind / motion model ----------

    def step_single_agent(
        self,
        pos: int,
        action_idx: int,
        wind_move: Tuple[int, int]
    ) -> int:
        """
        Apply action + wind to a single agent; clamp at boundaries.
        """
        i, j = self.pos_to_xy(pos)
        di, dj = ACTIONS[action_idx]
        wi, wj = wind_move
        ni = i + di + wi
        nj = j + dj + wj
        # If out of bounds, treat as staying on border cell and then crashing
        if not self.in_bounds(ni, nj):
            # clamp to border (still crash since boundary is treated as obstacle)
            ni = min(max(ni, 0), self.H - 1)
            nj = min(max(nj, 0), self.W - 1)
        return self.xy_to_pos(ni, nj)

    def wind_outcomes(self) -> List[Tuple[Tuple[int, int], float]]:
        """
        Discrete wind: with prob (1-wp) no wind, with prob wp, random cardinal noise.
        """
        wp = self.wind_prob
        if wp <= 0:
            return [((0, 0), 1.0)]

        noise_moves = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
        # probability mass: (1 - wp) on (0,0), remaining uniformly on others
        p0 = 1.0 - wp
        p_rest = wp / (len(noise_moves) - 1)
        probs = [p0] + [p_rest] * (len(noise_moves) - 1)
        return list(zip(noise_moves, probs))

    def joint_transition_distribution(
        self,
        p_pos: int,
        e_pos: int,
        a_p: int,
        a_e: int,
    ) -> List[Tuple[int, float]]:
        """
        Returns list of (next_state_idx, prob) for joint state.
        We treat wind on pursuer and evader as independent.
        """
        outcomes = []
        wind_out = self.wind_outcomes()

        # For each combo of wind for pursuer and evader
        for (wp, pp), (we, pe) in [(w1, w2) for w1 in wind_out for w2 in wind_out]:
            next_p = self.step_single_agent(p_pos, a_p, wp)
            next_e = self.step_single_agent(e_pos, a_e, we)
            next_s = next_p * self.n_cells + next_e
            prob = pp * pe
            outcomes.append((next_s, prob))

        # Merge duplicates
        dist: Dict[int, float] = {}
        for s, p in outcomes:
            dist[s] = dist.get(s, 0.0) + p

        return list(dist.items())


# ------------------------------------------------------------
# Level-k best-response / value iteration
# ------------------------------------------------------------

def build_random_policy(n_states: int, n_actions: int) -> np.ndarray:
    """
    Level-0 policy: uniform random over actions, independent of state.
    Shape: (n_states, n_actions).
    """
    pi = np.ones((n_states, n_actions), dtype=np.float64) / n_actions
    return pi


def value_iteration_best_response(
    env: PEGEnv,
    opponent_policy: np.ndarray,
    agent_is_pursuer: bool,
    max_iters: int = 2000,
    tol: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute best-response policy for one agent given opponent's policy.

    opponent_policy: shape (n_states, N_ACTIONS) – mixed policy of the other agent.
    Returns:
        V: value function for this agent (length n_states),
        pi: deterministic policy table (one-hot over actions) shape (n_states, N_ACTIONS).
    """
    n_cells = env.n_cells
    n_states = n_cells * n_cells

    # Precompute which states are terminal and their rewards
    terminal_mask = np.zeros(n_states, dtype=bool)
    terminal_reward = np.zeros(n_states, dtype=np.float64)

    for p_pos in range(n_cells):
        for e_pos in range(n_cells):
            s_idx = p_pos * n_cells + e_pos
            done, r = env.classify_terminal(p_pos, e_pos)
            if done:
                terminal_mask[s_idx] = True
                # pursuer reward; if evader is the agent, value is -r
                terminal_reward[s_idx] = r if agent_is_pursuer else -r

    V = np.zeros(n_states, dtype=np.float64)
    V[terminal_mask] = terminal_reward[terminal_mask]

    # Value iteration (no discount; absorbing terminals)
    for it in range(max_iters):
        V_old = V.copy()
        delta = 0.0

        for p_pos in range(n_cells):
            for e_pos in range(n_cells):
                s_idx = p_pos * n_cells + e_pos
                if terminal_mask[s_idx]:
                    continue  # fixed

                best_val = None  # max for pursuer, min for evader

                # Loop over this agent's actions
                for a_self in range(N_ACTIONS):
                    expected_value = 0.0

                    # Expectation over opponent actions
                    for a_opp in range(N_ACTIONS):
                        # opponent_policy is indexed by joint state
                        p_a_opp = opponent_policy[s_idx, a_opp]
                        if p_a_opp == 0:
                            continue

                        # Map actions depending on who is pursuer
                        if agent_is_pursuer:
                            a_p = a_self
                            a_e = a_opp
                        else:
                            a_p = a_opp
                            a_e = a_self

                        # Transition distribution under these actions
                        dist = env.joint_transition_distribution(p_pos, e_pos, a_p, a_e)

                        # Expected V over next states
                        for s_next, p_trans in dist:
                            if terminal_mask[s_next]:
                                expected_value += p_a_opp * p_trans * terminal_reward[s_next]
                            else:
                                expected_value += p_a_opp * p_trans * V_old[s_next]

                    if best_val is None:
                        best_val = expected_value
                    else:
                        if agent_is_pursuer:
                            best_val = max(best_val, expected_value)
                        else:
                            best_val = min(best_val, expected_value)

                if best_val is None:
                    best_val = 0.0  # no actions? shouldn't happen

                V[s_idx] = best_val
                delta = max(delta, abs(V[s_idx] - V_old[s_idx]))

        if delta < tol:
            print(f"Value iteration converged in {it+1} iterations (delta={delta:.2e})")
            break

    # Extract greedy deterministic policy (pure policy) at each state
    pi = np.zeros((n_states, N_ACTIONS), dtype=np.float64)

    for p_pos in range(n_cells):
        for e_pos in range(n_cells):
            s_idx = p_pos * n_cells + e_pos
            if terminal_mask[s_idx]:
                # No action needed; but pick arbitrary
                pi[s_idx, :] = 1.0 / N_ACTIONS
                continue

            # Evaluate each action again
            values = np.zeros(N_ACTIONS, dtype=np.float64)
            for a_self in range(N_ACTIONS):
                ev = 0.0
                for a_opp in range(N_ACTIONS):
                    p_a_opp = opponent_policy[s_idx, a_opp]
                    if p_a_opp == 0:
                        continue
                    if agent_is_pursuer:
                        a_p, a_e = a_self, a_opp
                    else:
                        a_p, a_e = a_opp, a_self
                    dist = env.joint_transition_distribution(p_pos, e_pos, a_p, a_e)
                    for s_next, p_trans in dist:
                        if terminal_mask[s_next]:
                            ev += p_a_opp * p_trans * terminal_reward[s_next]
                        else:
                            ev += p_a_opp * p_trans * V[s_next]
                values[a_self] = ev

            if agent_is_pursuer:
                best_a = np.argmax(values)
            else:
                best_a = np.argmin(values)
            pi[s_idx, :] = 0.0
            pi[s_idx, best_a] = 1.0

    return V, pi


def build_level_k_policies(env: PEGEnv, k_p_max: int, k_e_max: int):
    """
    Construct level-k policies for pursuer and evader, up to k_p_max / k_e_max.
    Returns:
        mu_p[k]: policy for pursuer at level k (shape (S, A))
        mu_e[k]: policy for evader at level k (shape (S, A))
    """
    n_states = env.n_cells * env.n_cells

    mu_p: Dict[int, np.ndarray] = {}
    mu_e: Dict[int, np.ndarray] = {}

    # Level-0: random policies
    mu_p[0] = build_random_policy(n_states, N_ACTIONS)
    mu_e[0] = build_random_policy(n_states, N_ACTIONS)

    # Build successive levels (simple alternating scheme)
    for k in range(1, max(k_p_max, k_e_max) + 1):
        print(f"\n--- Building level {k} policies ---")

        # pursuer level k responds to evader level k-1
        if k <= k_p_max:
            print("Computing pursuer level", k)
            _, mu_p[k] = value_iteration_best_response(
                env, opponent_policy=mu_e[k - 1], agent_is_pursuer=True
            )

        # evader level k responds to pursuer level k-1
        if k <= k_e_max:
            print("Computing evader level", k)
            _, mu_e[k] = value_iteration_best_response(
                env, opponent_policy=mu_p[k - 1], agent_is_pursuer=False
            )

    return mu_p, mu_e


# ------------------------------------------------------------
# Simulation
# ------------------------------------------------------------

def sample_action(policy: np.ndarray, s_idx: int) -> int:
    probs = policy[s_idx]
    # deterministic policy → one-hot; still use np.random.choice for generality
    return int(np.random.choice(np.arange(N_ACTIONS), p=probs))


def simulate_episode(
    env: PEGEnv,
    mu_p: np.ndarray,
    mu_e: np.ndarray,
    p_start: int,
    e_start: int,
    max_steps: int = 100,
) -> Tuple[List[int], List[int], float]:
    """
    Simulate one game: returns pursuer positions, evader positions, terminal reward (for pursuer).
    """
    p_pos = p_start
    e_pos = e_start
    traj_p = [p_pos]
    traj_e = [e_pos]

    n_cells = env.n_cells

    for t in range(max_steps):
        done, r = env.classify_terminal(p_pos, e_pos)
        if done:
            return traj_p, traj_e, r

        s_idx = p_pos * n_cells + e_pos

        a_p = sample_action(mu_p, s_idx)
        a_e = sample_action(mu_e, s_idx)

        # Sample wind
        wind_out = env.wind_outcomes()
        wp, pp = wind_out[np.random.randint(len(wind_out))]
        we, pe = wind_out[np.random.randint(len(wind_out))]

        next_p = env.step_single_agent(p_pos, a_p, wp)
        next_e = env.step_single_agent(e_pos, a_e, we)

        p_pos, e_pos = next_p, next_e
        traj_p.append(p_pos)
        traj_e.append(e_pos)

    # If max_steps reached without terminal, treat as no result
    return traj_p, traj_e, 0.0


# ------------------------------------------------------------
# Minimal demo with a toy grid
# ------------------------------------------------------------

if __name__ == "__main__":
    # Example: 8x8 grid:
    #  - border is obstacles
    #  - two evasion cells in top-right / bottom-right
    H, W = 8, 8
    grid = np.zeros((H, W), dtype=int)

    # set border as obstacles
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1

    # evasion cells
    grid[1, -2] = 2
    grid[-2, -2] = 2

    env = PEGEnv(grid=grid, capture_radius=0, wind_prob=0.1)

    # Build level-k policies (small k to keep things fast)
    k_p_max = 2
    k_e_max = 2
    mu_p_dict, mu_e_dict = build_level_k_policies(env, k_p_max, k_e_max)

    # Choose specific levels for simulation
    k_p = 2
    k_e = 1
    mu_p_k = mu_p_dict[k_p]
    mu_e_k = mu_e_dict[k_e]

    # Initial positions (manually pick some free cells)
    p_start = env.xy_to_pos(2, 2)  # pursuer
    e_start = env.xy_to_pos(5, 5)  # evader

    traj_p, traj_e, r = simulate_episode(env, mu_p_k, mu_e_k, p_start, e_start)

    print(f"Terminal reward for pursuer: {r}")
    print("Length of trajectory:", len(traj_p))

    # Optional: visualize trajectory
    try:
        import matplotlib.pyplot as plt
        img = np.copy(grid).astype(float)
        # Normalize for plotting
        img[img == 1] = 0.3   # obstacles
        img[img == 0] = 0.9   # free
        img[img == 2] = 0.1   # evasion

        plt.imshow(img, origin="upper", cmap="gray")

        for t, (pp, ee) in enumerate(zip(traj_p, traj_e)):
            ip, jp = env.pos_to_xy(pp)
            ie, je = env.pos_to_xy(ee)
            plt.scatter(jp, ip, c="orange", marker="s")
            plt.scatter(je, ie, c="blue", marker="o")

        plt.title(f"Trajectory (reward={r}, steps={len(traj_p)-1})")
        plt.show()
    except ImportError:
        pass
