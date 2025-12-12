from enum import Enum
import math
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib import animation   # <<< add this
from functools import lru_cache   # <<< MOD: added for transition caching

# Parameters
@dataclass
class GameParams:
    h: float  # grid size for discretization
    evader_init_pos: tuple  # (x, y)
    pursuer_init_pos: tuple  # (x, y)
    evader_speed: float
    pursuer_speed: float
    action_space: list  # list of possible actions (heaading angles)
    capture_radius: float
    sigma_w: float  # covariance of wind field
    wind_spatial_correlation: float  # spatial correlation length scale
    
    # 2D grid representing obstacles and evader regions
    game_grid: np.ndarray  # 0: free, 1: obstacle, 2: evader region E


class PursuitEvasionGame:
    def __init__(self, params: GameParams):
        self.h = params.h
        self.capture_radius = params.capture_radius
        self.action_space = params.action_space
        self.sigma_w = params.sigma_w # wind noise std dev
        self.grid: np.ndarray = params.game_grid
        self.v = [params.pursuer_speed, params.evader_speed]

        # use grid size to determine state space
        H, W = self.grid.shape
        self.position_space = [(x, y) for x in range(W) for y in range(H)]
        # joint state space: (x1,y1,x2,y2)
        self.state_space = [(x1, y1, x2, y2) 
                            for (x1, y1) in self.position_space
                            for (x2, y2) in self.position_space]
        self.state = np.array(
            [params.pursuer_init_pos[0], params.pursuer_init_pos[1],
             params.evader_init_pos[0],  params.evader_init_pos[1]],
            dtype=int
        )

        # Boost performance by precomputing reward and terminal maps
        self.R = {
            s: self.reward_function(s, self.grid, self.capture_radius)
            for s in self.state_space
        }
        self.is_terminal_map = {
            s: self.is_terminal(s, self.grid, self.capture_radius)
            for s in self.state_space
        }
        self.non_terminal_states = [
            s for s in self.state_space if not self.is_terminal_map[s]
        ]

        print(f"Total states: {len(self.state_space)}, Non-terminal states: {len(self.non_terminal_states)}")

        # TODO: generate wind_map
        self.wind_map = {s: (0.0, 0.0) for s in self.position_space} # for each position (x,y), return (wx, wy)

        # <<< MOD: build index mappings and NumPy versions of R and terminal mask
        self.num_states = len(self.state_space)
        self.state_to_idx = {s: i for i, s in enumerate(self.state_space)}
        self.idx_to_state = list(self.state_space)

        self.R_array = np.zeros(self.num_states, dtype=float)
        self.is_terminal_array = np.zeros(self.num_states, dtype=bool)
        for i, s in enumerate(self.idx_to_state):
            self.R_array[i] = self.R[s]
            self.is_terminal_array[i] = self.is_terminal_map[s]

        self.non_terminal_indices = np.where(~self.is_terminal_array)[0]
    
    def random_init_state(self):
        # randomly select a non-terminal state for initialization
        idx = np.random.choice(len(self.non_terminal_states))
        self.state = np.array(self.non_terminal_states[idx], dtype=int)
    
    def step(self, pursuer_action, evader_action):
        pass
  
    def plot_state(self):
        # matplotlib code to plot the current state
        # grid val: 0 white, 1 black, 2 red
        # pursuer: blue dot, evader: green dot
        H, W = self.grid.shape
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[self.grid == 0] = [255, 255, 255]  # free
        img[self.grid == 1] = [0, 0, 0]        # obstacle
        img[self.grid == 2] = [255, 0, 0]      # evader region E
        plt.imshow(img)
        plt.scatter(self.state[0], self.state[1], c='blue', label='Pursuer')
        plt.scatter(self.state[2], self.state[3], c='green', label='Evader')
        
        plt.gca().invert_yaxis()  # <<< flip y-axis to match your internal coordinates
        plt.gca().set_aspect("equal")

        plt.legend()
        plt.show()
        
    def plot_trajectory(self, trajectory, title="Pursuit–Evader Trajectory"):
        """
        trajectory: list of states [(x1,y1,x2,y2), ...]
                    where state[t] corresponds to time t.
        """
        H, W = self.grid.shape

        # -------------------------
        # Background grid
        # -------------------------
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[self.grid == 0] = [255, 255, 255]  # free = white
        img[self.grid == 1] = [0,   0,   0]    # obstacle = black
        img[self.grid == 2] = [255, 0,   0]    # evader region E = red

        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title(title)

        # -------------------------
        # Extract pursuer / evader paths
        # -------------------------
        px = [s[0] for s in trajectory]  # pursuer x
        py = [s[1] for s in trajectory]  # pursuer y
        ex = [s[2] for s in trajectory]  # evader x
        ey = [s[3] for s in trajectory]  # evader y

        # -------------------------
        # Plot trajectories
        # -------------------------
        plt.plot(px, py, '-o', color='blue', markersize=3, label="Pursuer path")
        plt.plot(ex, ey, '-o', color='green', markersize=3, label="Evader path")

        # -------------------------
        # arrows to show direction
        # -------------------------
        def plot_arrows(xs, ys, color):
            for i in range(len(xs) - 1):
                dx = xs[i+1] - xs[i]
                dy = ys[i+1] - ys[i]
                plt.arrow(xs[i], ys[i], dx*0.8, dy*0.8,
                          head_width=0.3, head_length=0.3,
                          color=color, length_includes_head=True)

        plot_arrows(px, py, 'blue')
        plot_arrows(ex, ey, 'green')

        # -------------------------
        # Mark start + end
        # -------------------------
        plt.scatter(px[0], py[0], s=80, c='cyan', edgecolor='black', label="Pursuer start")
        plt.scatter(ex[0], ey[0], s=80, c='lime', edgecolor='black', label="Evader start")

        plt.scatter(px[-1], py[-1], s=80, c='navy', edgecolor='yellow', label="Pursuer end")
        plt.scatter(ex[-1], ey[-1], s=80, c='darkgreen', edgecolor='yellow', label="Evader end")

        plt.legend(loc='lower right')
        plt.gca().invert_yaxis()       # because image coordinates start top-left
        plt.grid(False)
        plt.show()
        
    def b_vec(self, state, joint_action, wind_map, v):
        # Drift vector b for each state_entry
        # v: [v1, v2] # speed of evader and pursuer
        x1, y1, x2, y2 = state
        p1 = (x1, y1)
        p2 = (x2, y2)

        w_p1 = wind_map[p1]  # (wx1, wy1)
        w_p2 = wind_map[p2]  # (wx2, wy2)

        w = [w_p1, w_p2]
        # w: [wx1(p1), wy1(p1), wx2(p2), wy2(p2)] # mean wind speed at positions of p1 and p2
        b_vec = np.zeros(4)
        for i in range(2):
            theta = joint_action[i]
            b_vec[2*i + 0] = v[i] * np.cos(theta) + w[i][0]
            b_vec[2*i + 1] = v[i] * np.sin(theta) + w[i][1]

        return b_vec

    def Q_h(self, state, h, sigma_w):
        all_joint_actions = [(a1, a2) for a1 in self.action_space for a2 in self.action_space]
        # Qh(s`tate) = h * max{sum_i=1^4 |b_i(state,theta|} (optimal theta)
        x1, y1, x2, y2 = state
        max_val = -np.inf
        for joint_action in all_joint_actions:
            b_vec = self.b_vec(state, joint_action, self.wind_map, self.v)
            norm_b = sum([abs(bi) for bi in b_vec])
            if norm_b > max_val:
                max_val = norm_b
        return h * max_val + 4*sigma_w**2

    def state_transition(self, state, joint_action, sigma_w):
        state = np.array(state, dtype=int)
        '''
          P_h(s_h'| s_h, theta)
          @return: dict mapping s_h' to probability
        '''
        sigma2 = sigma_w ** 2
        # b_vect represent the drift velocity (agent vel + wind speed)
        b_vec = self.b_vec(state, joint_action, self.wind_map, self.v)  # length 4
        Q = self.Q_h(state, self.h, sigma_w)

        probs = {}
        total = 0.0

        for j in range(4):
            e_j = np.zeros(4)
            e_j[j] = 1

            b_j = b_vec[j]
            b_plus = max(b_j, 0.0)
            b_minus = max(-b_j, 0.0)

            # s_h + h e_j -> grid index (grid size h) s_h + e_j
            s_plus = tuple(state + e_j)
            p_plus = (sigma2 / 2.0 + self.h * b_plus) / Q
            probs[s_plus] = probs.get(s_plus, 0.0) + p_plus
            total += p_plus

            # s_h - h e_j
            s_minus = tuple(state - e_j)
            p_minus = (sigma2 / 2.0 + self.h * b_minus) / Q
            probs[s_minus] = probs.get(s_minus, 0.0) + p_minus
            total += p_minus

        # staying in the same state
        s_stay = tuple(state)
        probs[s_stay] = max(0.0, 1.0 - total)  # numerical safety

        return probs

    def reward_function(self, state, game_grid, capture_radius):
        # Reward G_h(s_h):
        # 1: (captured and no crash) or { evader collide with obstacle/boundary and pursuer not}
        # -1: (evader reaches region E and no crash) or (pursuer collide with obstacle/boundary and evader not)
        # 0: otherwise
        pursuer_pos = (state[0], state[1])
        evader_pos = (state[2], state[3])
        
        # Pursuer reward
        if self.is_crash(evader_pos, game_grid) and not self.is_crash(pursuer_pos, game_grid):
            return 1
        elif self.is_crash(evader_pos, game_grid) and self.is_crash(pursuer_pos, game_grid):
            return 0
        elif not self.is_crash(evader_pos, game_grid) and self.is_crash(pursuer_pos, game_grid):
            return -1
        else: # both not crash
            # captured or evaded
            if self.is_captured(state):
                return 1
            elif self.is_in_region_E(evader_pos, game_grid):
                return -1
            else:
                return 0
    

    # ================== Terminal States ========================= #
    def is_crash(self, pos_idx, game_grid: np.ndarray):
        """
        Simple crash test:
          - outside grid bounds
          - or cell == 1 (obstacle)
        """
        x, y = pos_idx[0], pos_idx[1]
        H, W = game_grid.shape
        if x < 0 or x >= W or y < 0 or y >= H:
            return True
        return game_grid[y, x] == 1  # obstacle

    def is_in_region_E(self, pos_idx, game_grid: np.ndarray):
        x, y = pos_idx[0], pos_idx[1]
        H, W = game_grid.shape
        if x < 0 or x >= W or y < 0 or y >= H:
            return False
        return game_grid[y, x] == 2  # evader region E
    
    def is_captured(self, state):
        pursuer_pos = (state[0], state[1])
        evader_pos = (state[2], state[3])
        distance = np.linalg.norm(np.array(pursuer_pos) - np.array(evader_pos))
        return distance <= self.capture_radius

    def is_terminal(self, state, game_grid: np.ndarray, capture_radius):
        pursuer_pos = (state[0], state[1])
        evader_pos = (state[2], state[3])
        
        # Check for crash
        if self.is_crash(evader_pos, game_grid) or self.is_crash(pursuer_pos, game_grid):
            return True
        
        # Check for capture
        if self.is_captured(state):
            return True
        
        # Check for evader reaching region E
        if self.is_in_region_E(evader_pos, game_grid):
            return True
        
        return False

    def get_non_terminal_states(self, state_space, game_grid, capture_radius):
        S_non_term = []
        for s in state_space:
            if not self.is_terminal(s, game_grid, capture_radius):
                S_non_term.append(s)
        return S_non_term

    # level-k thinking
    def level_k_policies(self, max_level, grid, reward_function):
        # level from 0 to k (for both pursuer and evader)
        evader_policies = {}
        pursuer_policies = {}
        # 0-level: uniform random over action space
        prob_L0 = 1.0 / len(self.action_space)
        uniform_mu = {
            s: {a_i: prob_L0 for a_i in range(len(self.action_space))}
            for s in self.state_space
        }
        evader_policies[0] = uniform_mu
        pursuer_policies[0] = uniform_mu
        # k-level: best response to (k-1)-level opponent
        for k in range(1, max_level+1):
            print(f"Computing level-{k} policies...")
            # Evader k-level
            mu_evader_k, _ = self.value_iteration_policy(
                opponent_policy=pursuer_policies[k - 1], 
                opt_type='MIN',
                grid=grid
            )
            evader_policies[k] = mu_evader_k
            
            # Pursuer k-level
            mu_pursuer_k, _ = self.value_iteration_policy(
                opponent_policy=evader_policies[k - 1], 
                opt_type='MAX',
                grid=grid
            )
            pursuer_policies[k] = mu_pursuer_k
        
        return evader_policies, pursuer_policies
    
    def value_iteration_policy(
        self,
        opponent_policy,    # dict: state -> {a_opp_idx: prob}
        opt_type,           # 'MAX' for pursuer, 'MIN' for evader
        grid,
        epsilon=1e-3,
        max_iter=20,
        gamma=1.0,
    ):
        n_actions = len(self.action_space)

        # --- Build opponent policy as NumPy array [num_states, n_actions] ---
        # <<< MOD
        opponent_pi_array = np.zeros((self.num_states, n_actions), dtype=float)
        for s, a_dict in opponent_policy.items():
            i_s = self.state_to_idx[s]
            for a_idx, p in a_dict.items():
                opponent_pi_array[i_s, a_idx] = p
        # <<< END MOD

        # --- INIT VALUE FUNCTION as arrays: terminals = R, non-terminals = 0 ---
        # <<< MOD
        V_old = np.zeros(self.num_states, dtype=float)
        V_old[self.is_terminal_array] = self.R_array[self.is_terminal_array]
        V_new = V_old.copy()
        # <<< END MOD

        # Init policy as dict, unchanged (states -> action distributions)
        policy = {
            s: {a_idx: 0.0 for a_idx in range(n_actions)}
            for s in self.non_terminal_states
        }

        # Choose extremum type
        if opt_type == "MAX":        # pursuer
            better = lambda val, best: val > best
            init_best = -float("inf")
        else:                        # evader (MIN)
            better = lambda val, best: val < best
            init_best = float("inf")

        # Cache transitions P_h(s' | s, a_p, a_e)
        @lru_cache(maxsize=None)
        def get_transition(s, a_p_idx, a_e_idx):
            joint_action = (
                self.action_space[a_p_idx],
                self.action_space[a_e_idx],
            )
            return self.state_transition(
                s,
                joint_action,
                sigma_w=self.sigma_w,
            )

        # 2. Value iteration loop over m
        for it in range(max_iter):
            #self.plot_value_slice_pursuer(V_old, evader_pos=(5,5), title=f"Value function at iteration {it+1}")
            #self.plot_value_slice_evader(V_old, pursuer_pos=(5,5), title=f"Value function at iteration {it+1}")
            print(f"  Value iteration it={it+1}...")
            delta = 0.0

            # Loop over all non-terminal states using indices
            for i_s in self.non_terminal_indices:
                s = self.idx_to_state[i_s]           # state tuple
                opp_pi_row = opponent_pi_array[i_s]  # shape (n_actions,)

                best_value = init_best
                best_actions = []

                # Loop over agent's own actions (indexed)
                for a_i in range(n_actions):
                    expected_value = 0.0

                    # Marginalize over opponent actions θ^{-i}
                    for a_opp_i, p_opp in enumerate(opp_pi_row):
                        if p_opp == 0.0:
                            continue

                        if opt_type == "MAX":
                            a_p_idx, a_e_idx = a_i, a_opp_i
                        else:
                            a_p_idx, a_e_idx = a_opp_i, a_i

                        # use cached transition
                        next_state_probs = get_transition(s, a_p_idx, a_e_idx)

                        # Bellman backup with reward + discount
                        for s_next, p_s in next_state_probs.items():
                            j = self.state_to_idx[s_next]  # successor index
                            if self.is_terminal_array[j]:
                                expected_value += p_opp * p_s * V_old[j]  # = R(s')
                            else:
                                expected_value += p_opp * p_s * (gamma * V_old[j])

                    # extremum over actions
                    if better(expected_value, best_value):
                        best_value = expected_value
                        best_actions = [a_i]
                    elif np.isclose(expected_value, best_value, atol=1e-10):
                        best_actions.append(a_i)

                # Update V_{m+1}(s_i)
                V_new[i_s] = best_value
                delta = max(delta, abs(V_new[i_s] - V_old[i_s]))

                # Extract greedy policy π^i,(k)(·|s): uniform over best_actions
                prob_br = 1.0 / len(best_actions)
                s_tuple = self.idx_to_state[i_s]
                policy[s_tuple] = {
                    a_idx: (prob_br if a_idx in best_actions else 0.0)
                    for a_idx in range(n_actions)
                }

            # swap buffers
            V_old, V_new = V_new, V_old

            print(f"    delta = {delta:.3e}")
            if delta < epsilon:
                print(f"Converged after {it+1} iterations.")
                break

        # return the final V as a dict too (if you want), or as an array
        V_final = {s: V_old[self.state_to_idx[s]] for s in self.state_space}
        return policy, V_final


    # =================== Oppoent Level Inferring ====================== #
    # Level inferring algorithm
    def step_transition_prob(
        self,
        s, s_next, my_level, opp_level,
        my_policies, opp_policies
    ):
        mu_self = my_policies[my_level][s]
        mu_opp  = opp_policies[opp_level][s]

        prob = 0.0
        for a_i, p_i in mu_self.items():
            for a_o, p_o in mu_opp.items():
                joint_action = (
                    self.action_space[a_i],
                    self.action_space[a_o]
                )
                trans = self.state_transition(
                    s, joint_action,
                    sigma_w=self.sigma_w
                )
                prob += p_i * p_o * trans.get(s_next, 0.0)

        return prob


    def infer_opponent_level_ML(
        self,
        trajectory_states,   # list [s_{N-w}, ..., s_N], length w+1
        my_level_window,     # list [k^i_{N-w}, ..., k^i_{N-1}], length w
        candidate_levels,    # iterable of candidate opponent levels, e.g. range(k_max)
        my_policies,
        opp_policies,
        eps=1e-12
    ):
        """
        maximum likelihood estimate of opponent level.

        Returns:
          best_level      - argmax_k P(s^{[N-w,N]} | k_i^{[N-w,N]}, k)
          best_log_prob   - log-likelihood of the best level
        """
        assert len(trajectory_states) == len(my_level_window) + 1
        w = len(my_level_window)

        best_level = None
        best_log_prob = -float("inf")

        for k_opp in candidate_levels:
            logp = 0.0
            for n in range(w):
                s     = trajectory_states[n]
                s_next = trajectory_states[n + 1]
                my_lvl = my_level_window[n]

                p_step = self.step_transition_prob(
                    s, s_next, my_lvl, k_opp,
                    my_policies, opp_policies
                )
                # accumulate log-likelihood; add eps to avoid log(0)
                logp += math.log(p_step + eps)

            if logp > best_log_prob:
                best_log_prob = logp
                best_level = k_opp

        return best_level, best_log_prob


    # ========= Dynamically adjust the rationality level to countermeasure the opponent ===========
    def choose_next_my_level(self,estimated_opp_level, my_level_max):
        """
        k_{N+1}^i = min{ \hat{k}_N^{-i} + 1, k_max^i }.
        """
        return min(estimated_opp_level + 1, my_level_max)


    # =================== Debug & Visualization Helpers ====================== #
    def plot_value_slice_pursuer(self, V_array, evader_pos, title=None, clim=None):
        """
        Plot V(x1, y1, x2*, y2*) as a heatmap over pursuer positions (x1,y1),
        fixing the evader position to evader_pos = (x2*, y2*).

        V_array: 1D np.array of shape (num_states,), indexed by self.state_to_idx.
        """
        H, W = self.grid.shape
        x2_fixed, y2_fixed = evader_pos

        V_map = np.full((H, W), np.nan, dtype=float)

        for y1 in range(H):
            for x1 in range(W):
                s = (x1, y1, x2_fixed, y2_fixed)
                s_idx = self.state_to_idx.get(s, None)
                if s_idx is None:
                    continue

                V_map[y1, x1] = V_array[s_idx]

        plt.figure(figsize=(6, 5))
        if clim is not None:
            im = plt.imshow(V_map, origin="lower", cmap="coolwarm",
                            vmin=clim[0], vmax=clim[1])
        else:
            im = plt.imshow(V_map, origin="lower", cmap="coolwarm")

        plt.colorbar(im, label="Value V(s)")
        plt.title(title or f"Value for pursuer (evader at {evader_pos})")

        # Obstacles and Evasion region
        obs_y, obs_x = np.where(self.grid == 1)
        E_y,   E_x   = np.where(self.grid == 2)
        plt.scatter(obs_x, obs_y, marker='s', s=30, c='black', label="Obstacle")
        plt.scatter(E_x, E_y, marker='s', s=30, c='red',   label="Region E")

        # evader
        plt.scatter([x2_fixed], [y2_fixed], c='green', s=80, edgecolor='k', label="Evader fixed")

        plt.xlabel("x1 (pursuer)")
        plt.ylabel("y1 (pursuer)")
        plt.gca().set_aspect("equal")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()
        
    def plot_value_slice_evader(self, V_array, pursuer_pos, title=None, clim=None):
        """
        Plot V(x1*, y1*, x2, y2) as a heatmap over evader positions (x2,y2),
        fixing the pursuer position to pursuer_pos = (x1*, y1*).

        V_array: 1D np.array of shape (num_states,), indexed by self.state_to_idx.
        """
        H, W = self.grid.shape
        x1_fixed, y1_fixed = pursuer_pos

        V_map = np.full((H, W), np.nan, dtype=float)

        for y2 in range(H):
            for x2 in range(W):
                s = (x1_fixed, y1_fixed, x2, y2)
                s_idx = self.state_to_idx.get(s, None)
                if s_idx is None:
                    continue

                V_map[y2, x2] = V_array[s_idx]

        plt.figure(figsize=(6, 5))
        if clim is not None:
            im = plt.imshow(V_map, origin="lower", cmap="coolwarm",
                            vmin=clim[0], vmax=clim[1])
        else:
            im = plt.imshow(V_map, origin="lower", cmap="coolwarm")

        plt.colorbar(im, label="Value V(s)")
        plt.title(title or f"Value for evader (pursuer at {pursuer_pos})")

        # overlay the game grid (obstacles, E region)
        obs_y, obs_x = np.where(self.grid == 1)
        E_y,   E_x   = np.where(self.grid == 2)
        plt.scatter(obs_x, obs_y, marker='s', s=30, c='black', label="Obstacle")
        plt.scatter(E_x, E_y, marker='s', s=30, c='red',   label="Region E")

        # mark pursuer fixed position
        plt.scatter([x1_fixed], [y1_fixed], c='blue', s=80, edgecolor='k', label="Pursuer fixed")

        plt.xlabel("x2 (evader)")
        plt.ylabel("y2 (evader)")
        plt.gca().set_aspect("equal")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()
    
    def animate_trajectory(self, trajectory, title="Pursuit–Evader Animation",
                           interval=200, save_path=None):
        """
        Animate a trajectory:
          trajectory: list of states [(x1,y1,x2,y2), ...]
          interval: time between frames in ms
          save_path: if not None, save as e.g. 'traj.mp4' or 'traj.gif'
        """
        H, W = self.grid.shape

        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[self.grid == 0] = [255, 255, 255]  # free = white
        img[self.grid == 1] = [0,   0,   0]    # obstacle = black
        img[self.grid == 2] = [255, 0,   0]    # evader region E = red

        px = [s[0] for s in trajectory]
        py = [s[1] for s in trajectory]
        ex = [s[2] for s in trajectory]
        ey = [s[3] for s in trajectory]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img)
        ax.set_title(title)
        ax.invert_yaxis()
        ax.set_aspect("equal")

        pursuer_path_line, = ax.plot([], [], '-o', color='blue', markersize=3,
                                     label="Pursuer path")
        evader_path_line,  = ax.plot([], [], '-o', color='green', markersize=3,
                                     label="Evader path")

        # Current positions as separate artists
        pursuer_point, = ax.plot([], [], 'o', markersize=8,
                                 color='cyan', markeredgecolor='black',
                                 label="Pursuer")
        evader_point,  = ax.plot([], [], 'o', markersize=8,
                                 color='lime', markeredgecolor='black',
                                 label="Evader")

        ax.legend(loc='lower right')

        def init():
            pursuer_path_line.set_data([], [])
            evader_path_line.set_data([], [])
            pursuer_point.set_data([], [])
            evader_point.set_data([], [])
            return pursuer_path_line, evader_path_line, pursuer_point, evader_point

        def update(k):
            # paths up to step k
            pursuer_path_line.set_data(px[:k+1], py[:k+1])
            evader_path_line.set_data(ex[:k+1], ey[:k+1])

            pursuer_point.set_data([px[k]], [py[k]]) 
            evader_point.set_data([ex[k]], [ey[k]])

            return pursuer_path_line, evader_path_line, pursuer_point, evader_point

        anim = animation.FuncAnimation(
            fig, update,
            init_func=init,
            frames=len(trajectory),
            interval=interval,
            blit=True
        )
        
        if save_path is not None:
            if save_path.lower().endswith('.gif'):
                writer = 'imagemagick'
            elif save_path.lower().endswith(('.mp4', '.mov')):
                writer = 'ffmpeg'
            else:
                print(f"Warning: Unknown file extension '{save_path}', defaulting to ffmpeg.")
                writer = 'ffmpeg'
            
            fps = 1000.0 / interval
            try:
                anim.save(
                    save_path, 
                    writer=writer, 
                    fps=fps, 
                    dpi=100 # Lower DPI is usually fine for GIFs
                )
                print(f"Animation saved successfully to {save_path} using {writer}.")
            except Exception as e:
                print(f"\n--- ERROR SAVING ANIMATION ---")
                print(f"Failed to save using writer '{writer}'. Check if ImageMagick (for GIF) or ffmpeg (for MP4) is installed and configured in your system PATH.")
                print(f"Original error: {e}")
            print(f"Animation saved to {save_path}")

        plt.show()