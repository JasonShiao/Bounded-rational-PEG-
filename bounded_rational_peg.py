from enum import Enum
import math
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt

# Parameters
@dataclass
class GameParams:
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
        self.capture_radius = params.capture_radius
        self.action_space = params.action_space
        self.sigma_w = params.sigma_w # wind noise std dev
        self.grid = params.game_grid
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
            dtype=float
        )
        self.non_terminal_states = self.get_non_terminal_states(
            self.state_space, self.grid, self.capture_radius
        )
        # TODO: generate wind_map
        self.wind_map = {s: (0.1, 0.1) for s in self.position_space} # for each position (x,y), return (wx, wy)
        # precompute k-level policies
        
    
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
        plt.legend()
        plt.show()
        
    def b_vec(self, state, joint_action, wind_map, v):
        # Drift vector b for each state_entry
        # v: [v1, v2] # speed of evader and pursuer
        w = [ wind_map((state[0], state[1])), wind_map((state[2], state[3])) ] # wind at pursuer pos
        # w: [wx1(p1), wy1(p1), wx2(p2), wy2(p2)] # mean wind speed at positions of p1 and p2
        b_vec = np.zeros(4)
        for i in range(2):
            b_vec[2*i+0] = v[i] * np.cos(joint_action[i]) + w[i][0]
            b_vec[2*i+1] = v[i] * np.sin(joint_action[i]) + w[i][1]

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

    def state_transition(self, state, joint_action, h, sigma_w):
        # TODO: Check this is correct
        # P_h(s_h'| s_h, theta)
        # ret: dict mapping s_h' to probability
        sigma2 = sigma_w ** 2
        # b_vect represent the drift velocity (agent vel + wind speed)
        b_vec = self.b_vec(state, joint_action, self.wind_map, self.v)  # length 4
        Q = self.Q_h(state, h, sigma_w)

        probs = {}
        total = 0.0

        for j in range(4):
            e_j = np.zeros(4)
            e_j[j] = 1.0

            b_j = b_vec[j]
            b_plus = max(b_j, 0.0)
            b_minus = max(-b_j, 0.0)

            # s_h + h e_j
            s_plus = tuple(state + h * e_j)
            p_plus = (sigma2 / 2.0 + h * b_plus) / Q
            probs[s_plus] = probs.get(s_plus, 0.0) + p_plus
            total += p_plus

            # s_h - h e_j
            s_minus = tuple(state - h * e_j)
            p_minus = (sigma2 / 2.0 + h * b_minus) / Q
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
            return -1
        elif self.is_crash(evader_pos, game_grid) and self.is_crash(pursuer_pos, game_grid):
            return 0
        elif not self.is_crash(evader_pos, game_grid) and self.is_crash(pursuer_pos, game_grid):
            return -1
        else: # both not crash
            # captured or evaded
            distance = np.linalg.norm(np.array(pursuer_pos) - np.array(evader_pos))
            if distance <= capture_radius: # captured
                return 1
            elif self.is_in_region_E(evader_pos, game_grid):
                return -1
            else:
                return 0
    

    # ================== Terminal States ========================= #
    def is_crash(self, pos, game_grid: np.ndarray):
        """
        Simple crash test:
          - outside grid bounds
          - or cell == 1 (obstacle)
        """
        x, y = int(round(pos[0])), int(round(pos[1]))
        H, W = game_grid.shape
        if x < 0 or x >= W or y < 0 or y >= H:
            return True
        return game_grid[y, x] == 1  # obstacle

    def is_in_region_E(self, pos, game_grid: np.ndarray):
        x, y = int(round(pos[0])), int(round(pos[1]))
        H, W = game_grid.shape
        if x < 0 or x >= W or y < 0 or y >= H:
            return False
        return game_grid[y, x] == 2  # evader region E

    def is_terminal(self, state, game_grid: np.ndarray, capture_radius):
        pursuer_pos = (state[0], state[1])
        evader_pos = (state[2], state[3])
        
        # Check for crash
        if self.is_crash(evader_pos, game_grid) or self.is_crash(pursuer_pos, game_grid):
            return True
        
        # Check for capture
        distance = np.linalg.norm(np.array(pursuer_pos) - np.array(evader_pos))
        if distance <= capture_radius:
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
        evader_policies[0] = {a_i: prob_L0 for a_i in range(len(self.action_space))}
        pursuer_policies[0] = {a_i: prob_L0 for a_i in range(len(self.action_space))}
        # k-level: best response to (k-1)-level opponent
        for k in range(1, max_level+1):
            # Evader k-level
            mu_evader_k = self.value_iteration_policy(
                opponent_policy=pursuer_policies[k - 1], 
                opt_type='MIN',
                grid=grid, 
                reward_function=reward_function
            )
            evader_policies[k] = mu_evader_k
            
            # Pursuer k-level
            mu_pursuer_k = self.value_iteration_policy(
                opponent_policy=evader_policies[k - 1], 
                opt_type='MAX',
                grid=grid, 
                reward_function=reward_function
            )
            pursuer_policies[k] = mu_pursuer_k
        
        return evader_policies, pursuer_policies
    
    def value_iteration_policy(
        self,
        opponent_policy,    # dict: state -> {a_opp_idx: prob}
        opt_type,           # 'MAX' for pursuer, 'MIN' for evader
        grid,
        epsilon=1e-5,
        max_iter=30,
    ):
        n_actions = len(self.action_space)

        # 1. Init V(s): terminal states get G_h(s) = reward_function(s),
        #               non-terminals start from 0.
        V = {}
        for s in self.state_space:
            Gh = self.reward_function(s, grid, self.capture_radius)  # ±1 or 0
            V[s] = Gh

        # Init policy: we’ll overwrite for non-terminal states
        policy = {s: {a_idx: 0.0 for a_idx in range(n_actions)}
                  for s in self.non_terminal_states}

        # Choose extremum type
        if opt_type == "MAX":        # pursuer
            better = lambda val, best: val > best + 1e-12
            init_best = -float("inf")
        else:                        # evader (MIN)
            better = lambda val, best: val < best - 1e-12
            init_best = float("inf")

        # 2. Value iteration loop over m
        for it in range(max_iter):
            V_old = V.copy()
            delta = 0.0

            # Loop over all non-terminal states s ∈ S_h^o
            for s in self.non_terminal_states:

                # opponent policy can be state-dependent: π^{-i}(·|s)
                # if you’re using a single stationary policy, opponent_policy
                # can itself just be {a_opp_idx: prob}, and you can set:
                # opp_pi = opponent_policy
                opp_pi = opponent_policy[s]

                best_value = init_best
                best_actions = []

                # Loop over agent's own actions (indexed)
                for a_i in range(n_actions):
                    expected_value = 0.0

                    # Marginalize over opponent actions θ^{-i}
                    for a_opp_i, p_opp in opp_pi.items():
                        # Order of actions in the joint action:
                        # MAX (pursuer):   (a_self, a_opp) = (a_p, a_e)
                        # MIN (evader):    (a_self, a_opp) = (a_e, a_p)
                        if opt_type == "MAX":
                            a_p_idx, a_e_idx = a_i, a_opp_i
                        else:
                            a_p_idx, a_e_idx = a_opp_i, a_i

                        joint_action = (
                            self.action_space[a_p_idx],
                            self.action_space[a_e_idx],
                        )

                        # P_h(s' | s, θ) from your cMDP construction
                        next_state_probs = self.state_transition(
                            s,
                            joint_action,
                            h=1.0,
                            sigma_w=self.sigma_w,
                        )

                        # Bellman backup: sum_{s'} P(s'|s,a) V_old(s')
                        for s_next, p_s in next_state_probs.items():
                            expected_value += p_opp * p_s * V_old[s_next]

                    # extremum over actions
                    if better(expected_value, best_value):
                        best_value = expected_value
                        best_actions = [a_i]
                    elif np.isclose(expected_value, best_value, atol=1e-10):
                        best_actions.append(a_i)

                # Update V_{m+1}(s)
                V[s] = best_value
                delta = max(delta, abs(best_value - V_old[s]))

                # 4. Extract greedy policy π^i,(k)(·|s): uniform over best_actions
                prob_br = 1.0 / len(best_actions)
                policy[s] = {
                    a_idx: (prob_br if a_idx in best_actions else 0.0)
                    for a_idx in range(n_actions)
                }

            if delta < epsilon:
                # print(f"Converged after {it+1} iterations, delta={delta:.2e}")
                break

        # print(f"Value iteration ended in {it+1} iterations. Final delta={delta:.2e}")
        return policy, V

    # =================== Oppoent Level Inferring ====================== #
    # Level inferring algorithm
    def step_transition_prob(
        self,
        s, s_next, my_level, opp_level,
        my_policies, opp_policies,
        action_space, transition_model):
        """
        Compute specific step transition probability:
          P_h(s_next | s, mu_i^{(my_level)}, mu_-i^{(opp_level)})
        = sum_{a_i,a_opp} P_h(s_next | s,a_i,a_opp) * mu_i(s,a_i) * mu_opp(s,a_opp)
        """
        mu_self = my_policies[my_level][s]     # dict {a: prob}
        mu_opp  = opp_policies[opp_level][s]   # dict {a: prob}

        prob = 0.0
        for a_i, p_i in mu_self.items():
            for a_o, p_o in mu_opp.items():
                trans = transition_model(s, a_i, a_o)   # dict {s_next: p}
                prob += p_i * p_o * trans.get(s_next, 0.0)

        return prob


    def infer_opponent_level_ML(
        self,
        trajectory_states,   # list [s_{N-w}, ..., s_N], length w+1
        my_level_window,     # list [k^i_{N-w}, ..., k^i_{N-1}], length w
        candidate_levels,    # iterable of candidate opponent levels, e.g. range(k_max)
        my_policies,
        opp_policies,
        action_space,
        transition_model,
        eps=1e-12,
    ):
        """
        Implement Eq. (14)-(15): maximum likelihood estimate of opponent level.

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
                    my_policies, opp_policies,
                    action_space, transition_model,
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
        Implements k_{N+1}^i = min{ \hat{k}_N^{-i} + 1, k_max^i }.
        """
        return min(estimated_opp_level + 1, my_level_max)
