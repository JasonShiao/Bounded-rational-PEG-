from bounded_rational_peg import PursuitEvasionGame, GameParams
import numpy as np
import pickle
import os


def sample_action_from_policy(mu_dict):
    actions = list(mu_dict.keys())
    probs   = np.array([mu_dict[a] for a in actions], dtype=float)
    probs   = probs / probs.sum()  # safety normalize
    return np.random.choice(actions, p=probs)

# Helper functions to reuse the policies
# Note: Value Iteration and Policy Extraction is time-consuming,
def save_policies(pursuer_policies_dict, evader_policies_dict, params_obj, filename):
    """Saves the policy dictionary to a file using pickle."""
    bundle = {
        'pursuer_policies': pursuer_policies_dict,
        'evader_policies': evader_policies_dict,
        'params': params_obj
    }
    try:
        with open(filename, 'wb') as f:
            pickle.dump(bundle, f)
        print(f"Successfully saved policy bundle to {filename}")
    except Exception as e:
        print(f"Error saving policies: {e}")

def load_policy_bundle(filename):
    """Loads the policy and the associated parameters."""
    try:
        with open(filename, 'rb') as f:
            bundle = pickle.load(f)
        print(f"Successfully loaded policy bundle from {filename}")
        return bundle['pursuer_policies'], bundle['evader_policies'], bundle['params']
    except Exception as e:
        print(f"Error loading policies: {e}")
        return None, None, None


if __name__ == "__main__":
    game_filename = 'game_level_k_policies4.pkl'
    # medium grid: 18x18
    sample_grid = np.array([
        [1]*18,
        [1]+[0]*14+[2]*0+[1]*2+[1],
        [1]+[0]*5+[1]*3+[0]*3+[2]*0+[0]*5+[1],  # Corrected: 8 zeros
        [1]+[0]*5+[1]*3+[0]*8+[1],  # Corrected: 8 zeros
        [1]+[0]*16+[1],
        [1]+[0]*9+[0]*7+[1],
        [1]+[0]*9+[0]*7+[1],
        [1]+[0]*16+[1],
        [1]+[0]*4+[1]*2+[0]*10+[1], # Corrected: 10 zeros
        [1]+[0]*4+[1]*2+[0]*10+[1], # Corrected: 10 zeros
        [1]+[0]*8+[2]*0+[0]*2+[1]*1+[0]*5+[1], # Corrected: 4 zeros
        [1]+[0]*7+[2]*1+[0]*2+[1]*0+[0]*6+[1], # Corrected: 4 zeros
        [1]+[0]*16+[1],
        [1]+[0]*16+[1],
        [1]+[0]*5+[1]*4+[0]*7+[1],  # Corrected: 7 zeros
        [1]+[0]*5+[1]*4+[0]*7+[1],  # Corrected: 7 zeros
        [1]+[0]*16+[1],
        [1]*18
    ])
    # small grid: 9x9
    # sample_grid = np.array([
    #     [1]*9,
    #     [1]+[0]*7+[1],
    #     [1]+[0]*3+[1]*2+[0]*2+[1],  # Corrected: 8 zeros
    #     [1]+[0]*4+[1]*1+[0]*2+[1],  # Corrected: 8 zeros
    #     [1]+[0]*7+[1],
    #     [1]+[0]*7+[1],
    #     [1]+[0]*2+[2]*1+[0]*2+[1]*1+[0]*1+[1], # Corrected: 4 zeros
    #     [1]+[0]*2+[2]*2+[0]*1+[1]*1+[0]*1+[1], # Corrected: 4 zeros
    #     [1]+[0]*3+[1]*2+[0]*2+[1],  # Corrected: 7 zeros
    #     [1]*9
    # ])


    if not os.path.exists(game_filename):
        print("Precomputed policy files not found. Computing level-k policies...")
        params = GameParams(
            h=1.0,
            evader_init_pos=(2, 2),
            pursuer_init_pos=(14, 13), # (4,5)
            evader_speed=1.0,
            pursuer_speed=1.0,
            action_space=[0, np.pi/2, np.pi, 3*np.pi/2],
            capture_radius=0.5,
            sigma_w=0.3,
            wind_spatial_correlation=0.0,
            game_grid=sample_grid
        )

        # Plot initial state
        game = PursuitEvasionGame(params)
        
        # Precompute level-k policies
        max_level = 4  # compute policies up to level 2 for both players

        
        evader_policies, pursuer_policies = game.level_k_policies(
                max_level=max_level,
                grid=game.grid,
                reward_function=game.reward_function,  # even if unused inside
        )
        save_policies(pursuer_policies, evader_policies, params, game_filename)
        print("Level-k policies computed.")
    else:        
        print("Precomputed policy files and parameters found. Skipping policy computation.")
        # Load precomputed policies
        pursuer_policies, evader_policies, loaded_params = load_policy_bundle(game_filename)
        
        game = PursuitEvasionGame(loaded_params)
    
    # Plot initial state
    game.plot_state()
    
    
    pursuer_level = 4   # pursuer plays level-4
    evader_level  = 2   # evader plays level-2
    
    total_episodes = 1500
    pursuer_wins = 0
    evader_wins = 0
    for exp in range(total_episodes): # experiment 1500 episodes for the same map
        
        game.random_init_state()  # random non-terminal init state
        
        # simulate
        T_max = 80
        trajectory = [tuple(map(int, game.state))]
        total_time_elapsed = 0.0

        for t in range(T_max):
            s = tuple(map(int, game.state))  # current state as indices (x1,y1,x2,y2)

            # stop if terminal
            if game.is_terminal(s, game.grid, game.capture_radius):
                print(f"Episode ended at step {t}, terminal state {s}")
                if game.is_crash(s[0:2], game.grid):
                    evader_wins += 1
                    print("Pursuer crashed!")
                elif game.is_crash(s[2:4], game.grid):
                    pursuer_wins += 1
                    print("Evader crashed!")
                elif game.is_captured(s):
                    pursuer_wins += 1
                    print("Pursuer captured the evader!")
                elif game.is_in_region_E(s[2:4], game.grid):
                    evader_wins += 1
                    print("Evader escaped!")
                break


            Q_h_val = game.Q_h(s, h=game.h, sigma_w=game.sigma_w)
            # Calculate Delta t_h(s) = h^2 / Q_h(s)
            delta_t = (game.h ** 2) / Q_h_val # <<< NEW: Calculate time step
            total_time_elapsed += delta_t       # <<< NEW: Accumulate time

            # get per-state policies at the chosen levels
            pi_p = pursuer_policies[pursuer_level][s]   # {a_idx: prob}
            pi_e = evader_policies[evader_level][s]    # {a_idx: prob}

            # sample discrete action indices
            a_p_idx = sample_action_from_policy(pi_p)
            a_e_idx = sample_action_from_policy(pi_e)

            # map indices to actual headings
            joint_action = (
                game.action_space[a_p_idx],
                game.action_space[a_e_idx]
            )

            # transition: P_h(s' | s, joint_action)
            next_state_probs = game.state_transition(
                s,
                joint_action,
                sigma_w=game.sigma_w
            )

            # Stochastic transition: sample next state from distribution
            next_states = list(next_state_probs.keys())
            probs       = np.array(list(next_state_probs.values()), dtype=float)
            probs       = probs / probs.sum()  # safety normalize
            s_next      = next_states[np.random.choice(len(next_states), p=probs)]

            # update game.state (as float for plotting, but content is indices)
            game.state = np.array(s_next, dtype=float)
            trajectory.append(s_next)

            print(f"n={t}, s={s}, a_p={a_p_idx}, a_e={a_e_idx}, s'={s_next}, T_elapsed={total_time_elapsed:.3f}s")

    # Animate a sample (the last) game trajectory
    game.animate_trajectory(trajectory,
                        title=f"Level-{pursuer_level} Pursuer vs Level-{evader_level} Evader",
                        interval=80,                # ms per frame
                        save_path="peg_traj.gif")   # or "peg_traj.mp4"

    pursuer_win_rate = pursuer_wins / total_episodes
    evader_win_rate = evader_wins / total_episodes
    print("\n--- Simulation Results ---")
    print("Pursuer Level:", pursuer_level)
    print("Evader Level:", evader_level)
    print(f"Total Episodes Run: {total_episodes}")
    print(f"Pursuer Wins: {pursuer_wins} ({pursuer_win_rate:.2%})")
    print(f"Evader Wins: {evader_wins} ({evader_win_rate:.2%})")