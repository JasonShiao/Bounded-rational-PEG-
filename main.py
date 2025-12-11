from bounded_rational_peg import PursuitEvasionGame, GameParams
import numpy as np

def sample_action_from_policy(mu_dict):
    actions = list(mu_dict.keys())
    probs   = np.array([mu_dict[a] for a in actions], dtype=float)
    probs   = probs / probs.sum()  # safety normalize
    return np.random.choice(actions, p=probs)

if __name__ == "__main__":
    # 18x18 sample grid
    sample_grid = np.array([
        [1]*18,
        [1]+[0]*16+[1],
        [1]+[0]*5+[1]*3+[0]*8+[1],  # Corrected: 8 zeros
        [1]+[0]*5+[1]*3+[0]*8+[1],  # Corrected: 8 zeros
        [1]+[0]*16+[1],
        [1]+[0]*9+[0]*7+[1],
        [1]+[0]*9+[0]*7+[1],
        [1]+[0]*16+[1],
        [1]+[0]*4+[1]*2+[0]*10+[1], # Corrected: 10 zeros
        [1]+[0]*4+[1]*2+[0]*10+[1], # Corrected: 10 zeros
        [1]+[0]*16+[1],
        [1]+[0]*6+[2]*2+[0]*2+[1]*3+[0]*3+[1], # Corrected: 4 zeros
        [1]+[0]*6+[2]*2+[0]*2+[1]*2+[0]*4+[1], # Corrected: 4 zeros
        [1]+[0]*16+[1],
        [1]+[0]*5+[1]*4+[0]*7+[1],  # Corrected: 7 zeros
        [1]+[0]*5+[1]*4+[0]*7+[1],  # Corrected: 7 zeros
        [1]+[0]*16+[1],
        [1]*18
    ])
    # Example usage
    params = GameParams(
        evader_init_pos=(2, 2),
        pursuer_init_pos=(15, 15),
        evader_speed=1.0,
        pursuer_speed=1.0,
        action_space=[0, np.pi/2, np.pi, 3*np.pi/2],
        capture_radius=1.0,
        sigma_w=0.4,
        wind_spatial_correlation=0.0,
        game_grid=sample_grid
    )

    # Plot initial state
    game = PursuitEvasionGame(params)
    game.plot_state()
    
    # Precompute level-k policies
    max_level = 2  # compute policies up to level 2 for both players

    print("Computing level-k policies...")
    evader_policies, pursuer_policies = game.level_k_policies(
            max_level=max_level,
            grid=game.grid,
            reward_function=game.reward_function,  # even if unused inside
    )
    print("Level-k policies computed.")
    
    pursuer_level = 2   # pursuer plays level-2
    evader_level  = 1   # evader plays level-1

    # simulate
    T_max = 50
    trajectory = [tuple(map(int, game.state))]

    for t in range(T_max):
        s = tuple(map(int, game.state))  # current state as indices (x1,y1,x2,y2)

        # stop if terminal
        if game.is_terminal(s, game.grid, game.capture_radius):
            print(f"Episode ended at step {t}, terminal state {s}")
            break

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
            h=1.0,
            sigma_w=game.sigma_w
        )

        # sample next state from distribution
        next_states = list(next_state_probs.keys())
        probs       = np.array(list(next_state_probs.values()), dtype=float)
        probs       = probs / probs.sum()  # safety normalize
        s_next      = next_states[np.random.choice(len(next_states), p=probs)]

        # update game.state (as float for plotting, but content is indices)
        game.state = np.array(s_next, dtype=float)
        trajectory.append(s_next)

        print(f"t={t}, s={s}, a_p={a_p_idx}, a_e={a_e_idx}, s'={s_next}")

    game.plot_trajectory(trajectory, title="Level-2 Pursuer vs Level-1 Evader")