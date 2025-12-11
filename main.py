from dataclasses import dataclass
from bounded_rational_peg import PursuitEvasionGame, GameParams
import numpy as np


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
        wind_sigma=0.4,
        wind_spatial_correlation=0.0,
        game_grid=sample_grid
    )

    game = PursuitEvasionGame(params)
    game.plot_state()