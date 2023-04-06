config = {
    "ticker": 'GE_10y_1d',
    "start_end": None,

    "trading_cost_bps": 1e-4,
    "time_cost_bps": 1e-5,

    "n_episodes": 2000,
    "steps_per_episode": 25,

    "gamma": .99,  # discount factor
    "tau": 100,  # target network update frequency
    "architecture": (128, 256),  # neurons per layer
    "learning_rate": 0.001,  # learning rate
    "l2_reg": 1e-6,  # L2 regularization
    "replay_capacity": int(1e6),
    "batch_size": 1500,

    "epsilon_start": 1.0,
    "epsilon_end": .01,
    "epsilon_decay_steps": 500,
    "epsilon_exponential_decay": .99
}
