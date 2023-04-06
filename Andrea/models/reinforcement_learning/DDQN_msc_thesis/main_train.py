import os.path
from time import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import gym

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from DDQN_trading import DDQNAgent
from DDQN_trading.config import config
from DDQN_trading import utils


# ------------------------------------------------------------------------ SETTINGS ------------------------------------
# --- shutdown warnings
import warnings
for i in (UserWarning, FutureWarning, RuntimeWarning, DeprecationWarning):
    warnings.simplefilter("ignore", i)
# --- matplotlib fixing
mpl.use('TkAgg')
# --- seeds and seaborn
np.random.seed(42)
tf.random.set_seed(42)
sns.set_style('whitegrid')
# --- GPU setting
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    print('Using GPU')
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
else:
    print('Using CPU')
# ----------------------------------------------------------------------------------------------------------------------

dt_str = utils.get_timestamp_for_file()
results_path = utils.get_results_path()
if not results_path.exists():
    results_path.mkdir(parents=True)

# ----------------------------------------------------- Setup Gym Environment
gym.register(
    id='trading-v0',
    entry_point='trading_environment:TradingEnvironment',
    max_episode_steps=config["steps_per_episode"]
)
print(f'Trading costs: {config["trading_cost_bps"]:.2%} | Time costs: {config["time_cost_bps"]:.2%}')
trading_environment = gym.make(
    'trading-v0',
    ticker=config["ticker"],
    steps_per_episode=config["steps_per_episode"],
    trading_cost_bps=config["trading_cost_bps"],
    time_cost_bps=config["time_cost_bps"],
    start_end=config["start_end"]
)
trading_environment.seed(42)

# ----------------------------------------------------- Get Environment Parameters
state_dim = trading_environment.observation_space.shape[0]
num_actions = trading_environment.action_space.n
max_episode_steps = trading_environment.spec.max_episode_steps

# ======================================================================================================================
# ----------------------------------------------------- Create a DDQN-Agent --------------------------------------------
# ======================================================================================================================
tf.keras.backend.clear_session()
ddqn = DDQNAgent(
    state_dim=state_dim,
    num_actions=num_actions,
    learning_rate=config["learning_rate"],
    gamma=config["gamma"],
    epsilon_start=config["epsilon_start"],
    epsilon_end=config["epsilon_end"],
    epsilon_decay_steps=config["epsilon_decay_steps"],
    epsilon_exponential_decay=config["epsilon_exponential_decay"],
    replay_capacity=config["replay_capacity"],
    architecture=config["architecture"],
    l2_reg=config["l2_reg"],
    tau=config["tau"],
    batch_size=config["batch_size"]
)
# --------------------------------------------------- print DDQN-Online-ANN architecture
print(ddqn.online_network.summary())

# -------------------------------------------------- Run-experiment
# ------------- Set parameters
# total_steps = 0

# ------------- Initialize variables
episode_time, navs, market_navs, diffs, episode_eps = [], [], [], [], []

# ---------------------------------------------------- TRAIN AGENT
start = time()
results = []
for episode in range(1, config["n_episodes"] + 1):
    this_state = trading_environment.reset()  # reset to 0 the environment due to new episode was started
    # iterate over the episode's steps
    for episode_step in range(max_episode_steps):
        action = ddqn.epsilon_greedy_policy(this_state.reshape(-1, state_dim))  # get an action
        next_state, reward, done, _, __ = trading_environment.step(action)  # given the action get S', R(t+1) and done

        ddqn.memorize_transition(s=this_state, a=action, r=reward, s_prime=next_state, not_done=0.0 if done else 1.0)

        # if we have to train ANN, do the experience replay approach to update ANNs models
        if ddqn.train:
            # if experience has enough obs (>= batch_size) re-train ANN each_step! (update target-ANN each tau steps)
            ddqn.experience_replay()
        if done:
            break
        this_state = next_state  # update current state with the next one

    # get DataFrame with sequence of actions, returns and nav values
    result = trading_environment.env.simulator.result()

    # get results of last step
    final = result.iloc[-1]

    # apply return (net of cost) of last action to last starting nav
    nav = final.nav * (1 + final.strategy_return)
    navs.append(nav)

    # market nav
    market_nav = final.market_nav
    market_navs.append(market_nav)

    # track difference between agent an market NAV results
    diff = nav - market_nav
    diffs.append(diff)

    # every 10 episode, print the temporary-results
    if episode % 10 == 0:
        utils.track_results(
            episode,
            # show mov. average results for 100 (10) periods
            np.mean(navs[-100:]),
            np.mean(navs[-10:]),
            np.mean(market_navs[-100:]),
            np.mean(market_navs[-10:]),
            # share of agent wins, defined as higher ending nav
            np.sum([s > 0 for s in diffs[-100:]]) / min(len(diffs), 100),
            time() - start,
            ddqn.epsilon
        )

    if len(diffs) > 25 and all([r > 0 for r in diffs[-25:]]):
        print(result.tail())
        break

trading_environment.close()

# --------------------------------------------------------------- STORE results
results = pd.DataFrame(
    {
        'n_episode': list(range(1, episode + 1)),
        'agent_nav': navs,
        'mkt_nav': market_navs,
        'delta': diffs
    }
).set_index('n_episode')

results['Strategy Wins (%)'] = (results["delta"] > 0).rolling(100).sum()
results.info()

# ------------------------------------------------------------------------------------------ STORE RESULTS -------------
utils.store_results(config=config, results=results, path=results_path)  # store results
keras.models.save_model(ddqn.online_network, os.path.join(results_path, "ddqn_online_ann.h5"))  # store online_ann
keras.models.save_model(ddqn.online_network, os.path.join(results_path, "ddqn_target_ann.h5"))  # store target_ann
# ----------------------------------------------------------------------------------------------------------------------

with sns.axes_style('white'):
    sns.distplot(results.delta)
    sns.despine()
plt.show()

results.info()

fig, axes = plt.subplots(ncols=2, figsize=(14, 4), sharey=True)

df1 = (results[['agent_nav', 'mkt_nav']]
       .sub(1)
       .rolling(100)
       .mean())
df1.plot(
    ax=axes[0],
    title='Annual Returns (Moving Average)',
    lw=1
)

df2 = results['Strategy Wins (%)'].div(100).rolling(50).mean()
df2.plot(
    ax=axes[1],
    title='Agent Outperformance (%, Moving Average)'
)

for ax in axes:
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: '{:,.0f}'.format(x)))
axes[1].axhline(.5, ls='--', c='k', lw=1)

dt_str = utils.get_timestamp_for_file()
sns.despine()
fig.tight_layout()
fig.savefig(results_path / f'{dt_str}_performance', dpi=300)
plt.show()
