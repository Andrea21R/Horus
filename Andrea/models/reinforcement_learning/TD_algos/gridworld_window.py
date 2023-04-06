import numpy as np
import random
from itertools import product
from copy import deepcopy
from typing import Optional, Tuple, Union, Callable, Dict, List
import matplotlib.pyplot as plt
import pandas as pd


def get_allowed_action(state: Tuple[int, int]) -> List[int]:
    actions = [0, 1, 2, 3]

    row, col = state
    if row == 0:
        actions.remove(1)
    if row == 6:
        actions.remove(3)
    if col == 0:
        actions.remove(0)
    if col == 9:
        actions.remove(2)

    return actions


def create_the_game_states(random_: bool) -> Dict[tuple, Dict[int, float]]:
    """
    It's a rectangle 10x7
    """
    states = product(range(7), range(10))  # {0: left, 1: up, 2: right, 3: down}

    gridworld = {}
    for state in states:
        actions = get_allowed_action(state=state)
        if random_:
            gridworld[state] = {action: random.random() for action in actions}
        else:
            gridworld[state] = {action: 0 for action in actions}
    return gridworld


def epsilon_greedy_policy(
        greedy: bool,
        current_state: Tuple[int, int],
        states: Dict[tuple, dict],
        epsilon: Optional[float] = None
) -> Tuple[
    Tuple[int, int],
    int
]:
    """
    {0: left, 1: up, 2: right, 3: down}
    """
    allowed_actions = list(states[current_state].keys())

    if not greedy or (epsilon and (random.random() < epsilon)):
        action = random.choice(allowed_actions)
        next_state = get_next_state(current_state=current_state, action=action)
        while not next_state:
            action = random.choice(allowed_actions)
            next_state = get_next_state(current_state=current_state, action=action)
    else:
        next_states_info = {}
        for action in allowed_actions:
            next_state = get_next_state(current_state=current_state, action=action)
            # if next_state is out of the rectangle, `get_next_state` returns None
            if next_state:
                next_states_info[next_state] = {'action': action, 'qval': states[current_state][action]}

        greedy_result = max(next_states_info.items(), key=lambda x: x[1]['qval'])
        next_state, action = greedy_result[0], greedy_result[1]['action']

    return next_state, action


def insert_crosswind_bias(current_state: Tuple[int, int], next_state: Tuple[int, int]) -> Tuple[int, int]:
    """
    Hard defined for a rectangle 10x7
    """
    curr_col = current_state[1]
    next_row, next_col = deepcopy(next_state)

    if curr_col in (3, 4, 5, 8):
        next_row -= 1
    elif curr_col in (6, 7):
        next_row -= 2
    else:
        pass
    next_row = next_row if next_row >= 0 else 0

    return next_row, next_col


def get_next_state(
        current_state: Tuple[int, int],
        action: int
) -> Union[Tuple[int, int], None]:
    """
    Rectangle 10x7
    """
    tot_row, tot_col = 7 - 1, 10 - 1
    curr_row, curr_col = current_state

    if action in (0, 2):  # sx, dx
        delta = -1 if action == 0 else 1
        next_state = (curr_row, curr_col + delta)
    else:  # up, down
        delta = -1 if action == 1 else 1
        next_state = (curr_row + delta, curr_col)

    next_row, next_col = insert_crosswind_bias(current_state, next_state)

    # go out from the rectangle
    if (0 <= next_row <= tot_row) and (0 <= next_col <= tot_col):
        return next_row, next_col
    else:
        return None


def get_reward(state: Tuple[int, int], tgt_state: Tuple[int, int]) -> int:
    if state != tgt_state:
        return -1
    else:
        return 0


def update_qvalue(
        states: Dict[tuple, Dict[int, float]],
        current_state: Tuple[int, int],
        next_state: Tuple[int, int],
        action: int,
        alpha: float,
        gamma: float,
        reward: int,
        rl_method: str,
        greedy_policy: bool,
        epsilon: Optional[float] = None
) -> np.ndarray:

    if rl_method == "sarsa":
        next_next_state, next_action = epsilon_greedy_policy(
            greedy=greedy_policy,
            current_state=next_state,
            states=states,
            epsilon=epsilon
        )
        # Sarsa takes the action which the agent will made in S'
        next_qval_state_action = states[next_state][next_action]
    elif rl_method == "exp_sarsa":
        # Expected-Sarsa takes the mean of the qvalue available in S'
        next_qval_state_action = np.mean(list(states[next_state].values()))
    elif rl_method == "qlearning":
        # Q-Learning takes the action with the max qvalue in S'
        next_qval_state_action = max(states[next_state].values())
    else:
        raise KeyError("Wrong `rl_method`. You can choose among ['qlearning', 'sarsa', 'exp_sarsa']")
    old_qval = states[current_state][action]
    new_qval = old_qval + alpha * (reward + gamma * next_qval_state_action - old_qval)

    states[current_state][action] = new_qval
    return states


def store_step(current_state: tuple, action: int, next_state: tuple, reward: int) -> dict:
    return {
            'current_state': current_state,
            'action': action,
            'next_state': next_state,
            'reward': reward
        }


def run_episode(
        states: np.ndarray,
        s_0: Tuple[int, int],
        tgt_state: Tuple[int, int],
        alpha: float,
        gamma: float,
        rl_method: str,
        policy: Callable,
        greedy_policy: bool,
        epsilon: Optional[float] = None,
        verbose: bool = True,
) -> tuple:
    current_state = deepcopy(s_0)
    states_ = deepcopy(states)
    workflow = {}
    visits = {k: 0 for k in states.keys()}
    reward = None

    count = 0
    while reward != 0:
        next_state, action = policy(
            greedy=greedy_policy,
            current_state=current_state,
            states=states_,
            epsilon=epsilon
        )
        visits[next_state] += 1
        reward = get_reward(state=next_state, tgt_state=tgt_state)
        states_ = update_qvalue(
            states=states_,
            current_state=current_state,
            next_state=next_state,
            action=action,
            alpha=alpha,
            gamma=gamma,
            reward=reward,
            rl_method=rl_method,
            greedy_policy=greedy_policy,
            epsilon=epsilon
        )
        workflow[f'step{count}'] = store_step(current_state, action, next_state, reward)
        if verbose:
            print(workflow[f'step{count}'])

        current_state = next_state
        count += 1
        # ------------------------------------------------------------------------------ WIN STEP
        if next_state == tgt_state:
            workflow['win_step'] = store_step(current_state, None, None, "You Win")
            if verbose:
                print(workflow['win_step'])
                print(f'Steps to win: {len(workflow)}')
                print(f'States visits: \n{visits}')

    return states_, workflow, len(workflow) - 1, visits


def train_model(
        n_episodes: int,
        states: np.ndarray,
        s_0: Tuple[int, int],
        tgt_state: Tuple[int, int],
        alpha: float,
        gamma: float,
        rl_method: str,
        policy: Callable,
        greedy_policy: bool,
        epsilon: Optional[float] = None,
        decay_epsilon: Optional[float] = None,
        verbose_train: bool = True,
        verbose_episode: bool = False,
):
    states_ = deepcopy(states)
    workflow = {'start': {'states': states_, 'episode_workflow': None, 'n_steps': None}}
    # visits = {k: 0 for k in states.keys()}
    epsilon_ = deepcopy(epsilon)

    for episode in range(n_episodes):
        if verbose_train:
            print(f'--- EPISODE{episode} -----------------------------------------------------------------------------')

        states_, episode_workflow, n_steps, visits_ = run_episode(
            states=states_,
            s_0=s_0,
            tgt_state=tgt_state,
            alpha=alpha,
            gamma=gamma,
            rl_method=rl_method,
            policy=policy,
            greedy_policy=greedy_policy,
            epsilon=epsilon_,
            verbose=verbose_episode
        )
        if epsilon and decay_epsilon:
            epsilon_ = epsilon_ * (1 - decay_epsilon)
        # workflow[f'episode{episode}'] = {'states': states_, 'episode_workflow': episode_workflow, 'n_steps': n_steps}
        workflow[f'episode{episode}'] = {'episode_workflow': episode_workflow, 'n_steps': n_steps}
        # visits = {k: v + visits_[k] for k,v in visits.items()}

        if verbose_train:
            print(f'steps required: {n_steps}')

    # states_workflow = {k: v['states'] for k,v in workflow.items()}
    episodes_workflow = {k: v['episode_workflow'] for k,v in workflow.items()}
    n_steps = pd.Series({k: v['n_steps'] for k,v in workflow.items()}, name=rl_method)

    # return states_workflow, episodes_workflow, n_steps
    return episodes_workflow, states_, n_steps


def make_learning_graph(n_steps: Union[pd.Series, pd.DataFrame], rl_method: Optional[str] = None) -> None:
    plt.figure()
    plt.title(f"Windy GridWorld Game | {rl_method} RL", fontweight='bold', fontsize=16)
    plt.suptitle("Pag.152: Sutton & Barto 'Reinforcment Learning: an Introduction', 2nd Edition, 2022")
    plt.plot(np.arange(len(n_steps)), n_steps.values, linewidth=1.5)
    if isinstance(n_steps, pd.Series):
        first_win = n_steps.index.get_loc(n_steps[n_steps == n_steps.min()].index[0])
        plt.axvline(first_win, color='green', linestyle='--', linewidth=1)
        plt.axhline(n_steps.min(), color='red', linestyle='--', linewidth=0.5)
        plt.legend(['N. steps to win', f'first win game with minimum step ({n_steps.min()})', f'{n_steps.min()}-steps'])
    else:
        plt.legend(n_steps.columns)
    plt.grid(color='silver', linestyle='--')
    plt.ylabel('N. steps to win', fontweight='bold', style='italic')
    plt.xlabel('Episodes', fontweight='bold', style='italic')
    plt.show()


def print_final_solution(episodes_workflow: dict) -> None:
    for k, v in episodes_workflow['episode299'].items():
        print(k, v)


if __name__ == "__main__":

    # --- matplotlib fixing
    import matplotlib as mpl
    mpl.use('TkAgg')

    import warnings
    warnings.simplefilter("ignore", FutureWarning)

    rl_methods = ("sarsa", "exp_sarsa", "qlearning")

    learning_path = {k: [] for k in ('episodes_workflows', 'mapped_states', 'n_steps')}
    for rl_method in rl_methods:
        gridworld = create_the_game_states(random_=False)
        episodes_workflow, mapped_states, n_steps  = train_model(
            n_episodes=150,       # converge around 50-80th episode
            states=gridworld,
            s_0=(3, 0),
            tgt_state=(3, 7),
            alpha=0.9,            # Agent has to learn as fast as possible
            gamma=0.99,           # Future rewards are more important because agent have to learn the path
            rl_method=rl_method,
            policy=epsilon_greedy_policy,
            greedy_policy=True,
            epsilon=0.15,         # Agent has to explore a lot, especially at the beginning
            decay_epsilon=0.1,    # but I want a fast decrease in exploration
            verbose_train=True,
            verbose_episode=False
        )
        learning_path['n_steps'].append(n_steps)
        # ------------------------------------------------------------------------------------------------------------------
    n_steps = pd.concat(learning_path['n_steps'], axis=1)
    make_learning_graph(n_steps=n_steps, rl_method=None)
    # print_final_solution(episodes_workflow=episodes_workflow)
