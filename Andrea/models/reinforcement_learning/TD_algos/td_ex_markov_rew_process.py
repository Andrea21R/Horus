"""
Markov Reward Process
-------------------------------------------------------------------------
Game: -1 <-- A <-0-> B <-0-> C <-0-> D <-0-> E -> +1
s0: "c"
Algorithm solution: TD (Temporal Difference) with some "added chips" =D
-------------------------------------------------------------------------
Author: Andrea Chiaverini, Jan-2023
"""
from typing import Optional, Dict, Tuple, Callable
import random
from copy import deepcopy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class Policy():
    def __init__(self):
        self.sx_a = 0.5
        self.sx_a_sample = [0.5]
        self.dx_e = 0.5
        self.dx_e_sample = [0.5]

    def use_policy(
            self,
            greedy: bool,
            super_smart: bool,
            current_state: Optional[str] = None,
            states: Optional[dict] = None,
            epsilon: Optional[float] = None
    ) -> int:
        if greedy and (current_state not in ['a', 'e']):
            if epsilon and (random.random() < epsilon):
                return random.choice([0, 1])
            else:
                sx_state_v = states[go_left(current_state)]
                dx_state_v = states[go_right(current_state)]
                if sx_state_v > dx_state_v:
                    return 0
                else:
                    return 1
        else:
            if super_smart:
                if current_state == "a":
                    if states["b"] > self.sx_a:
                        return 1
                    else:
                        return 0
                else:  # current_state == "e"
                    if states["d"] > self.dx_e:
                        return 0
                    else:
                        return 1
            elif random.random() > 0.5:
                return 1
            else:
                return 0

    def update_terminal_value_memory(self, current_state: str):
        if current_state == "a":
            self.sx_a_sample.append(-1)
            self.sx_a = np.mean(self.sx_a_sample)
        else:
            self.dx_e_sample.append(1)
            self.dx_e = np.mean(self.dx_e_sample)


def initialize_states(states: dict, random_: bool = False) -> dict:
    "random states initialization"
    states_ = {}
    for k,v in states.items():
        if random_:
            states_[k] = random.random()
        else:
            states_[k] = 0.5
    return states_


def get_reward(current_state: str, action: int) -> int:
    if (current_state == "a") and (action == 0):
        return -1
    elif (current_state == "e") and (action == 1):
        return 1
    else:
        return 0


def go_left(current_state: str) -> str:
    return chr(ord(current_state) - 1)


def go_right(current_state: str) -> str:
    return chr(ord(current_state) + 1)


def get_next_state(current_state: str, action: int):

    if (current_state == "a" and action == 0) or (current_state == "e" and action == 1):
        return None
    else:
        if action == 0:
            return go_left(current_state=current_state)
        else:
            return go_right(current_state=current_state)


def cap_number(n: float, floor: float, top: float) -> float:
    if n < floor:
        return floor
    elif n > top:
        return top
    else:
        return n


def update_state_value(
        states: dict,
        current_state: str,
        action: int,
        alpha: float,
        reward: float,
        gamma: float
) -> Dict[str, float]:
    current_state_v = states[current_state]
    next_state = get_next_state(current_state, action)
    if not next_state:
        if current_state == "a":
            next_state_v = -1
        else:
            next_state_v = 1
    else:
        next_state_v = states[next_state]

    new_state_v = current_state_v + alpha * (reward + gamma * next_state_v - states[current_state])
    states.update({current_state: cap_number(new_state_v, 0, 1)})
    return states


def _format_states(states: dict) -> dict:
    return {k: round(v * 100, 2) for k,v in states.items()}


def run_episode(
        states: dict,
        s_0: str,
        alpha: float,
        gamma: float,
        policy: Callable,
        greedy_policy: bool,
        super_smart_policy: bool,
        epsilon: Optional[float] = None,
        verbose: bool = True
) -> Tuple[Dict[str, float], int, dict]:

    workflow = {}
    current_state = deepcopy(s_0)
    states_ = deepcopy(states)
    visits = {k: 0 for k in states_.keys()}

    count = 0
    while current_state:
        action = policy(
            greedy=greedy_policy,
            super_smart=super_smart_policy,
            current_state=current_state,
            states=states_,
            epsilon=epsilon
        )
        reward = get_reward(current_state=current_state, action=action)
        states_ = update_state_value(
            states=states_,
            current_state=current_state,
            action=action,
            alpha=alpha,
            reward=reward,
            gamma=gamma
        )
        workflow[f'step{count}'] = (current_state, action, reward)

        if verbose and current_state:
            if count == 0:
                print(f'start: {s_0}')
            print(f'step{count}: {(current_state, action, reward)} | states_values: {_format_states(states_)}')

        visits[current_state] += 1
        current_state = get_next_state(current_state=current_state, action=action)

        if not current_state:
            workflow['terminal_state'] = reward
            if verbose:
                print(f'terminal_state: {reward} | states_values: {_format_states(states_)}')

        count += 1

    return states_, reward, visits


def train_model(
        n_episods: int,
        states: dict,
        s_0: str,
        alpha: float,
        gamma: float,
        greedy_policy: bool,
        super_smart_policy: bool,
        epsilon: Optional[float] = None,
        decay_epsilon: Optional[float] = None,
        verbose: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    states_ = deepcopy(states)

    policy = Policy()
    workflow = {'start': {**states, 'reward': 0, 'avg_rew': 0}}
    visits = {k: 0 for k in states_.keys()}
    epsilon_ = deepcopy(epsilon)

    for episode in range(n_episods):
        if verbose:
            print(f'--- EPISODE{episode} -----------------------------------------------------------------------------')
        states_, reward, visits_ = run_episode(
            states=states_,
            s_0=s_0,
            alpha=alpha,
            gamma=gamma,
            policy=policy.use_policy,
            greedy_policy=greedy_policy,
            super_smart_policy=super_smart_policy,
            epsilon=epsilon_,
            verbose=verbose
        )
        if epsilon and decay_epsilon:
            epsilon_ = epsilon_ * (1 - decay_epsilon)
        if super_smart_policy:
            policy.update_terminal_value_memory(current_state="a" if reward -1 else "e")
            # print('=======')
            # print(f'sx_a: {policy.sx_a}')
            # print(f'dx_e: {policy.dx_e}')
            # print('=======')
        rewards = [reward]
        rewards.extend([d['reward'] if workflow else 0 for d in workflow.values()])
        workflow[f'episode{episode}'] = {**states_, 'reward': reward, 'avg_rew': sum(rewards) / len(rewards)}
        visits = {k: v + visits_[k] for k,v in visits.items()}
    workflow = pd.DataFrame(workflow).transpose()
    workflow.iloc[:, :-2] = workflow.iloc[:, :-2] * 100
    return workflow.round(2), pd.Series(visits, name='visits')


def plot_results(rl_workflow: pd.DataFrame, visits: pd.Series) -> None:
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle("TD (Temporal Difference) Outputs | Markov Reward Process", fontweight='bold', fontsize=20)
    grid = plt.GridSpec(2, 2, wspace=0.2, hspace=0.2)

    plot0 = fig.add_subplot(grid[0, 0])
    plot0.plot(np.arange(len(rl_workflow)), rl_workflow.iloc[:, :-2].values)
    plot0.legend(rl_workflow.columns[:-2])
    plot0.grid(linestyle='--', color='silver')
    plot0.set_ylabel('V(s) (% probability of win)', fontweight='bold', style='italic')
    plot0.set_title("State-Value function learning", fontweight='bold', fontsize=14)

    plot1 = fig.add_subplot(grid[0, 1])
    plot1.bar(x=rl_workflow.columns[:-2], height=rl_workflow.iloc[-1, :-2])
    plot1.set_title("Final V(s) for each state", fontweight='bold', fontsize=14)
    plot1.set_ylabel("V(s) (% probability of win)", style='italic')

    plot2 = fig.add_subplot(grid[1, 0])
    plot2.plot(np.arange(len(rl_workflow)), rl_workflow['avg_rew'].values)
    plot2.grid(linestyle='--', color='silver')
    plot2.set_ylabel('Average Reward', fontweight='bold')
    plot2.set_xlabel('n_episode')
    plot2.set_title("Avg-Rewards (intelligence learning)", fontweight='bold', fontsize=14)

    plot3 = fig.add_subplot(grid[1, 1])
    plot3.bar(x=visits.index, height=visits.values)
    plot3.set_title("Total N. of visits for state", fontweight='bold', fontsize=14)

    plt.show()


if __name__ == "__main__":

    # --- matplotlib fixing
    import matplotlib as mpl
    mpl.use('TkAgg')

    """
    This game is a Markov Reward Process: agent starts in a specified state (a, b, c, d or e), tipically c, and it has
    two possible action: go to right or go to left. All the action will get a reward equal to 0, but if the agent will
    go to the left when it is in state 'a', then it will lose (i.e. reward equal to -1), and if it will go to the right
    when it is in state 'e', then it will win (i.e. reward equal to 0). Obviously, as always, I want to win =D
    
    Game schema:
                         -1 <-- A <-0-> B <-0-> C <-0-> D <-0-> E -> +1
                         
    Solution Implemented: TD (Temporal Difference) Reinforcement Learning (with some added chips =D)
     
    Some tips:
    - try to change `n_samples` but the most important changes occur at the first 30-50 episodes
    - try to change `alpha`: 0.5 for a super fast learning; 0.01 for a slower learning; 0.1 is a standard
    - try to change `gamma`: 0.9 for example implies that the next state-action if really important and that is the best
        calibration for this game, because we want to win (go to the right in "e"); 0.1 for less importance for future
    - try to change `greedy_policy`: when it's True, policy is smart and is able to distinguish between two choise; if 
        it's False then it will be blind (it will choice randomly between right and left) in fact avg_reward will not
        converge to 1
    - try to change `super_smart_policy`: it's my custom implementation. It assumes that policy has memory of the 
        terminal state (i.e. left to 'a' and right to 'e'), then over time the policy will memorize a value for the TS
        
    - BEST CALIBRATION (with no usage of `super_smart` skill) is:
        . n_episods = 500  (but just from 200-300 the agent become super smart)
        . alpha = 0.5  (I want to improve the agent's knowledge fastly)
        . gamma = 0.9  (Next state value is super important because is a sequential game with a terminal state win/lose)
        . greedy_policy = True
        . super_smart_policy = False (no free lunch)
        . epsilon = 0.01  (for this game I don't want to explore a lot)
        . decay_epsilon = 0.05
        
    Enjoy the Algo 💪
    """

    states_values = initialize_states({'a': 0, 'b': 0, 'c': 0, 'd': 0, 'e': 0}, random_=False)
    rl_workflow, visits = train_model(
        n_episods=500,            # number of episodes to simulate
        states=states_values,     # initial states values
        s_0="c",                  # initial state (i.e. starting point)
        alpha=0.1,                # learning rate
        gamma=0.1,                # long term discount rate
        greedy_policy=True,       # True if you want to select state with max expected_value; False for random choice
        super_smart_policy=False, # True if you want to memorize in cache a win rate for terminal state (sx_a, dx_e)
        epsilon=0.01,             # policy's exploration probability
        decay_epsilon=0.05,       # decay factor for epsilon (over time I want to reduce the exploration)
        verbose=True              # True if you want to print in console each episode sequence
    )
    # ------------------------------------------------------------------------------------------------------------------
    print(rl_workflow)
    print(visits)
    plot_results(rl_workflow, visits)
