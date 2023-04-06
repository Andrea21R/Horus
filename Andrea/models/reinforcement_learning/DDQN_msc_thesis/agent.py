import tensorflow as tf
import numpy as np
import random
from collections import deque
import sys
from time import perf_counter


from typing import *


class DDQNAgent:
    def __init__(
            self,
            state_dim: int,
            num_actions: int,
            learning_rate: float,
            gamma: float,
            epsilon_start: float,
            epsilon_end: float,
            epsilon_decay_steps: int,
            epsilon_exponential_decay: float,
            replay_capacity: int,
            architecture: tuple,  # tuple of int, each int is the number of neurons for layers
            l2_reg,
            tau: int,
            batch_size: int
    ):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.experience = deque([], maxlen=replay_capacity)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.architecture = architecture
        self.l2_reg = l2_reg

        self.online_network = self.build_model()
        self.target_network = self.build_model(trainable=False)
        self.update_target()

        self.epsilon = epsilon_start
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
        self.epsilon_exponential_decay = epsilon_exponential_decay
        self.epsilon_history = []

        self.total_steps = 0
        self.train_steps = 0
        self.episodes = 0
        self.episode_length = 0
        self.train_episodes = 0
        self.steps_per_episode = []
        self.episode_reward = 0
        self.rewards_history = []

        self.batch_size = batch_size
        self.tau = tau
        self.losses = []
        self.idx = tf.range(batch_size)  # tf.range is equal to range() built-in python function
        self.train = True

    def build_model(self, trainable: bool = True):
        """
        Build an ANN model
        - number of layers is defined by self.architecture

        :param trainable: True for the online ANN and False for the target one
        :return: ANN model
        """
        layers = []
        # n = len(self.architecture)
        for i, units in enumerate(self.architecture, 1):
            layers.append(
                tf.keras.layers.Dense(
                    units=units,
                    input_dim=self.state_dim if i == 1 else None,
                    activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg),
                    name=f'Dense_{i}',
                    trainable=trainable
                )
            )
        layers.append(tf.keras.layers.Dropout(.1))
        layers.append(
            tf.keras.layers.Dense(
                units=self.num_actions,
                trainable=trainable,
                name='Output'
            )
        )
        model = tf.keras.models.Sequential(layers)
        model.compile(
            loss='mean_squared_error',
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        )
        return model

    def update_target(self) -> NoReturn:
        self.target_network.set_weights(self.online_network.get_weights())

    def epsilon_greedy_policy(self, state: np.ndarray) -> int:
        self.total_steps += 1
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            q = self.online_network.predict(state, verbose=False)
            return np.argmax(q, axis=1).squeeze()

    def memorize_transition(
            self,
            s: np.ndarray,        # state_features (t)
            a: int,               # action taken (t)
            r: float,             # reward gained (t + 1)
        s_prime: np.ndarray,  # state_features (t + 1)
            not_done: float       # 0 or 1, i.e. like boolean, to understand if episode is finished
    ) -> NoReturn:

        if not_done:
            self.episode_reward += r  # increment episode's reward
            self.episode_length += 1  # increment episode's length
        else:
            if self.train:
                # decrease epsilon (i.e. exploration) until # of episodes is less than # epsilon decay steps
                if self.episodes < self.epsilon_decay_steps:
                    self.epsilon -= self.epsilon_decay
                # otherwise decrease epsilon using an exponential approach
                else:
                    self.epsilon *= self.epsilon_exponential_decay

            self.episodes += 1  # increment # of episodes when the episode is finished
            self.rewards_history.append(self.episode_reward) # store reward history
            # print("reward_history size:", round(sys.getsizeof(self.rewards_history) / 1e6, 3), "MB")
            self.steps_per_episode.append(self.episode_length)  # store episode's length
            # print("steps_per_episode size:", round(sys.getsizeof(self.steps_per_episode) / 1e6, 3), "MB")
            self.episode_reward, self.episode_length = 0, 0  # reset to 0 both episode's reward and episode's length

        self.experience.append((s, a, r, s_prime, not_done))  # store transition data (i.e. one step)

    def experience_replay(self) -> NoReturn:
        """
        It trains ANN (both target and online) using the experience replay approach
        """
        # Make experience replay approach only if the stored experienced has at least self.batch_size elements
        if self.batch_size > len(self.experience):
            pass
        else:
            start_ = perf_counter()
            # ----- It uses a vectorize approach, instead of a for loop ------------------------------------------------

            # create a minibatch, taking N (self.batch_size) experiences (from self.experience). Using random.sample
            # will return a list of list, then zipping them we will have a list of 5 tuple where each tuple will be a
            # "category" (state, action, ...). Then mapping them with function np.array, we will have the input for the model
            minibatch = map(np.array, zip(*random.sample(self.experience, self.batch_size)))
            states, actions, rewards, next_states, not_done = minibatch

            next_q_values = self.online_network.predict_on_batch(next_states)  # Q-value (t + 1) with ONLINE-ANN
            best_actions = tf.argmax(next_q_values, axis=1)  # best action (i.e. greedy-online) for t+1

            next_q_values_target = self.target_network.predict_on_batch(next_states)  # Q-value (t + 1) with TARGET-ANN
            # It's a way to choice between online-ann choice and target-ann choice. i.e. Double-DQN approach
            target_q_values = tf.gather_nd(
                next_q_values_target,
                # tf.stack: stacks a list of rank-R tensors into one rank-(R+1) tensor
                tf.stack(
                    (
                        self.idx,
                        tf.cast(best_actions, tf.int32)  # tf.cast convert a type to another one (here to int32)
                    ),
                    axis=1
                )
            )
            # update q-value (t) using TARGET or ONLINE ANN. This is where we apply the DOUBLE-DQN approach
            targets = rewards + not_done * self.gamma * target_q_values

            q_values = self.online_network.predict_on_batch(states)  # current Q-value (t)
            q_values[self.idx, actions] = targets  # update q-values

            start = perf_counter()
            loss = self.online_network.train_on_batch(x=states, y=q_values)
            # print(f'train_on_batch elapsed_time: {perf_counter() - start}')
            self.losses.append(loss)
            # print("losses size:", round(sys.getsizeof(self.steps_per_episode) / 1e6, 3), "MB")

            # update TARGET-ANN every self.tau steps. Better understand the code
            if self.total_steps % self.tau == 0:
                self.update_target()

            # print(f'-- Total elapsed time for experience replay: {perf_counter() - start_}')
