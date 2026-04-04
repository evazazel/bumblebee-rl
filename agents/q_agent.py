import numpy as np


class QLearningAgent:
    """
    Q-learning agent for the redesigned FlowerWorld.

    State:  [last_reward, avg_reward, cues_visible]
    Action: 0 (visit uncued flower) or 1 (follow a cue)

    The Q-table now has a clear semantic meaning:
        Q[state][0] = expected reward for ignoring cues
        Q[state][1] = expected reward for following a cue
    """

    def __init__(
        self,
        n_actions=2,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.999,    # slower decay → more exploration
    ):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Bins for each state dimension
        # [last_reward, avg_reward, cues_visible]
        self.bins = [
            np.linspace(0, 1, 5),   # last reward: 5 bins
            np.linspace(0, 1, 5),   # avg reward: 5 bins
            np.array([0, 1]),         # cues visible: binary
        ]

        # Q-table: (10, 10, 2, 2) — state bins × n_actions
        q_shape = tuple(len(b) for b in self.bins) + (n_actions,)
        self.q_table = np.zeros(q_shape)

        self.episode_rewards = []
        self.cue_following_rate = []

    def discretise(self, obs):
        state = []
        for i, val in enumerate(obs):
            idx = int(np.digitize(val, self.bins[i])) - 1
            idx = max(0, min(idx, len(self.bins[i]) - 1))
            state.append(idx)
        return tuple(state)

    def select_action(self, obs):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        state = self.discretise(obs)
        return int(np.argmax(self.q_table[state]))

    def update(self, obs, action, reward, next_obs, terminated):
        state = self.discretise(obs)
        next_state = self.discretise(next_obs)
        current_q = self.q_table[state + (action,)]
        if terminated:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state + (action,)] += self.lr * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def record_episode(self, total_reward, cue_visits, total_visits):
        self.episode_rewards.append(total_reward)
        rate = cue_visits / total_visits if total_visits > 0 else 0
        self.cue_following_rate.append(rate)