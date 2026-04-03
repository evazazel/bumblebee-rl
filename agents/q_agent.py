import numpy as np


class QLearningAgent:
    """
    A Q-learning agent for the FlowerWorld environment.

    Q-learning is a model-free RL algorithm — the agent doesn't know anything
    about the environment in advance. It learns purely from trial and error,
    updating a table of (state, action) values based on received rewards.
    """

    def __init__(
        self,
        n_actions=12,
        learning_rate=0.1,       # α: how fast to update Q-values
        discount_factor=0.95,    # γ: how much to value future rewards
        epsilon=1.0,             # starting exploration rate (100% random)
        epsilon_min=0.05,        # minimum exploration (always explore 5%)
        epsilon_decay=0.995,     # how fast to reduce exploration
    ):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # -----------------------------------------------------------------------
        # STATE DISCRETISATION
        # Our observation has 4 continuous values. We bin each into buckets
        # so we can use a table.
        #
        # [current_flower, cue_present, last_reward, avg_reward]
        #
        # current_flower: already 0-11, so 12 bins
        # cue_present:    already 0 or 1, so 2 bins
        # last_reward:    normalised 0-1, split into 5 bins
        # avg_reward:     normalised 0-1, split into 5 bins
        # -----------------------------------------------------------------------
        self.bins = [
            np.linspace(0, 11, 12),   # current flower (12 values)
            np.array([0, 1]),          # cue present (binary)
            np.linspace(0, 1, 5),     # last reward (5 bins)
            np.linspace(0, 1, 5),     # avg recent reward (5 bins)
        ]

        # Q-table shape: (12, 2, 5, 5, 12)
        # = all combinations of state bins × number of actions
        q_shape = tuple(len(b) for b in self.bins) + (n_actions,)
        self.q_table = np.zeros(q_shape)

        # For tracking learning progress
        self.episode_rewards = []
        self.cue_following_rate = []  # key metric: how often agent follows cues

    # -----------------------------------------------------------------------
    # DISCRETISE STATE
    # Converts a continuous observation into a tuple of bin indices
    # e.g. [5.0, 1.0, 0.3, 0.15] → (5, 1, 1, 0)
    # -----------------------------------------------------------------------
    def discretise(self, obs):
        state = []
        for i, val in enumerate(obs):
            # np.digitize finds which bin a value falls into
            idx = int(np.digitize(val, self.bins[i])) - 1
            # Clamp to valid range (avoids edge case errors)
            idx = max(0, min(idx, len(self.bins[i]) - 1))
            state.append(idx)
        return tuple(state)

    # -----------------------------------------------------------------------
    # SELECT ACTION: Epsilon-Greedy Policy
    #
    # This is the explore vs exploit trade-off — one of the most important
    # concepts in RL:
    #
    #   Explore: try random actions to discover new things
    #   Exploit: use what you've already learned
    #
    # Early in training epsilon=1.0, so the agent explores 100% of the time.
    # Over time epsilon decays, so it exploits its learned Q-values more.
    # -----------------------------------------------------------------------
    def select_action(self, obs, cued_flowers=None):
        state = self.discretise(obs)

        if np.random.random() < self.epsilon:
            # EXPLORE: random action
            return np.random.randint(self.n_actions)
        else:
            # EXPLOIT: pick the action with the highest Q-value
            return int(np.argmax(self.q_table[state]))

    # -----------------------------------------------------------------------
    # UPDATE: The Q-learning Formula
    #
    # After every step, we update the Q-value for (state, action):
    #
    # Q(s,a) ← Q(s,a) + α * [reward + γ * max_a'(Q(s',a')) - Q(s,a)]
    #                         └─────────────────────────────────────┘
    #                                      TD error
    #
    # TD error = how wrong our previous estimate was
    # If reward was better than expected → increase Q(s,a)
    # If reward was worse than expected  → decrease Q(s,a)
    # -----------------------------------------------------------------------
    def update(self, obs, action, reward, next_obs, terminated):
        state = self.discretise(obs)
        next_state = self.discretise(next_obs)

        # Current Q-value for this (state, action) pair
        current_q = self.q_table[state + (action,)]

        if terminated:
            # No future rewards if episode is over
            target = reward
        else:
            # Bellman equation: reward + discounted best future Q-value
            target = reward + self.gamma * np.max(self.q_table[next_state])

        # Update Q-table
        self.q_table[state + (action,)] += self.lr * (target - current_q)

    # -----------------------------------------------------------------------
    # DECAY EPSILON
    # Called at the end of each episode to gradually shift from
    # exploring → exploiting
    # -----------------------------------------------------------------------
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # -----------------------------------------------------------------------
    # RECORD CUE FOLLOWING RATE
    # This is our key behavioural metric — mirrors Figure 1b) in the paper.
    # We track: out of all visits, what proportion went to cued flowers?
    # -----------------------------------------------------------------------
    def record_episode(self, total_reward, cue_visits, total_visits):
        self.episode_rewards.append(total_reward)
        rate = cue_visits / total_visits if total_visits > 0 else 0
        self.cue_following_rate.append(rate)