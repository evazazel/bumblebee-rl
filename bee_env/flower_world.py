import numpy as np
import gymnasium as gym
from gymnasium import spaces


class FlowerWorldEnv(gym.Env):
    """
    A 12-flower foraging environment based on:
    'Copy-when-uncertain: bumblebees relyon social information when rewards are highly variable' (Smolla et al., 2016)

    The agent (a virtual bee) visits flowers and collects rewards (sucrose).
    Flowers may or may not have a cue attached (social or non-social).
    Resource distributions are either:
        - high_variance: 2 flowers are rich, 10 are empty
        - no_variance:   all 12 flowers share equal reward
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, variance_condition="high", cue_type="social", cue_reliability=0.5):
        super().__init__()

        self.n_flowers = 12
        self.n_cued_flowers = 4        # matches the paper: 4 out of 12 have a cue
        self.variance_condition = variance_condition
        self.cue_type = cue_type
        self.cue_reliability = cue_reliability  # prob that a cued flower is actually rewarding

        # --- Total sucrose budget (matches paper: 100ul split across flowers) ---
        self.total_reward = 100.0

        # -----------------------------------------------------------------------
        # CUE SALIENCE
        # Encodes the paper's finding that social cues are more perceptually
        # salient than non-social cues, due to evolved attentional mechanisms.
        # Social:     agent perceives the cue 90% of the time
        # Non-social: agent perceives the cue 40% of the time
        # The cue is always physically present — but perception is probabilistic.
        # -----------------------------------------------------------------------
        self.cue_salience = 0.95 if cue_type == "social" else 0.2

        self.action_space = spaces.Discrete(self.n_flowers)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0.0, 0.0]),
            high=np.array([self.n_flowers - 1, 1, 1.0, 1.0]),
            dtype=np.float32
        )

        self.flower_rewards = None
        self.cued_flowers = None
        self.current_flower = None
        self.recent_rewards = []
        self.steps_taken = 0
        self.max_steps = 20

        # -----------------------------------------------------------------------
        # ACTION SPACE
        # In RL, the action space defines every possible action the agent can take.
        # Here: the agent picks which flower to visit next (0 to 11)
        # Discrete(12) means "choose one integer from 0 to 11"
        # -----------------------------------------------------------------------
        self.action_space = spaces.Discrete(self.n_flowers)

        # -----------------------------------------------------------------------
        # OBSERVATION SPACE (STATE)
        # The observation is what the agent *sees* at each step.
        # We give it:
        #   [0]    which flower it's currently at (0-11)
        #   [1]    whether the current flower has a cue (0 or 1)
        #   [2]    reward received at the last step (normalised 0-1)
        #   [3]    running average reward of last 5 visits (normalised 0-1)
        #
        # Box() means continuous values within a range [low, high]
        # -----------------------------------------------------------------------
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0.0, 0.0]),
            high=np.array([self.n_flowers - 1, 1, 1.0, 1.0]),
            dtype=np.float32
        )

        # Internal state (set properly in reset())
        self.flower_rewards = None
        self.cued_flowers = None
        self.current_flower = None
        self.recent_rewards = []
        self.steps_taken = 0
        self.max_steps = 20  # one foraging bout = 20 flower visits

    # -----------------------------------------------------------------------
    # RESET
    # Called at the start of every episode (one foraging bout).
    # Must return the initial observation and an info dict.
    # -----------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Generate rewards AND cue positions together
        # so we can guarantee rich flowers get cues in high-variance
        self.flower_rewards, self.cued_flowers = self._generate_world()

        self.recent_rewards = [0.0] * 5
        self.steps_taken = 0
        self.current_flower = None
        self.recent_rewards = [0.0] * 5

        return self._get_obs(), {}

    # -----------------------------------------------------------------------
    # STEP
    # The core of the environment. The agent passes an action (flower index),
    # and we return:
    #   observation  - what the agent sees next
    #   reward       - how much sucrose it got
    #   terminated   - True if the episode is over (max steps reached)
    #   truncated    - True if cut off early (we won't use this)
    #   info         - any extra debug info you want to track
    # -----------------------------------------------------------------------
    def step(self, action):
        self.current_flower = action
        self.steps_taken += 1

        # --- Calculate reward ---
        reward = self._get_reward(action)

        # Update rolling average (last 5 visits)
        self.recent_rewards.pop(0)
        self.recent_rewards.append(reward / self.total_reward)  # normalise

        terminated = self.steps_taken >= self.max_steps
        obs = self._get_obs()
        info = {
            "flower": action,
            "had_cue": bool(self.cued_flowers[action]),
            "reward": reward,
            "condition": self.variance_condition
        }

        return obs, reward, terminated, False, info

    # -----------------------------------------------------------------------
    # HELPER: Generate flower rewards based on variance condition
    # -----------------------------------------------------------------------
    def _generate_world(self):
        """
    Generate flower rewards AND cue assignments together.

    High variance (matching paper exactly):
        - 2 flowers are rich (50ul each)
        - 4 flowers get cues
        - Both rich flowers are guaranteed to be among the 4 cued flowers
        - The other 2 cued flowers are empty (mimics resource depletion)
        - 8 uncued flowers are all empty

    No variance:
        - All 12 flowers share reward equally (100/12 each)
        - 4 random flowers get cues (no advantage to following them)

    This matches the paper's training setup directly.
        """
        rewards = np.zeros(self.n_flowers)
        cued = np.zeros(self.n_flowers, dtype=int)

        if self.variance_condition == "high":
            # Pick 2 rich flowers
            rich_indices = self.np_random.choice(
                self.n_flowers, 2, replace=False
            )
            rewards[rich_indices] = self.total_reward / 2  # 50ul each

            # Pick 2 MORE flowers (empty) to also get cues
            remaining = np.setdiff1d(np.arange(self.n_flowers), rich_indices)
            empty_cued = self.np_random.choice(remaining, 2, replace=False)

            # All 4 cued flowers = 2 rich + 2 empty (exactly like the paper)
            cued_indices = np.concatenate([rich_indices, empty_cued])
            cued[cued_indices] = 1

        elif self.variance_condition == "no":
            # Equal reward everywhere — cues give no useful information
            rewards[:] = self.total_reward / self.n_flowers  # 8.33ul each
            random_cued = self.np_random.choice(
                self.n_flowers, self.n_cued_flowers, replace=False
            )
            cued[random_cued] = 1

        return rewards, cued

    # -----------------------------------------------------------------------
    # HELPER: Determine actual reward when visiting a flower
    # Cue reliability < 1.0 mimics exploitative competition from the paper:
    # another bee may have already emptied the flower
    # -----------------------------------------------------------------------
    def _get_reward(self, flower_idx):
        base_reward = self.flower_rewards[flower_idx]

        if self.cued_flowers[flower_idx] == 1:
            # Cue present: reward depends on reliability
            if self.np_random.random() < self.cue_reliability:
                return float(base_reward)
            else:
                return 0.0
        else:
            return float(base_reward)

    # -----------------------------------------------------------------------
    # HELPER: Build the observation vector
    # -----------------------------------------------------------------------
    def _get_obs(self):
        # -----------------------------------------------------------------------
        # PERCEIVED CUE
        # Even if a cue is present, the agent only registers it with probability
        # cue_salience. This is the key difference between social/non-social.
        # -----------------------------------------------------------------------
        # On the very first call after reset, no flower has been visited yet
        if self.current_flower is None:
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        cue_physically_present = self.cued_flowers[self.current_flower]

        if cue_physically_present:
            perceived_cue = 1 if self.np_random.random() < self.cue_salience else 0
        else:
            perceived_cue = 0

        return np.array([
            float(self.current_flower),
            float(perceived_cue),        # ← perceived, not raw physical cue
            self.recent_rewards[-1],
            float(np.mean(self.recent_rewards))
        ], dtype=np.float32)
    # -----------------------------------------------------------------------
    # RENDER: Simple text output so we can see what's happening
    # -----------------------------------------------------------------------
    def render(self):
        print(f"Step {self.steps_taken} | Flower: {self.current_flower} | "
              f"Cue: {bool(self.cued_flowers[self.current_flower])} | "
              f"Rewards: {np.round(self.flower_rewards, 1)}")