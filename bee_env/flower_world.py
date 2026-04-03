import numpy as np
import gymnasium as gym
from gymnasium import spaces


class FlowerWorldEnv(gym.Env):
    """
    Redesigned FlowerWorld environment.

    The agent makes a binary choice each step:
        Action 0 = visit an uncued flower (personal information)
        Action 1 = visit a cued flower    (social/non-social information)

    This directly models the paper's key measure:
    does the agent preferentially choose cued flowers?

    The four conditions:
        high variance + social cue:     cues reliably predict rich rewards
        high variance + non-social cue: cues unreliably perceived
        no variance + social cue:       cues give no reward advantage
        no variance + non-social cue:   both working against cue use
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, variance_condition="high", cue_type="social", cue_reliability=0.9):
        super().__init__()

        self.n_flowers = 12
        self.n_cued_flowers = 4
        self.variance_condition = variance_condition
        self.cue_type = cue_type
        self.cue_reliability = cue_reliability
        self.total_reward = 100.0

        # -------------------------------------------------------------------
        # CUE SALIENCE
        # Social cues are reliably perceived (evolved attentional bias).
        # Non-social cues are often missed.
        # This is the key mechanistic difference from the paper.
        # -------------------------------------------------------------------
        self.cue_salience = 0.95 if cue_type == "social" else 0.2

        # -------------------------------------------------------------------
        # ACTION SPACE: binary
        # 0 = visit a random uncued flower
        # 1 = visit a random cued flower
        # -------------------------------------------------------------------
        self.action_space = spaces.Discrete(2)

        # -------------------------------------------------------------------
        # OBSERVATION SPACE
        # [0] last_reward      (normalised 0-1)
        # [1] avg_reward       (normalised 0-1, rolling last 5)
        # [2] cues_visible     (0 or 1: did agent perceive any cues this step?)
        # -------------------------------------------------------------------
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        self.flower_rewards = None
        self.cued_flowers = None
        self.recent_rewards = []
        self.steps_taken = 0
        self.max_steps = 20
        self.last_visited_flower = None
        self.last_had_cue = False
        self.current_perceived_cues = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.flower_rewards = self._generate_rewards()

        self.cued_flowers = self._assign_cues()

        self.recent_rewards = [0.0] * 5
        self.steps_taken = 0
        self.last_visited_flower = None
        self.last_had_cue = False

        self.current_perceived_cues = self._get_perceived_cued_flowers()

        return self._get_obs(), {}

    def step(self, action):
        self.steps_taken += 1

        # -------------------------------------------------------------------
        # RESOLVE ACTION
        # Action 1 (follow cue): pick from perceived cued flowers
        # Action 0 (ignore cue): pick from uncued flowers
        #
        # If agent tries to follow a cue but perceives none,
        # it falls back to a random flower — cue salience determines
        # how often social/non-social cues are even available to follow
        # -------------------------------------------------------------------
        perceived_cued = self.current_perceived_cues
        uncued = np.where(self.cued_flowers == 0)[0]

        if action == 1 and len(perceived_cued) > 0:
            # Follow a perceived cue
            flower = int(self.np_random.choice(perceived_cued))
            actually_followed_cue = True
        else:
            # Visit a random uncued flower (or fallback if no cues perceived)
            flower = int(self.np_random.choice(
                uncued if len(uncued) > 0 else np.arange(self.n_flowers)
            ))
            actually_followed_cue = False

        self.last_visited_flower = flower
        self.last_had_cue = bool(self.cued_flowers[flower])

        # Get reward
        reward = self._get_reward(flower)

        # Update rolling average
        self.recent_rewards.pop(0)
        self.recent_rewards.append(reward / self.total_reward)

        terminated = self.steps_taken >= self.max_steps

        info = {
            "flower": flower,
            "had_cue": self.last_had_cue,
            "followed_cue": actually_followed_cue,
            "reward": reward,
            "condition": self.variance_condition,
            "cue_type": self.cue_type
        }

        self.current_perceived_cues = self._get_perceived_cued_flowers()

        return self._get_obs(), reward, terminated, False, info

    def _get_perceived_cued_flowers(self):
        """
        Returns indices of cued flowers the agent actually perceives.
        Each cued flower is perceived independently with probability cue_salience.
        This is where social vs non-social diverges.
        """
        perceived = []
        for i in np.where(self.cued_flowers == 1)[0]:
            if self.np_random.random() < self.cue_salience:
                perceived.append(i)
        return perceived

    def _generate_rewards(self):
        rewards = np.zeros(self.n_flowers)
        if self.variance_condition == "high":
            rich = self.np_random.choice(self.n_flowers, 2, replace=False)
            rewards[rich] = self.total_reward / 2
        elif self.variance_condition == "no":
            rewards[:] = self.total_reward / self.n_flowers
        return rewards
    
    def _assign_cues(self):
        """
    Assign cues according to the paper design.

    High variance:
        - both rich flowers are cued
        - two additional empty flowers are also cued
        - total = 4 cued flowers

    No variance:
        - cues placed randomly
        """
        cues = np.zeros(self.n_flowers, dtype=int)

        if self.variance_condition == "high":
            # find rich flowers
            rich = np.where(self.flower_rewards > 0)[0]

            # always cue both rewarding flowers
            cues[rich] = 1

            # choose 2 additional empty flowers
            empty = np.where(self.flower_rewards == 0)[0]
            extra_cues = self.np_random.choice(empty, 2, replace=False)

            cues[extra_cues] = 1

        elif self.variance_condition == "no":
            # keep random placement in no-variance
            cued_indices = self.np_random.choice(
                self.n_flowers,
                self.n_cued_flowers,
                replace=False
            )
            cues[cued_indices] = 1

        return cues

    def _get_reward(self, flower_idx):
        base = self.flower_rewards[flower_idx]
        if self.cued_flowers[flower_idx] == 1:
            return float(base) if self.np_random.random() < self.cue_reliability else 0.0
        return float(base)

    def _get_obs(self):
        cues_visible = 1.0 if len(self.current_perceived_cues) > 0 else 0.0

        return np.array([
            self.recent_rewards[-1],
            float(np.mean(self.recent_rewards)),
            cues_visible
        ], dtype=np.float32)

    def render(self):
        print(f"Step {self.steps_taken} | "
              f"Last flower: {self.last_visited_flower} | "
              f"Had cue: {self.last_had_cue} | "
              f"Rewards: {np.round(self.flower_rewards, 1)}")