import sys
import os
import numpy as np
import json

# Make sure Python can find our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bee_env.flower_world import FlowerWorldEnv
from agents.q_agent import QLearningAgent


def run_training(
    variance_condition,
    cue_type,
    n_episodes=5000,
    cue_reliability=0.8,
    seed=42
):
    """
    Train a Q-learning agent in the FlowerWorld environment.

    Tracks TWO behavioural metrics:
        1. chosen cue rate   -> policy preference
        2. actual cue rate   -> realised cue following
    """

    np.random.seed(seed)

    env = FlowerWorldEnv(
        variance_condition=variance_condition,
        cue_type=cue_type,
        cue_reliability=cue_reliability
    )
    agent = QLearningAgent()

    # Metrics
    all_rewards = []
    all_chosen_cue_rates = []
    all_actual_cue_rates = []
    all_epsilon = []
    smoothed_rewards = []

    print(f"\nTraining: variance={variance_condition}, cue={cue_type}")
    print(f"Episodes: {n_episodes} | Cue reliability: {cue_reliability}")
    print("-" * 70)

    for episode in range(n_episodes):

        obs, _ = env.reset()

        total_reward = 0.0
        total_visits = 0

        # NEW: separate metrics
        chosen_cue_actions = 0
        actual_cue_follows = 0

        terminated = False

        while not terminated:

            # Agent decision
            action = agent.select_action(obs)

            # Track intended policy choice
            if action == 1:
                chosen_cue_actions += 1

            # Environment step
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Track realised cue following
            if info["followed_cue"]:
                actual_cue_follows += 1

            # Q-learning update
            agent.update(obs, action, reward, next_obs, terminated)

            # Update totals
            total_reward += reward
            total_visits += 1

            # Move state forward
            obs = next_obs

        # End episode
        agent.decay_epsilon()

        chosen_rate = (
            chosen_cue_actions / total_visits
            if total_visits > 0 else 0
        )

        actual_rate = (
            actual_cue_follows / total_visits
            if total_visits > 0 else 0
        )

        all_rewards.append(total_reward)
        all_chosen_cue_rates.append(chosen_rate)
        all_actual_cue_rates.append(actual_rate)
        all_epsilon.append(agent.epsilon)

        # Keep compatibility with agent logging
        agent.record_episode(
            total_reward,
            chosen_cue_actions,
            total_visits
        )

        # Smoothed reward
        if len(all_rewards) >= 100:
            smoothed_rewards.append(np.mean(all_rewards[-100:]))
        else:
            smoothed_rewards.append(np.mean(all_rewards))

        # Progress output
        if (episode + 1) % 500 == 0:
            avg_reward = np.mean(all_rewards[-100:])
            avg_chosen = np.mean(all_chosen_cue_rates[-100:])
            avg_actual = np.mean(all_actual_cue_rates[-100:])

            print(
                f"Episode {episode + 1:>5} | "
                f"Avg reward: {avg_reward:>6.1f} | "
                f"Chosen cue rate: {avg_chosen:.2f} | "
                f"Actual cue rate: {avg_actual:.2f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    print("\nTraining complete!")
    print(
        f"Final chosen cue rate (last 100): "
        f"{np.mean(all_chosen_cue_rates[-100:]):.3f}"
    )
    print(
        f"Final actual cue rate (last 100): "
        f"{np.mean(all_actual_cue_rates[-100:]):.3f}"
    )

    results = {
        "variance_condition": variance_condition,
        "cue_type": cue_type,
        "all_rewards": all_rewards,
        "all_chosen_cue_rates": all_chosen_cue_rates,
        "all_actual_cue_rates": all_actual_cue_rates,
        "all_epsilon": all_epsilon,
        "smoothed_rewards": smoothed_rewards,
        "final_chosen_cue_rate": float(
            np.mean(all_chosen_cue_rates[-100:])
        ),
        "final_actual_cue_rate": float(
            np.mean(all_actual_cue_rates[-100:])
        ),
        "final_avg_reward": float(
            np.mean(all_rewards[-100:])
        )
    }

    return agent, results


def run_all_conditions(n_episodes=5000, save=True):
    """
    Run all four conditions from the paper's 2x2 design and save results.
    """

    conditions = [
        ("high", "social"),
        ("high", "non_social"),
        ("no",   "social"),
        ("no",   "non_social"),
    ]

    all_results = {}

    for variance, cue in conditions:
        key = f"{variance}_variance_{cue}_cue"
        agent, results = run_training(
            variance_condition=variance,
            cue_type=cue,
            n_episodes=n_episodes
        )
        all_results[key] = results

    # Save results to file so we can plot without retraining
    if save:
        os.makedirs("results", exist_ok=True)
        with open("results/training_results.json", "w") as f:
            json.dump(all_results, f)
        print("\nResults saved to results/training_results.json")

    return all_results

if __name__ == "__main__":
    results = run_all_conditions(n_episodes=5000)

    print("\n" + "=" * 60)
    print("SUMMARY: Final Cue Rates")
    print("=" * 60)

    for key, res in results.items():
        print(
            f"{key:<40} "
            f"{res['final_chosen_cue_rate']:>10.3f} "
            f"{res['final_actual_cue_rate']:>10.3f}"
        )