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
    cue_reliability=0.9,
    seed=42
):
    """
    Train a Q-learning agent in the FlowerWorld environment.

    Args:
        variance_condition: "high" or "no"
        cue_type:           "social" or "non_social"
        n_episodes:         how many foraging bouts to train for
        cue_reliability:    probability that a cued flower actually has reward
        seed:               for reproducibility

    Returns:
        agent:              the trained agent
        results:            dict of recorded metrics
    """

    np.random.seed(seed)

    # Initialise environment and agent
    env = FlowerWorldEnv(
        variance_condition=variance_condition,
        cue_type=cue_type,
        cue_reliability=cue_reliability
    )
    agent = QLearningAgent()

    # -----------------------------------------------------------------------
    # METRICS WE'LL TRACK
    # These let us recreate something like Figure 1 from the paper
    # -----------------------------------------------------------------------
    all_rewards = []           # total reward per episode
    all_cue_rates = []         # proportion of visits to cued flowers per episode
    all_epsilon = []           # epsilon value over time (shows explore→exploit shift)
    smoothed_rewards = []      # rolling average for plotting

    print(f"\nTraining: variance={variance_condition}, cue={cue_type}")
    print(f"Episodes: {n_episodes} | Cue reliability: {cue_reliability}")
    print("-" * 50)

    for episode in range(n_episodes):

        # --- Reset for new episode ---
        obs, _ = env.reset()
        total_reward = 0.0
        cue_visits = 0
        total_visits = 0

        # --- Run one episode (up to max_steps) ---
        terminated = False
        while not terminated:

            # Agent picks a flower to visit
            action = agent.select_action(obs)

            # Environment tells us what happened
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Agent updates its Q-table based on what it experienced
            agent.update(obs, action, reward, next_obs, terminated)

            # Track metrics
            total_reward += reward
            total_visits += 1
            if info["had_cue"]:
                cue_visits += 1

            # Move to next state
            obs = next_obs

        # --- End of episode ---
        agent.decay_epsilon()
        agent.record_episode(total_reward, cue_visits, total_visits)

        all_rewards.append(total_reward)
        all_cue_rates.append(cue_visits / total_visits if total_visits > 0 else 0)
        all_epsilon.append(agent.epsilon)

        # Rolling average over last 100 episodes
        if len(all_rewards) >= 100:
            smoothed_rewards.append(np.mean(all_rewards[-100:]))
        else:
            smoothed_rewards.append(np.mean(all_rewards))

        # Print progress every 500 episodes
        if (episode + 1) % 500 == 0:
            avg_reward = np.mean(all_rewards[-100:])
            avg_cue_rate = np.mean(all_cue_rates[-100:])
            print(f"  Episode {episode+1:>5} | "
                  f"Avg reward: {avg_reward:>6.1f} | "
                  f"Cue follow rate: {avg_cue_rate:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f}")

    print(f"\nTraining complete!")
    print(f"Final cue following rate (last 100 eps): "
          f"{np.mean(all_cue_rates[-100:]):.3f}")

    results = {
        "variance_condition": variance_condition,
        "cue_type": cue_type,
        "all_rewards": all_rewards,
        "all_cue_rates": all_cue_rates,
        "all_epsilon": all_epsilon,
        "smoothed_rewards": smoothed_rewards,
        "final_cue_rate": float(np.mean(all_cue_rates[-100:])),
        "final_avg_reward": float(np.mean(all_rewards[-100:]))
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


# -----------------------------------------------------------------------
# Run training when this file is executed directly
# -----------------------------------------------------------------------
if __name__ == "__main__":
    results = run_all_conditions(n_episodes=5000)

    # Quick summary table
    print("\n" + "=" * 60)
    print("SUMMARY: Final Cue Following Rates")
    print("=" * 60)
    print(f"{'Condition':<40} {'Cue Rate':>10}")
    print("-" * 60)
    for key, res in results.items():
        print(f"{key:<40} {res['final_cue_rate']:>10.3f}")
    print("=" * 60)
    print("\nExpected from paper:")
    print("  high_variance_social_cue → HIGH cue following rate")
    print("  all others               → LOW cue following rate")