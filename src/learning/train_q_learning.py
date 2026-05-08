from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.learning.environment import EnergyInvestmentEnv


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs"


def train_q_learning(
    episodes: int = 600,
    alpha: float = 0.12,
    gamma: float = 0.92,
    epsilon: float = 0.25,
    epsilon_decay: float = 0.995,
    seed: int = 42,
) -> Dict[str, object]:
    env = EnergyInvestmentEnv(seed=seed)
    rng = np.random.default_rng(seed)
    q_table: Dict[str, List[float]] = defaultdict(lambda: [0.0] * env.n_actions)
    rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            key = repr(state)
            if rng.random() < epsilon:
                action = int(rng.integers(env.n_actions))
            else:
                action = int(np.argmax(q_table[key]))

            next_state, reward, done, _ = env.step(action)
            next_key = repr(next_state)
            best_next = max(q_table[next_key])
            q_table[key][action] += alpha * (reward + gamma * best_next - q_table[key][action])
            state = next_state
            total_reward += reward

        rewards.append(total_reward)
        epsilon = max(0.03, epsilon * epsilon_decay)

    return {
        "episodes": episodes,
        "alpha": alpha,
        "gamma": gamma,
        "final_epsilon": epsilon,
        "average_reward_last_50": float(np.mean(rewards[-50:])),
        "q_table": dict(q_table),
        "reward_history": rewards,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    result = train_q_learning(episodes=args.episodes, seed=args.seed)
    path = OUTPUT_DIR / "q_table.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved Q-learning model: {path}")
    print(f"Average reward last 50 episodes: {result['average_reward_last_50']:.4f}")


if __name__ == "__main__":
    main()

