from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict

import numpy as np

from src.learning.environment import ACTION_TO_POSITION, EnergyInvestmentEnv


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs"


def load_q_policy(path: Path) -> Callable[[tuple[int, ...], Dict[str, object]], int]:
    q_table = json.loads(path.read_text(encoding="utf-8"))["q_table"]

    def policy(state: tuple[int, ...], _: Dict[str, object]) -> int:
        values = q_table.get(repr(state), [0.0, 0.0, 0.0, 0.0])
        return int(np.argmax(values))

    return policy


def conservative_policy(_: tuple[int, ...], info: Dict[str, object]) -> int:
    risk = float(info["risk_score"])
    prob8 = float(info["prob8"])
    if risk < 45 and prob8 >= 55:
        return 3
    if risk < 60 and prob8 >= 45:
        return 2
    if risk < 72:
        return 1
    return 0


def risk_neutral_policy(_: tuple[int, ...], info: Dict[str, object]) -> int:
    prob8 = float(info["prob8"])
    if prob8 >= 65:
        return 3
    if prob8 >= 52:
        return 2
    if prob8 >= 40:
        return 1
    return 0


def random_policy_factory(seed: int = 7) -> Callable[[tuple[int, ...], Dict[str, object]], int]:
    rng = np.random.default_rng(seed)
    return lambda _state, _info: int(rng.integers(4))


def evaluate(policy: Callable[[tuple[int, ...], Dict[str, object]], int], episodes: int = 100, seed: int = 99) -> Dict[str, float]:
    rewards = []
    positions = []
    env = EnergyInvestmentEnv(seed=seed)
    for _ in range(episodes):
        state, info = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = policy(state, info)
            positions.append(ACTION_TO_POSITION[action])
            state, reward, done, info = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    return {
        "episodes": episodes,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_position": float(np.mean(positions)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=OUTPUT_DIR / "q_table.json")
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

    policies = {
        "random": random_policy_factory(),
        "conservative_rule": conservative_policy,
        "risk_neutral_rule": risk_neutral_policy,
    }
    if args.model.exists():
        policies["q_learning"] = load_q_policy(args.model)

    results = {name: evaluate(policy, episodes=args.episodes) for name, policy in policies.items()}
    OUTPUT_DIR.mkdir(exist_ok=True)
    out = OUTPUT_DIR / "evaluation.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved evaluation: {out}")


if __name__ == "__main__":
    main()

