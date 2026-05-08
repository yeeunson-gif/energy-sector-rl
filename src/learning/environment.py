from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.utils.features import DEFAULT_DATA_PATH, FEATURE_COLUMNS, discretize_row, load_energy_risk_data, minmax_scale


ACTION_TO_POSITION = {
    0: 0.00,  # no investment
    1: 0.33,  # small allocation
    2: 0.67,  # medium allocation
    3: 1.00,  # high allocation
}


@dataclass
class RewardConfig:
    risk_aversion: float = 0.20
    volatility_penalty: float = 0.15
    transaction_cost: float = 0.005
    policy_shock_penalty: float = 0.04


class EnergyInvestmentEnv:
    """A lightweight RL environment for country-energy investment allocation.

    This class intentionally avoids requiring gymnasium so the first GitHub version
    runs with only pandas/numpy. Its API mirrors the familiar reset/step pattern.
    """

    def __init__(
        self,
        data_path: str | Path = DEFAULT_DATA_PATH,
        reward_config: RewardConfig | None = None,
        seed: int = 42,
        episode_length: int = 160,
    ) -> None:
        self.raw_df = load_energy_risk_data(data_path)
        self.df = minmax_scale(self.raw_df)
        self.reward_config = reward_config or RewardConfig()
        self.rng = np.random.default_rng(seed)
        self.episode_length = min(episode_length, len(self.df))
        self.order = np.arange(len(self.df))
        self.step_index = 0
        self.previous_position = 0.0

    @property
    def n_actions(self) -> int:
        return len(ACTION_TO_POSITION)

    def reset(self) -> Tuple[tuple[int, ...], Dict[str, object]]:
        self.order = self.rng.permutation(len(self.df))[: self.episode_length]
        self.step_index = 0
        self.previous_position = 0.0
        return self._state(), self._info()

    def step(self, action: int) -> Tuple[tuple[int, ...], float, bool, Dict[str, object]]:
        action = int(action)
        position = ACTION_TO_POSITION[action]
        raw = self._raw_row()
        reward = self._reward(raw, position)
        self.previous_position = position
        self.step_index += 1
        done = self.step_index >= self.episode_length - 1
        return self._state(), reward, done, self._info(action=action, reward=reward)

    def _state(self) -> tuple[int, ...]:
        return discretize_row(self.df.iloc[self.order[self.step_index]])

    def _raw_row(self) -> pd.Series:
        return self.raw_df.iloc[self.order[self.step_index]]

    def _info(self, action: int | None = None, reward: float | None = None) -> Dict[str, object]:
        row = self._raw_row()
        info = {
            "country": row["country"],
            "region": row["region"],
            "energy": row["energy"],
            "risk_score": float(row["countrySpecificRiskScore"]),
            "expected_annual_return": float(row["expected_annual_return"]),
            "prob8": float(row["prob8"]),
            "mainRiskDriver": row.get("mainRiskDriver", ""),
        }
        if action is not None:
            info["action"] = action
            info["position"] = ACTION_TO_POSITION[action]
        if reward is not None:
            info["reward"] = reward
        return info

    def _reward(self, row: pd.Series, position: float) -> float:
        cfg = self.reward_config
        expected_return = float(row["expected_annual_return"]) / 100.0
        risk = float(row["countrySpecificRiskScore"]) / 100.0
        volatility = float(row["estimated_sigma"])
        policy_risk = float(row["policyScore"]) / 100.0
        turnover = abs(position - self.previous_position)

        return (
            position * expected_return
            - position * cfg.risk_aversion * risk
            - position * cfg.volatility_penalty * volatility
            - position * cfg.policy_shock_penalty * policy_risk
            - cfg.transaction_cost * turnover
        )

