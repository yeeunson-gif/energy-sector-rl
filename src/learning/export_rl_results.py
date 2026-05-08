from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.learning.environment import ACTION_TO_POSITION
from src.utils.features import DEFAULT_DATA_PATH, discretize_row, load_energy_risk_data, minmax_scale


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs"

ACTION_LABELS = {
    0: "No Investment",
    1: "Small Allocation",
    2: "Medium Allocation",
    3: "High Allocation",
}


def main() -> None:
    model_path = OUTPUT_DIR / "q_table.json"
    if not model_path.exists():
        raise SystemExit("Train first: python -m src.learning.train_q_learning")

    q_table = json.loads(model_path.read_text(encoding="utf-8"))["q_table"]
    raw = load_energy_risk_data(DEFAULT_DATA_PATH)
    scaled = minmax_scale(raw)
    actions = []

    for idx, row in scaled.iterrows():
        state = discretize_row(row)
        q_values = q_table.get(repr(state), [0.0, 0.0, 0.0, 0.0])
        action = int(np.argmax(q_values))
        actions.append(
            {
                "rlAction": action,
                "rlActionLabel": ACTION_LABELS[action],
                "rlPosition": ACTION_TO_POSITION[action],
                "rlQValues": q_values,
            }
        )

    result = pd.concat([raw.reset_index(drop=True), pd.DataFrame(actions)], axis=1)
    OUTPUT_DIR.mkdir(exist_ok=True)
    csv_path = OUTPUT_DIR / "rl_recommendations.csv"
    js_path = OUTPUT_DIR / "rl_recommendations.js"
    dashboard_js_path = ROOT / "dashboard" / "rl_recommendations.js"
    result.to_csv(csv_path, index=False, encoding="utf-8-sig")
    js_payload = "window.RL_RECOMMENDATIONS = " + result.to_json(orient="records", force_ascii=False) + ";\n"
    js_path.write_text(js_payload, encoding="utf-8")
    dashboard_js_path.write_text(js_payload, encoding="utf-8")
    print(f"Saved RL recommendations: {csv_path}")
    print(f"Saved dashboard JS: {js_path}")
    print(f"Updated dashboard JS: {dashboard_js_path}")


if __name__ == "__main__":
    main()
