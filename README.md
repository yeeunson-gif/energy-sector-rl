# Energy Sector Investment Analysis with Reinforcement Learning

This project extends the Energy Sector Investment Analysis dashboard into a learning-based investment simulation framework.

The original dashboard estimates country-energy investment risk using market, policy/news, and macroeconomic indicators. This repository adds a reinforcement learning layer that learns how aggressively to allocate capital to each country-energy opportunity under risk and return constraints.

## Project Structure

```text
energy-sector-rl/
├─ dashboard/
│  ├─ index.html
│  └─ real_dashboard_data.js
├─ data/
│  └─ real_energy_risk_results.csv
├─ src/
│  ├─ data_pipeline/
│  │  └─ build_real_dashboard_data.py
│  ├─ learning/
│  │  ├─ environment.py
│  │  ├─ train_q_learning.py
│  │  ├─ evaluate_agent.py
│  │  └─ export_rl_results.py
│  └─ utils/
│     └─ features.py
├─ docs/
│  └─ methodology.md
├─ outputs/
├─ requirements.txt
└─ README.md
```

## Learning Problem

### State

Each state represents one country-energy investment condition:

- Market Risk
- Policy Risk
- Macro Risk
- Country-specific Risk Score
- Country-specific Market/Policy/Macro weights
- CAPM beta
- ETF volatility
- GDP growth
- Inflation
- Debt/GDP
- FX volatility
- Political stability
- Estimated annual return
- Estimated risk-adjusted probability

### Action

The agent selects an allocation intensity:

```text
0 = No Investment
1 = Small Allocation
2 = Medium Allocation
3 = High Allocation
```

### Reward

The reward balances expected return against risk:

```text
reward =
  position × expected_return
  - position × risk_aversion × country_specific_risk
  - position × volatility_penalty × estimated_sigma
  - position × policy_shock_penalty × policy_risk
  - transaction_cost × turnover
```

This is a research simulation reward, not real investment advice.

## Quick Start

From the repository root:

```powershell
pip install -r requirements.txt
python -m src.learning.train_q_learning --episodes 600
python -m src.learning.evaluate_agent
python -m src.learning.export_rl_results
```

Outputs:

```text
outputs/q_table.json
outputs/evaluation.json
outputs/rl_recommendations.csv
outputs/rl_recommendations.js
dashboard/rl_recommendations.js
```

Open the dashboard:

```text
dashboard/index.html
```

The dashboard displays both the country-specific risk model and the RL recommended allocation for the selected country-energy pair.

## Why Reinforcement Learning?

The dashboard does not simply rank risk indicators. It can be reframed as a sequential decision problem:

```text
Observe country-energy conditions
→ choose allocation level
→ receive risk-adjusted reward
→ update policy
```

This structure is useful because investment decisions are not only predictions. They are actions under uncertainty.

## Roadmap

- Phase 1: Deterministic country-specific risk weights
- Phase 2: RL simulation environment
- Phase 3: Q-learning baseline agent
- Phase 4: DQN/PPO with Gymnasium and Stable-Baselines3
- Phase 5: Historical realized return rewards
- Phase 6: Dashboard integration of RL recommended actions

## Disclaimer

This project is for academic and portfolio demonstration purposes. It is not financial advice and should not be used for real investment decisions.
