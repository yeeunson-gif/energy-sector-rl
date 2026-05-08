# Methodology

## 1. From Risk Indicator to Learning System

The first version of the project produced an energy investment risk dashboard. It calculated Market Risk, Policy Risk, and Macro Risk, then combined them into a country-specific risk score.

The reinforcement learning extension treats the dashboard as an investment decision environment. Instead of stopping at a risk score, the model learns an action: how much capital to allocate to a country-energy opportunity.

## 2. State Definition

The state vector combines market, policy, macroeconomic, and country-specific variables:

- Market Risk
- Policy Risk
- Macro Risk
- Country-specific Risk Score
- Country-specific Market/Policy/Macro weights
- CAPM beta
- Annualized ETF volatility
- GDP growth
- Inflation
- Debt/GDP
- FX volatility
- Political stability
- Estimated annual return
- Estimated sigma
- Probability of exceeding an 8% annual return proxy

These features are normalized and discretized for tabular Q-learning.

## 3. Action Space

The action space is intentionally simple in the first version:

| Action | Meaning |
|---:|---|
| 0 | No Investment |
| 1 | Small Allocation |
| 2 | Medium Allocation |
| 3 | High Allocation |

This keeps the environment interpretable and easy to evaluate before moving to continuous portfolio weights.

## 4. Reward Function

The reward function is:

```text
reward =
  position × expected_return
  - position × risk_aversion × country_specific_risk
  - position × volatility_penalty × estimated_sigma
  - position × policy_shock_penalty × policy_risk
  - transaction_cost × turnover
```

This makes the agent prefer opportunities with attractive expected returns, while penalizing high country-specific risk, volatility, policy uncertainty, and excessive position changes.

## 5. Baselines

The RL agent is evaluated against:

- Random policy
- Conservative rule-based policy
- Risk-neutral rule-based policy
- Q-learning policy

The goal is not to claim real trading profitability. The goal is to show that the dashboard can evolve from a static indicator system into a decision-learning framework.

## 6. Future Extensions

The current version uses proxy expected returns and risk penalties. A stronger version can use:

- Historical ETF realized returns as rewards
- Country-level energy investment flows
- Project finance default or delay events
- GDELT policy shock events
- Continuous action portfolio allocation
- PPO, DQN, or actor-critic algorithms

