from __future__ import annotations

import csv
import json
import math
import os
import statistics
import time
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from io import BytesIO, TextIOWrapper
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from energy_investment_analysis import COUNTRIES, ENERGY_TYPES, POLICY_KEYWORDS, risk_label


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT
USER_AGENT = "Mozilla/5.0 energy-sector-investment-analysis/1.0"

ISO3 = {
    "United States": "USA", "Canada": "CAN", "Mexico": "MEX", "Brazil": "BRA", "Chile": "CHL",
    "Argentina": "ARG", "United Kingdom": "GBR", "Germany": "DEU", "France": "FRA", "Italy": "ITA",
    "Spain": "ESP", "Netherlands": "NLD", "Norway": "NOR", "Sweden": "SWE", "Poland": "POL",
    "Turkiye": "TUR", "Russia": "RUS", "China": "CHN", "Japan": "JPN", "South Korea": "KOR",
    "India": "IND", "Indonesia": "IDN", "Vietnam": "VNM", "Thailand": "THA", "Philippines": "PHL",
    "Malaysia": "MYS", "Singapore": "SGP", "Australia": "AUS", "New Zealand": "NZL",
    "Saudi Arabia": "SAU", "United Arab Emirates": "ARE", "Qatar": "QAT", "Israel": "ISR",
    "South Africa": "ZAF", "Egypt": "EGY", "Nigeria": "NGA", "Kenya": "KEN", "Morocco": "MAR",
    "Ghana": "GHA", "Colombia": "COL",
}

WB_INDICATORS = {
    "gdp_growth": "NY.GDP.MKTP.KD.ZG",
    "inflation": "FP.CPI.TOTL.ZG",
    "debt_to_gdp": "GC.DOD.TOTL.GD.ZS",
    "exchange_rate": "PA.NUS.FCRF",
}

ETF_FALLBACKS = {
    "KOL": "XLE",
    "HDRO": "ICLN",
    "GRID": "ICLN",
    "KRBN": "ICLN",
    "PBD": "ICLN",
}

ENERGY_NEWS_TERMS = {
    "Solar PV": "solar energy",
    "Onshore Wind": "onshore wind",
    "Offshore Wind": "offshore wind",
    "Hydropower": "hydropower",
    "Geothermal": "geothermal energy",
    "Biomass": "biomass energy",
    "Nuclear": "nuclear energy",
    "Natural Gas": "natural gas",
    "LNG": "liquefied natural gas",
    "Oil": "oil energy",
    "Coal": "coal power",
    "Green Hydrogen": "green hydrogen",
    "Battery Storage": "battery storage",
    "Transmission Grid": "electric grid",
    "Carbon Capture": "carbon capture",
    "Energy Efficiency": "energy efficiency",
}


def fetch_json(url: str, timeout: int = 20) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def latest_world_bank_value(iso3: str, indicator: str) -> Tuple[Optional[float], Optional[int]]:
    url = f"https://api.worldbank.org/v2/country/{iso3}/indicator/{indicator}?format=json&per_page=20"
    data = fetch_json(url)
    if not isinstance(data, list) or len(data) < 2:
        return None, None
    for item in data[1]:
        value = item.get("value")
        if value is not None:
            return float(value), int(item["date"])
    return None, None


def world_bank_indicator_batch(indicator: str, per_page: int = 1200) -> Dict[str, List[Tuple[int, float]]]:
    countries = ";".join(ISO3.values())
    rows: Dict[str, List[Tuple[int, float]]] = {}
    page = 1
    pages = 1
    while page <= pages:
        url = (
            f"https://api.worldbank.org/v2/country/{countries}/indicator/{indicator}"
            f"?format=json&per_page={per_page}&page={page}"
        )
        data = fetch_json(url)
        if not isinstance(data, list) or len(data) < 2:
            break
        pages = int(data[0].get("pages", 1))
        for item in data[1]:
            value = item.get("value")
            iso3 = item.get("countryiso3code")
            if value is not None and iso3:
                rows.setdefault(iso3, []).append((int(item["date"]), float(value)))
        page += 1
    for values in rows.values():
        values.sort()
    return rows


def latest_from_batch(batch: Dict[str, List[Tuple[int, float]]], iso3: str) -> Tuple[Optional[float], Optional[int]]:
    values = batch.get(iso3, [])
    if not values:
        return None, None
    year, value = values[-1]
    return value, year


def exchange_rate_volatility_from_batch(batch: Dict[str, List[Tuple[int, float]]], iso3: str) -> Tuple[Optional[float], Optional[str]]:
    values = batch.get(iso3, [])
    values.sort()
    changes = []
    for (_, previous), (_, current) in zip(values, values[1:]):
        if previous:
            changes.append((current - previous) / previous * 100)
    if len(changes) < 2:
        return None, None
    return statistics.stdev(changes[-5:]), f"{values[max(0, len(values)-6)][0]}-{values[-1][0]}"


def political_stability_score(iso3: str) -> Tuple[Optional[float], Optional[int]]:
    value, year = latest_world_bank_value(iso3, WB_INDICATORS["political_stability"])
    if value is None:
        return None, None
    return max(0.0, min(100.0, (value + 2.5) / 5.0 * 100.0)), year


def wgi_political_stability() -> Dict[str, Tuple[float, int]]:
    url = "https://databank.worldbank.org/data/download/WGI_CSV.zip"
    data = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=45).read()
    archive = zipfile.ZipFile(BytesIO(data))
    result: Dict[str, Tuple[float, int]] = {}
    with archive.open("WGICSV.csv") as raw:
        reader = csv.DictReader(TextIOWrapper(raw, encoding="utf-8-sig"))
        for row in reader:
            if row.get("Indicator Code") != "GOV_WGI_PV.EST":
                continue
            iso3 = row.get("Country Code", "")
            year_values = []
            for key, value in row.items():
                if key.isdigit() and value:
                    year_values.append((int(key), float(value)))
            if year_values:
                year, raw_value = sorted(year_values)[-1]
                result[iso3] = (max(0.0, min(100.0, (raw_value + 2.5) / 5.0 * 100.0)), year)
    return result


def world_bank_macro() -> Tuple[Dict[str, Dict[str, object]], List[str]]:
    rows: Dict[str, Dict[str, object]] = {}
    warnings: List[str] = []
    batches: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}
    for key, indicator in WB_INDICATORS.items():
        try:
            batches[key] = world_bank_indicator_batch(indicator)
        except Exception as exc:
            batches[key] = {}
            warnings.append(f"World Bank batch {key}: {exc}")
    try:
        stability_batch = wgi_political_stability()
    except Exception as exc:
        stability_batch = {}
        warnings.append(f"WGI political stability: {exc}")
    for country in COUNTRIES:
        iso3 = ISO3[country.country]
        row: Dict[str, object] = {
            "country": country.country,
            "iso3": iso3,
            "region": country.region,
            "development": country.development,
            "source": "World Bank API",
        }
        for key, indicator in WB_INDICATORS.items():
            if key in {"exchange_rate", "political_stability"}:
                continue
            value, year = latest_from_batch(batches.get(key, {}), iso3)
            fallback = getattr(country, key)
            row[key] = round(value if value is not None else fallback, 4)
            row[f"{key}_year"] = year or "fallback"
        fx_vol, fx_period = exchange_rate_volatility_from_batch(batches.get("exchange_rate", {}), iso3)
        row["fx_volatility"] = round(fx_vol if fx_vol is not None else country.fx_volatility, 4)
        row["fx_volatility_period"] = fx_period or "fallback"
        stability, stability_year = stability_batch.get(iso3, (None, None))
        row["policy_stability"] = round(stability if stability is not None else country.policy_stability, 4)
        row["policy_stability_year"] = stability_year or "fallback"
        rows[country.country] = row
    return rows, warnings


def yahoo_prices(symbol: str, range_value: str = "2y") -> List[Tuple[int, float]]:
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol)}"
        f"?range={range_value}&interval=1mo&events=history"
    )
    data = fetch_json(url)
    result = data["chart"]["result"][0]
    timestamps = result.get("timestamp") or []
    closes = result["indicators"]["quote"][0]["close"]
    return [(int(ts), float(close)) for ts, close in zip(timestamps, closes) if close is not None]


def monthly_returns(prices: List[Tuple[int, float]]) -> List[float]:
    returns = []
    for (_, previous), (_, current) in zip(prices, prices[1:]):
        if previous:
            returns.append((current - previous) / previous)
    return returns


def alpha_beta(asset_returns: List[float], market_returns: List[float]) -> Tuple[float, float, float]:
    size = min(len(asset_returns), len(market_returns))
    asset_returns = asset_returns[-size:]
    market_returns = market_returns[-size:]
    if size < 6:
        raise ValueError("not enough return observations")
    market_mean = statistics.mean(market_returns)
    asset_mean = statistics.mean(asset_returns)
    covariance = sum((m - market_mean) * (a - asset_mean) for m, a in zip(market_returns, asset_returns))
    variance = sum((m - market_mean) ** 2 for m in market_returns)
    beta = covariance / variance if variance else 0.0
    alpha = asset_mean - beta * market_mean
    volatility = statistics.stdev(asset_returns) * math.sqrt(12)
    return alpha, beta, volatility


def market_data() -> Tuple[Dict[str, Dict[str, object]], List[str]]:
    warnings: List[str] = []
    market_returns = monthly_returns(yahoo_prices("SPY", "2y"))
    symbols = sorted({e.etf_proxy for e in ENERGY_TYPES} | set(ETF_FALLBACKS.values()))
    rows: Dict[str, Dict[str, object]] = {}
    for symbol in symbols:
        requested = symbol
        used = ETF_FALLBACKS.get(symbol, symbol)
        try:
            asset_returns = monthly_returns(yahoo_prices(used, "2y"))
            alpha, beta, volatility = alpha_beta(asset_returns, market_returns)
            rows[requested] = {
                "requested_symbol": requested,
                "used_symbol": used,
                "alpha": round(alpha, 5),
                "beta": round(beta, 3),
                "annualized_volatility": round(volatility, 4),
                "source": "Yahoo Finance chart API",
            }
        except Exception as exc:
            warnings.append(f"Yahoo Finance {requested}: {exc}")
        time.sleep(0.12)
    return rows, warnings


def gdelt_policy(country: str) -> Tuple[float, str, int, str]:
    query = f'"{country}" energy (regulation OR tax OR ban OR subsidy OR tariff OR permit OR government OR carbon)'
    params = urllib.parse.urlencode({
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": 75,
        "timespan": "180d",
        "sort": "datedesc",
    })
    url = f"https://api.gdeltproject.org/api/v2/doc/doc?{params}"
    try:
        data = fetch_json(url, timeout=25)
        articles = data.get("articles") or []
    except Exception:
        articles = []
    text = " ".join(
        f"{article.get('title', '')} {article.get('seendate', '')}".lower()
        for article in articles
    )
    matched = []
    raw = 0.0
    for keyword, weight in POLICY_KEYWORDS.items():
        if keyword in text:
            raw += weight
            matched.append(keyword)
    score = max(0.0, min(100.0, 34 + raw * 8 + min(len(articles), 75) * 0.35))
    if not matched and articles:
        matched.append("energy news")
    return round(score, 2), ", ".join(matched[:8]), len(articles), "GDELT 2.1 DOC API"


def macro_score(row: Dict[str, object]) -> float:
    gdp = float(row["gdp_growth"])
    inflation = float(row["inflation"])
    debt = float(row["debt_to_gdp"])
    fx = float(row["fx_volatility"])
    stability = float(row["policy_stability"])
    growth_component = max(-16.0, min(28.0, (4.5 - gdp) * 7))
    score = (
        18
        + growth_component
        + max(0.0, min(42.0, inflation * 1.6))
        + max(0.0, min(38.0, debt * 0.18))
        + max(0.0, min(36.0, fx * 1.35))
        - stability * 0.12
    )
    return round(max(0.0, min(100.0, score)), 2)


def market_score(market: Dict[str, object], energy) -> float:
    alpha = float(market["alpha"])
    beta = float(market["beta"])
    volatility = float(market["annualized_volatility"])
    score = 22 + beta * 23 + volatility * 165 + energy.commodity_exposure * 22 - alpha * 650
    return round(max(0.0, min(100.0, score)), 2)


def calculate_country_specific_weights(row: Dict[str, object]) -> Dict[str, float]:
    beta = abs(float(row["beta"]))
    volatility = float(row["annualized_volatility"])
    policy_risk = float(row["policyScore"])
    macro_risk = float(row["macroScore"])
    governance_score = float(row["policy_stability"])
    inflation = float(row["inflation"])
    debt_to_gdp = float(row["debt"])
    fx_risk = float(row["fx"])
    gdp_growth = float(row["gdp"])

    raw_market = 1 + beta * 0.30 + volatility * 0.40 + max(0.0, beta - 1) * 0.20
    raw_policy = 1 + policy_risk * 0.02 + (100 - governance_score) * 0.02
    raw_macro = 1 + inflation * 0.05 + debt_to_gdp * 0.01 + fx_risk * 0.03 + macro_risk * 0.01 - gdp_growth * 0.03

    raw_market = max(0.25, raw_market)
    raw_policy = max(0.25, raw_policy)
    raw_macro = max(0.25, raw_macro)
    total = raw_market + raw_policy + raw_macro

    weights = [raw_market / total, raw_policy / total, raw_macro / total]
    weights = [max(0.15, min(0.60, weight)) for weight in weights]
    clamped_total = sum(weights)
    weights = [weight / clamped_total for weight in weights]
    return {
        "marketWeight": round(weights[0], 4),
        "policyWeight": round(weights[1], 4),
        "macroWeight": round(weights[2], 4),
    }


def calculate_country_specific_risk(row: Dict[str, object]) -> Dict[str, object]:
    weights = calculate_country_specific_weights(row)
    score = (
        weights["marketWeight"] * float(row["marketScore"])
        + weights["policyWeight"] * float(row["policyScore"])
        + weights["macroWeight"] * float(row["macroScore"])
    )
    drivers = [
        ("Market Risk", weights["marketWeight"] * float(row["marketScore"])),
        ("Policy Risk", weights["policyWeight"] * float(row["policyScore"])),
        ("Macro Risk", weights["macroWeight"] * float(row["macroScore"])),
    ]
    return {
        **weights,
        "countrySpecificRiskScore": round(score, 2),
        "mainRiskDriver": sorted(drivers, key=lambda item: item[1], reverse=True)[0][0],
    }


def adjusted_policy_score(base_score: float, energy) -> float:
    score = base_score + energy.policy_exposure * 18 - energy.esg_score * 8
    return round(max(0.0, min(100.0, score)), 2)


def normal_cdf(value: float) -> float:
    return 0.5 * (1 + math.erf(value / math.sqrt(2)))


def estimate_probability_metrics(row: Dict[str, object], energy) -> Dict[str, object]:
    gdp = float(row["gdp"])
    inflation = float(row["inflation"])
    debt = float(row["debt"])
    fx = float(row["fx"])
    stability = float(row["policy_stability"])
    total = float(row["totalScore"])
    alpha = float(row["alpha"])
    volatility = float(row["annualized_volatility"])

    growth_coeff = max(0.10, min(0.70, 0.35 + gdp / 20 - inflation / 80))
    stability_coeff = max(0.05, min(0.35, stability / 300))
    risk_coeff = max(0.02, min(0.12, total / 900 + fx / 1200 + debt / 4000))
    volatility_coeff = max(0.08, min(0.55, volatility + fx / 300 + max(0, inflation - 4) / 150))

    expected_return = (
        alpha * 12
        + (gdp / 100) * growth_coeff
        + energy.esg_score * 0.025
        + stability_coeff * 0.04
        - (inflation / 100) * 0.18
        - (debt / 100) * 0.025
        - (total / 100) * risk_coeff
    )
    sigma = max(0.08, volatility_coeff + total / 350 + fx / 600)

    def probability_above(target: float) -> float:
        return round((1 - normal_cdf((target - expected_return) / sigma)) * 100, 1)

    return {
        "expected_annual_return": round(expected_return * 100, 2),
        "estimated_sigma": round(sigma, 4),
        "probPositive": probability_above(0),
        "prob5": probability_above(0.05),
        "prob8": probability_above(0.08),
        "prob10": probability_above(0.10),
        "downsideProbability": round(100 - probability_above(0), 1),
        "growth_coefficient": round(growth_coeff, 4),
        "stability_coefficient": round(stability_coeff, 4),
        "risk_coefficient": round(risk_coeff, 4),
        "volatility_coefficient": round(volatility_coeff, 4),
        "probability_model_source": "Country-calibrated proxy coefficients estimated from World Bank/WGI macro data and Yahoo Finance ETF volatility",
    }


def fallback_explain(row: Dict[str, object]) -> str:
    strongest = max(
        [("시장", row["marketScore"]), ("정책/뉴스", row["policyScore"]), ("거시경제", row["macroScore"])],
        key=lambda item: float(item[1]),
    )


def call_llm_interpretation(row: Dict[str, object], api_key: str) -> str:
    prompt = (
        "Write a concise Korean investment-risk interpretation for an academic dashboard. "
        "Do not give investment advice. Explain the main risk driver and mention that data comes from "
        "Yahoo Finance, World Bank/WGI, and GDELT.\n\n"
        f"Country: {row['country']}\n"
        f"Energy: {row['energy']}\n"
        f"Market risk: {row['marketScore']}, Policy risk: {row['policyScore']}, Macro risk: {row['macroScore']}\n"
        f"Country-specific risk score: {row.get('countrySpecificRiskScore', row['totalScore'])}\n"
        f"Beta: {row['beta']}, Volatility: {row['annualized_volatility']}\n"
        f"GDP growth: {row['gdp']}%, Inflation: {row['inflation']}%, Debt/GDP: {row['debt']}%, FX risk: {row['fx']}%\n"
        f"Policy keywords: {row['keywords']}\n"
    )
    body = {
        "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        "messages": [
            {"role": "system", "content": "You are an academic financial risk analysis assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 260,
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": USER_AGENT,
        },
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"].strip()
    return (
        f"{row['country']}의 {row['energy']} 투자는 실제 API 데이터 기준 {row['riskLabel']} 수준으로 평가된다. "
        f"가장 큰 요인은 {strongest[0]} 리스크이며, ETF proxy {row['etf']}의 beta는 {row['beta']}이다. "
        f"World Bank 거시지표와 GDELT 정책/뉴스 키워드를 함께 반영했으므로, 투자 판단 시 최신 정책 변화와 "
        f"거시경제 지표의 발표 시점을 추가 확인하는 것이 필요하다."
    )


def write_csv(rows: Iterable[Dict[str, object]], path: Path) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_dashboard_js(rows: List[Dict[str, object]], metadata: Dict[str, object], path: Path) -> None:
    payload = json.dumps(rows, ensure_ascii=False)
    meta = json.dumps(metadata, ensure_ascii=False)
    path.write_text(
        f"window.REAL_RESULTS = {payload};\nwindow.REAL_DATA_METADATA = {meta};\n",
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    warnings: List[str] = []
    llm_api_key = os.environ.get("OPENAI_API_KEY", "")
    macro_rows, macro_warnings = world_bank_macro()
    market_rows, market_warnings = market_data()
    warnings.extend(macro_warnings)
    warnings.extend(market_warnings)
    policy_cache: Dict[str, Tuple[float, str, int, str]] = {}
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(gdelt_policy, country.country): country.country for country in COUNTRIES}
        for future in as_completed(futures):
            country_name = futures[future]
            try:
                policy_cache[country_name] = future.result()
            except Exception as exc:
                warnings.append(f"GDELT {country_name}: {exc}")
                policy_cache[country_name] = (45.0, "news unavailable", 0, "GDELT 2.1 DOC API")

    rows: List[Dict[str, object]] = []
    for country in COUNTRIES:
        macro = macro_rows[country.country]
        macro_risk = macro_score(macro)
        for energy in ENERGY_TYPES:
            market = market_rows.get(energy.etf_proxy) or market_rows.get(ETF_FALLBACKS.get(energy.etf_proxy, energy.etf_proxy))
            if market is None:
                warnings.append(f"missing market data for {energy.name}; skipped")
                continue
            base_policy_score, keywords, article_count, policy_source = policy_cache[country.country]
            p_score = adjusted_policy_score(base_policy_score, energy)
            m_score = market_score(market, energy)
            total = round(m_score * 0.40 + p_score * 0.30 + macro_risk * 0.30, 2)
            row = {
                "country": country.country,
                "region": country.region,
                "development": country.development,
                "energy": energy.name,
                "energy_type": energy.name,
                "group": energy.group,
                "energy_group": energy.group,
                "etf": market["used_symbol"],
                "etf_proxy": market["used_symbol"],
                "marketScore": m_score,
                "market_score": m_score,
                "policyScore": p_score,
                "policy_score": p_score,
                "macroScore": macro_risk,
                "macro_score": macro_risk,
                "totalScore": total,
                "total_score": total,
                "riskLabel": risk_label(total),
                "risk_label": risk_label(total),
                "alpha": market["alpha"],
                "beta": market["beta"],
                "annualized_volatility": market["annualized_volatility"],
                "gdp": macro["gdp_growth"],
                "gdp_growth": macro["gdp_growth"],
                "inflation": macro["inflation"],
                "debt": macro["debt_to_gdp"],
                "debt_to_gdp": macro["debt_to_gdp"],
                "fx": macro["fx_volatility"],
                "fx_volatility": macro["fx_volatility"],
                "policy_stability": macro["policy_stability"],
                "keywords": keywords or "no matched keyword",
                "policy_keywords": keywords or "no matched keyword",
                "news_article_count": article_count,
                "macro_source": macro["source"],
                "market_source": market["source"],
                "policy_source": policy_source,
                "llm_source": "OpenAI Chat Completions API" if llm_api_key else "not generated: OPENAI_API_KEY is not set",
            }
            row.update(calculate_country_specific_risk(row))
            row["riskDifference"] = round(row["countrySpecificRiskScore"] - total, 2)
            row.update(estimate_probability_metrics(row, energy))
            if llm_api_key:
                try:
                    row["llm_interpretation"] = call_llm_interpretation(row, llm_api_key)
                    row["llm_status"] = "generated"
                    time.sleep(0.05)
                except Exception as exc:
                    row["llm_interpretation"] = fallback_explain(row)
                    row["llm_status"] = f"LLM call failed: {exc}"
                    warnings.append(f"LLM {country.country} {energy.name}: {exc}")
            else:
                row["llm_interpretation"] = ""
                row["llm_status"] = "OPENAI_API_KEY is not set; no LLM interpretation generated"
            row["ai"] = row["llm_interpretation"]
            row["ai_interpretation"] = row["llm_interpretation"]
            rows.append(row)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "row_count": len(rows),
        "country_count": len(COUNTRIES),
        "energy_count": len(ENERGY_TYPES),
        "sources": [
            "World Bank API: GDP growth, inflation, debt-to-GDP, exchange rate",
            "World Governance Indicators CSV: political stability",
            "Yahoo Finance chart API: ETF and SPY monthly prices for CAPM alpha/beta",
            "GDELT 2.1 DOC API: energy policy/news keyword search",
            "OpenAI Chat Completions API: LLM interpretation when OPENAI_API_KEY is available",
        ],
        "llm_status": "generated" if llm_api_key else "not generated: OPENAI_API_KEY is not set",
        "warnings": warnings[:80],
    }
    write_csv(rows, OUTPUT_DIR / "real_energy_risk_results.csv")
    write_dashboard_js(rows, metadata, OUTPUT_DIR / "real_dashboard_data.js")
    (OUTPUT_DIR / "real_data_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Created {len(rows)} real-data rows.")
    print(f"Dashboard data: {OUTPUT_DIR / 'real_dashboard_data.js'}")
    print(f"CSV: {OUTPUT_DIR / 'real_energy_risk_results.csv'}")
    if warnings:
        print(f"Warnings: {len(warnings)}")


if __name__ == "__main__":
    main()
