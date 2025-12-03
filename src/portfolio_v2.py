"""Version 2 portfolio optimization utilities with BONMIN mixed-integer modeling.

This module extends the original Markowitz pipeline by adding binary activation
variables, sector constraints, risk scenarios, and a simple paper-trading style
backtest. It is intended to be self-contained for quick experimentation (e.g.,
in Google Colab) assuming BONMIN is available on the execution path.
"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    NonNegativeReals,
    Objective,
    Param,
    Set,
    Var,
    maximize,
)
from pyomo.opt import SolverFactory, TerminationCondition

IPOPT_PATH = "/content/bin/ipopt"


class SolverUnavailableError(RuntimeError):
    """Raised when a required solver binary cannot be located or is unusable."""


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def download_monthly_returns(
    tickers: Sequence[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Download daily prices from Yahoo Finance and convert to monthly returns."""

    price_dict: dict[str, pd.Series] = {}

    for ticker in tickers:
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval="1d",
                progress=False,
                auto_adjust=False,
            )
            if df.empty:
                print(f"[warn] no data for {ticker}, skipping")
                continue
            if "Close" not in df.columns:
                print(f"[warn] 'Close' column missing for {ticker}, skipping")
                continue

            close_series = df["Close"]
            if not close_series.empty and isinstance(close_series.index, pd.DatetimeIndex):
                price_dict[ticker] = close_series
            else:
                print(
                    f"[warn] Skipping {ticker} due to empty or malformed 'Close' series/index"
                )

        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"[error] downloading {ticker}: {exc}")

    if not price_dict:
        raise RuntimeError("No valid price data downloaded for any ticker. Check tickers/date range.")

    daily_prices = pd.concat(price_dict.values(), axis=1, keys=price_dict.keys())
    daily_prices = daily_prices.dropna(how="all")
    daily_returns = daily_prices.pct_change().dropna(how="all")

    monthly_returns = (1 + daily_returns).resample("ME").prod() - 1
    monthly_returns = monthly_returns.dropna(how="any")

    if isinstance(monthly_returns, pd.Series):
        monthly_returns = monthly_returns.to_frame()

    print("Monthly returns shape:", monthly_returns.shape)
    return monthly_returns


def _normalize_asset_columns(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure asset columns are flat strings for downstream optimization models."""
    returns_df = returns_df.copy()
    # Flatten any MultiIndex / non-string column labels to simple strings
    returns_df.columns = [str(c) for c in returns_df.columns]
    return returns_df

# ---------------------------------------------------------------------------
# Sector helpers
# ---------------------------------------------------------------------------

def get_sector_mapping(tickers: Iterable[str]) -> Dict[str, str]:
    """Return a mapping from ticker to sector using yfinance, with fallbacks.

    Each ticker is queried via ``Ticker.info``. When no sector is available the
    ticker is assigned to the generic sector ``"Unknown"`` so the grouping
    constraints remain feasible.
    """

    sector_map: Dict[str, str] = {}
    for ticker in tickers:
        sector = None
        try:
            info = yf.Ticker(ticker).info
            sector = info.get("sector") if isinstance(info, dict) else None
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"[warn] unable to fetch sector for {ticker}: {exc}")
        sector_map[ticker] = sector or "Unknown"
    return sector_map


# ---------------------------------------------------------------------------
# Core Markowitz model builders
# ---------------------------------------------------------------------------

def build_markowitz_model(returns_df: pd.DataFrame):
    """Build a continuous Pyomo Markowitz model (long-only, fully invested)."""

    returns_df = _normalize_asset_columns(returns_df)

    assets = list(returns_df.columns)
    mu = returns_df.mean()
    sigma = returns_df.cov()

    model = ConcreteModel()
    model.Assets = Set(initialize=assets)
    model.x = Var(model.Assets, within=NonNegativeReals, bounds=(0, 1))
    model.mu = Param(model.Assets, initialize=mu.to_dict())

    sigma_dict = {(i, j): float(sigma.loc[i, j]) for i in assets for j in assets}
    model.Sigma = Param(model.Assets, model.Assets, initialize=sigma_dict)

    def total_return(m):
        return sum(m.mu[a] * m.x[a] for a in m.Assets)

    model.obj = Objective(rule=total_return, sense=maximize)

    def budget(m):
        return sum(m.x[a] for a in m.Assets) == 1

    model.budget = Constraint(rule=budget)

    return model, assets, mu, sigma


def build_markowitz_mip_model(
    returns_df: pd.DataFrame,
    sector_map: Dict[str, str],
    min_weight: float = 0.02,
    max_weight: float = 0.2,
    min_stocks: int = 5,
):
    """Build a mixed-integer Markowitz model with activation variables.

    The binary variables ``y[i]`` denote whether asset ``i`` is included.
    Continuous weights ``x[i]`` are linked to ``y[i]`` using bounds so that
    selecting a stock forces its weight to lie between ``min_weight`` and
    ``max_weight``.
    """

    returns_df = _normalize_asset_columns(returns_df)

    assets = list(returns_df.columns)
    mu = returns_df.mean()
    sigma = returns_df.cov()

    model = ConcreteModel()
    model.Assets = Set(initialize=assets)
    model.Sectors = Set(initialize=sorted(set(sector_map.get(a, "Unknown") for a in assets)))

    model.x = Var(model.Assets, within=NonNegativeReals, bounds=(0, 1))
    model.y = Var(model.Assets, within=Binary)
    model.mu = Param(model.Assets, initialize=mu.to_dict())

    sigma_dict = {(i, j): float(sigma.loc[i, j]) for i in assets for j in assets}
    model.Sigma = Param(model.Assets, model.Assets, initialize=sigma_dict)

    def total_return(m):
        return sum(m.mu[a] * m.x[a] for a in m.Assets)

    model.obj = Objective(rule=total_return, sense=maximize)

    def budget(m):
        return sum(m.x[a] for a in m.Assets) == 1

    model.budget = Constraint(rule=budget)

    def min_link(m, a):
        return m.x[a] >= min_weight * m.y[a]

    def max_link(m, a):
        return m.x[a] <= max_weight * m.y[a]

    model.min_link = Constraint(model.Assets, rule=min_link)
    model.max_link = Constraint(model.Assets, rule=max_link)

    # Ensure at least one selection per sector
    sector_to_assets = {
        sector: [a for a in assets if sector_map.get(a, "Unknown") == sector]
        for sector in model.Sectors
    }

    def sector_cover_rule(m, s):
        asset_list = sector_to_assets[s]
        return sum(m.y[a] for a in asset_list) >= 1

    model.sector_cover = Constraint(model.Sectors, rule=sector_cover_rule)

    # Overall minimum number of holdings
    def min_count(m):
        return sum(m.y[a] for a in m.Assets) >= min_stocks

    model.min_count = Constraint(rule=min_count)

    return model, assets, mu, sigma


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def portfolio_variance(weights: Sequence[float], sigma: pd.DataFrame | np.ndarray) -> float:
    weights_arr = np.array(weights, dtype=float)
    sigma_arr = sigma.values if isinstance(sigma, pd.DataFrame) else sigma
    return float(weights_arr @ sigma_arr @ weights_arr)


def choose_medium_frontier_point(frontier_df: pd.DataFrame) -> pd.Series:
    """Return the middle row of the frontier as a simple "medium risk" proxy."""

    if frontier_df.empty:
        raise ValueError("Frontier data is empty; cannot select a medium point.")
    mid_idx = len(frontier_df) // 2
    return frontier_df.iloc[mid_idx]


# ---------------------------------------------------------------------------
# Efficient frontier solvers
# ---------------------------------------------------------------------------

def sweep_efficient_frontier_nlp(
    returns_df: pd.DataFrame,
    sector_map: Dict[str, str],
    bonmin_path: str | None = None,
    n_points: int = 60,
    min_weight: float = 0.02,
    max_weight: float = 0.2,
    min_stocks: int = 5,
):
    """Solve a continuous efficient frontier using IPOPT as a fallback.

    This variant mirrors the MIP API but omits binary activation variables and
    selection count constraints, making it compatible with continuous solvers
    such as IPOPT. A variance cap is swept across ``n_points`` values to trace
    the efficient frontier.
    """

    model, assets, mu, sigma = build_markowitz_model(returns_df)
    sigma_np = sigma.values
    n_assets = len(assets)

    eq_weights = np.ones(n_assets) / n_assets
    min_var = portfolio_variance(eq_weights, sigma_np)
    max_var_single = float(np.max(np.diag(sigma_np)))

    min_cap = max(min_var * 0.5, 1e-8)
    max_cap = max(max_var_single * 1.5, min_cap * 5)
    caps = np.linspace(min_cap, max_cap, n_points)

    try:
        solver = SolverFactory("ipopt", executable=IPOPT_PATH)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise SolverUnavailableError(
            f"Failed to create IPOPT solver with executable={IPOPT_PATH}."
        ) from exc

    if not solver.available(False):
        raise SolverUnavailableError(
            "IPOPT solver is not available. Please ensure IPOPT is installed at IPOPT_PATH."
        )

    frontier_data = {"Risk": [], "Return": []}
    alloc_data = {asset: [] for asset in assets}
    alloc_data["Risk"] = []

    print(
        f"Solving {len(caps)} NLP portfolio problems from cap={min_cap:.3e} to {max_cap:.3e} using IPOPT..."
    )

    for cap in caps:
        if hasattr(model, "risk_constraint"):
            model.del_component(model.risk_constraint)

        def risk_con(m):
            return (
                sum(m.Sigma[i, j] * m.x[i] * m.x[j] for i in m.Assets for j in m.Assets)
                <= cap
            )

        model.risk_constraint = Constraint(rule=risk_con)

        try:
            result = solver.solve(model, tee=False)
        except Exception as exc:  # pragma: no cover - solver safety
            print(f"[warn] IPOPT failed for cap={cap:.4e}: {exc}")
            continue

        term = result.solver.termination_condition
        if term not in (TerminationCondition.optimal, TerminationCondition.locallyOptimal):
            print(f"[warn] IPOPT did not converge for cap={cap:.4e} (status={term})")
            continue

        weights = [model.x[a]() for a in assets]
        realized_var = portfolio_variance(weights, sigma_np)
        realized_ret = float(np.dot(mu.values, np.array(weights)))

        frontier_data["Risk"].append(realized_var)
        frontier_data["Return"].append(realized_ret)

        alloc_data["Risk"].append(realized_var)
        for asset, weight in zip(assets, weights):
            alloc_data[asset].append(weight)

    if len(frontier_data["Risk"]) == 0:
        raise RuntimeError(
            "No feasible portfolios found with IPOPT. Try different tickers/dates or adjust inputs."
        )

    frontier_df = pd.DataFrame(frontier_data).sort_values("Risk").reset_index(drop=True)
    alloc_df = pd.DataFrame(alloc_data).sort_values("Risk").set_index("Risk")

    return frontier_df, alloc_df

def sweep_efficient_frontier_mip(
    returns_df: pd.DataFrame,
    sector_map: Dict[str, str],
    bonmin_path: str | None = None,
    n_points: int = 60,
    min_weight: float = 0.02,
    max_weight: float = 0.2,
    min_stocks: int = 5,
):
    """Solve the mixed-integer efficient frontier using BONMIN.

    The solver explores a grid of variance caps to trace the frontier while
    respecting activation variables, linking constraints, sector coverage, and
    a minimum number of holdings.
    """

    model, assets, mu, sigma = build_markowitz_mip_model(
        returns_df,
        sector_map,
        min_weight=min_weight,
        max_weight=max_weight,
        min_stocks=min_stocks,
    )
    sigma_np = sigma.values
    n_assets = len(assets)

    eq_weights = np.ones(n_assets) / n_assets
    min_var = portfolio_variance(eq_weights, sigma_np)
    max_var_single = float(np.max(np.diag(sigma_np)))

    min_cap = max(min_var * 0.5, 1e-8)
    max_cap = max(max_var_single * 1.5, min_cap * 5)
    caps = np.linspace(min_cap, max_cap, n_points)

    try:
        solver = (
            SolverFactory("bonmin", executable=bonmin_path)
            if bonmin_path
            else SolverFactory("bonmin")
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        raise SolverUnavailableError(
            "BONMIN solver could not be created. Verify installation or bonmin_path."
        ) from exc

    if not solver.available(False):
        raise SolverUnavailableError(
            "BONMIN solver is not available. Please install BONMIN or adjust bonmin_path."
        )

    frontier_data = {"Risk": [], "Return": []}
    alloc_data = {asset: [] for asset in assets}
    alloc_data["Risk"] = []

    print(
        f"Solving {len(caps)} MIP portfolio problems from cap={min_cap:.3e} to {max_cap:.3e} using BONMIN..."
    )

    for cap in caps:
        if hasattr(model, "risk_constraint"):
            model.del_component(model.risk_constraint)

        def risk_con(m):
            return (
                sum(m.Sigma[i, j] * m.x[i] * m.x[j] for i in m.Assets for j in m.Assets)
                <= cap
            )

        model.risk_constraint = Constraint(rule=risk_con)

        result = solver.solve(model, tee=False)
        term = result.solver.termination_condition
        if term not in (TerminationCondition.optimal, TerminationCondition.locallyOptimal):
            print(f"[warn] solver did not converge for cap={cap:.4e} (status={term})")
            continue

        weights = [model.x[a]() for a in assets]
        realized_var = portfolio_variance(weights, sigma_np)
        realized_ret = float(np.dot(mu.values, np.array(weights)))

        frontier_data["Risk"].append(realized_var)
        frontier_data["Return"].append(realized_ret)

        alloc_data["Risk"].append(realized_var)
        for asset, weight in zip(assets, weights):
            alloc_data[asset].append(weight)

    if len(frontier_data["Risk"]) == 0:
        raise RuntimeError("No feasible portfolios found. Try different tickers/dates.")

    frontier_df = pd.DataFrame(frontier_data).sort_values("Risk").reset_index(drop=True)
    alloc_df = pd.DataFrame(alloc_data).sort_values("Risk").set_index("Risk")

    return frontier_df, alloc_df


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_frontier(frontier_df: pd.DataFrame) -> None:
    """Plot the efficient frontier."""

    plt.figure(figsize=(8, 5))
    plt.plot(
        frontier_df["Risk"],
        frontier_df["Return"],
        marker="o",
        linestyle="-",
        markersize=3,
    )
    plt.title("Efficient Frontier")
    plt.xlabel("Portfolio Risk (Variance)")
    plt.ylabel("Expected Monthly Return")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_allocations(alloc_df: pd.DataFrame) -> None:
    """Plot allocation weights as a function of portfolio risk."""

    plt.figure(figsize=(10, 6))
    for col in alloc_df.columns:
        plt.plot(
            alloc_df.index,
            alloc_df[col],
            marker="o",
            markersize=3,
            linewidth=0.7,
            label=str(col),
        )
    plt.title("Optimal Allocation vs Portfolio Risk")
    plt.xlabel("Portfolio Risk (Variance)")
    plt.ylabel("Proportion Invested")
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Risk scenario construction
# ---------------------------------------------------------------------------

def build_risk_scenarios(
    returns_df: pd.DataFrame,
    frontier_df: pd.DataFrame,
    alloc_df: pd.DataFrame,
) -> pd.DataFrame:
    """Construct low/medium/high risk scenarios with weights and risk stats."""

    assets = list(returns_df.columns)
    mu = returns_df.mean()
    sigma = returns_df.cov()

    scenarios: list[dict[str, object]] = []

    # Low risk: equal weight portfolio
    eq_weights = np.ones(len(assets)) / len(assets)
    scenarios.append(
        {
            "Scenario": "low",
            "Weights": dict(zip(assets, eq_weights)),
            "ExpectedReturn": float(np.dot(mu.values, eq_weights)),
            "Variance": portfolio_variance(eq_weights, sigma),
        }
    )

    # Medium risk: middle of frontier
    mid_point = choose_medium_frontier_point(frontier_df)
    mid_risk = mid_point["Risk"]
    medium_weights = alloc_df.loc[mid_risk].reindex(assets)
    scenarios.append(
        {
            "Scenario": "medium",
            "Weights": medium_weights.to_dict(),
            "ExpectedReturn": float(np.dot(mu.values, medium_weights.values)),
            "Variance": float(mid_risk),
        }
    )

    # High risk: all-in on asset with highest expected return
    best_asset = mu.idxmax()
    high_weights = {a: 1.0 if a == best_asset else 0.0 for a in assets}
    scenarios.append(
        {
            "Scenario": "high",
            "Weights": high_weights,
            "ExpectedReturn": float(mu.loc[best_asset]),
            "Variance": float(sigma.loc[best_asset, best_asset]),
        }
    )

    return pd.DataFrame(scenarios)


# ---------------------------------------------------------------------------
# Paper trading / out-of-sample backtest
# ---------------------------------------------------------------------------

def _compute_strategy_metrics(weights: Dict[str, float], returns: pd.DataFrame) -> dict:
    # Align weights to columns and normalize
    aligned_weights = pd.Series(weights).reindex(returns.columns).fillna(0.0)
    aligned_weights /= aligned_weights.sum() if aligned_weights.sum() != 0 else 1.0

    portfolio_returns = returns.mul(aligned_weights, axis=1).sum(axis=1)
    cumulative = float((1 + portfolio_returns).prod() - 1)
    mean_monthly = float(portfolio_returns.mean())
    volatility = float(portfolio_returns.std())
    return {
        "CumulativeReturn": cumulative,
        "MeanMonthlyReturn": mean_monthly,
        "Volatility": volatility,
    }


def backtest_strategies(
    tickers: Sequence[str],
    train_start: str = "2024-01-01",
    train_end: str = "2025-07-31",
    test_start: str = "2025-08-01",
    test_end: str = "2025-10-31",
    bonmin_path: str | None = None,
    min_weight: float = 0.02,
    max_weight: float = 0.2,
    min_stocks: int = 5,
) -> pd.DataFrame:
    """Compare optimized vs benchmark strategies on an out-of-sample window."""

    # Training phase
    train_returns = download_monthly_returns(tickers, train_start, train_end)
    sector_map = get_sector_mapping(train_returns.columns)

    try:
        frontier_df, alloc_df = sweep_efficient_frontier_mip(
            train_returns,
            sector_map,
            bonmin_path=bonmin_path,
            n_points=40,
            min_weight=min_weight,
            max_weight=max_weight,
            min_stocks=min_stocks,
        )
    except SolverUnavailableError as exc:
        print(f"[info] BONMIN unavailable during backtest ({exc}); using IPOPT fallback.")
        frontier_df, alloc_df = sweep_efficient_frontier_nlp(
            train_returns,
            sector_map,
            bonmin_path=bonmin_path,
            n_points=40,
            min_weight=min_weight,
            max_weight=max_weight,
            min_stocks=min_stocks,
        )
    scenarios = build_risk_scenarios(train_returns, frontier_df, alloc_df)
    optimized_weights = scenarios.loc[scenarios["Scenario"] == "medium", "Weights"].iloc[0]
    optimized_weights = dict(optimized_weights)

    # Testing data
    test_tickers = list(dict.fromkeys(list(tickers) + ["^GSPC", "BTC-USD"]))
    test_returns = download_monthly_returns(test_tickers, test_start, test_end)

    # Slice the optimized portfolio returns to available tickers
    opt_returns = test_returns[[t for t in tickers if t in test_returns.columns]]
    eq_weights = {t: 1.0 / len(opt_returns.columns) for t in opt_returns.columns}

    metrics = {
        "Optimized_MediumRisk": _compute_strategy_metrics(optimized_weights, opt_returns),
        "EqualWeight": _compute_strategy_metrics(eq_weights, opt_returns),
    }

    # Benchmarks: S&P 500 and Bitcoin
    if "^GSPC" in test_returns.columns:
        metrics["SP500"] = _compute_strategy_metrics({"^GSPC": 1.0}, test_returns[["^GSPC"]])
    if "BTC-USD" in test_returns.columns:
        metrics["Bitcoin"] = _compute_strategy_metrics({"BTC-USD": 1.0}, test_returns[["BTC-USD"]])

    metrics_df = pd.DataFrame.from_dict(metrics, orient="index")
    metrics_df.index.name = "Strategy"
    return metrics_df.reset_index()


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def run_portfolio_example_v2(
    tickers: Sequence[str],
    start_date: str,
    end_date: str,
    bonmin_path: str | None = None,
    n_points: int = 60,
    min_weight: float = 0.02,
    max_weight: float = 0.2,
    min_stocks: int = 5,
    run_backtest: bool = False,
):
    """End-to-end demo: download data, solve MIP frontier, and plot results."""

    monthly_returns = download_monthly_returns(tickers, start_date, end_date)
    sector_map = get_sector_mapping(monthly_returns.columns)

    try:
        frontier_df, alloc_df = sweep_efficient_frontier_mip(
            monthly_returns,
            sector_map,
            bonmin_path=bonmin_path,
            n_points=n_points,
            min_weight=min_weight,
            max_weight=max_weight,
            min_stocks=min_stocks,
        )
    except SolverUnavailableError as exc:
        print(f"[info] BONMIN unavailable ({exc}); falling back to IPOPT continuous frontier.")
        frontier_df, alloc_df = sweep_efficient_frontier_nlp(
            monthly_returns,
            sector_map,
            bonmin_path=bonmin_path,
            n_points=n_points,
            min_weight=min_weight,
            max_weight=max_weight,
            min_stocks=min_stocks,
        )

    scenarios_df = build_risk_scenarios(monthly_returns, frontier_df, alloc_df)

    plot_frontier(frontier_df)
    plot_allocations(alloc_df)

    backtest_df = None
    if run_backtest:
        backtest_df = backtest_strategies(
            tickers,
            train_start=start_date,
            train_end=end_date,
            test_start=end_date,
            test_end=end_date,
            bonmin_path=bonmin_path,
            min_weight=min_weight,
            max_weight=max_weight,
            min_stocks=min_stocks,
        )

    return monthly_returns, frontier_df, alloc_df, scenarios_df, backtest_df


__all__ = [
    "download_monthly_returns",
    "get_sector_mapping",
    "build_markowitz_model",
    "build_markowitz_mip_model",
    "portfolio_variance",
    "sweep_efficient_frontier_mip",
    "sweep_efficient_frontier_nlp",
    "plot_frontier",
    "plot_allocations",
    "build_risk_scenarios",
    "backtest_strategies",
    "run_portfolio_example_v2",
]
