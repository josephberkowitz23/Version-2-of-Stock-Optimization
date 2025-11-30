"""Command-line interface for the portfolio optimization pipeline."""

from __future__ import annotations

import argparse
from typing import List

from src.portfolio_pipeline import IPOPT_PATH, run_portfolio_example


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mean-variance efficient frontier sweep using Pyomo + IPOPT.",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["GE", "KO", "NVDA"],
        help="List of ticker symbols (space separated).",
    )
    parser.add_argument(
        "--start-date",
        default="2020-01-01",
        help="Start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        default="2024-01-01",
        help="End date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--ipopt-path",
        default=IPOPT_PATH,
        help="Path to the IPOPT executable (installed via idaes get-extensions).",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=200,
        help="Number of variance caps to sweep across the frontier.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_portfolio_example(
        tickers=args.tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        ipopt_path=args.ipopt_path,
        n_points=args.n_points,
    )


if __name__ == "__main__":
    main()
