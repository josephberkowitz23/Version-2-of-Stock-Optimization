# Mean-Variance Portfolio Optimization Pipeline

This repository downloads historical prices from Yahoo Finance, converts them to
monthly returns, sweeps a variance cap to trace the efficient frontier, and
visualizes both the frontier and the corresponding allocations. The v2 pipeline
adds BONMIN-powered mixed-integer modeling, sector and linking constraints,
risk scenario analysis, and an optional paper-trading style backtest.

## Guided Google Colab walkthrough
> Run these three cells in order inside a Colab notebook. Each cell includes a
> short note so you know what it is doing.

**1) Clone and set up the repo** – pull the code, move into the folder, and
install Python dependencies.
```python
# Clone the repo (edit <your-username> if needed)
!git clone https://github.com/josephberkowitz23/Version-2-of-Stock-Optimization.git

# Move into the project folder
%cd Version-2-of-Stock-Optimization

# Install Python dependencies
!python -m pip install -r requirements
```

**2) Add the solver** – download BONMIN (plus supporting IPOPT binaries) to a
local `/content/bin` directory.
```python
!idaes get-extensions --to /content/bin
```

**3) Run the tutorial** – set your tickers and execute the v2 pipeline, pointing
to the BONMIN binary you just downloaded (IPOPT still works for the continuous
model, but BONMIN is required for the mixed-integer version).
```python
TICKERS = "AAPL MSFT NVDA AMZN"

!python - <<'EOF'
from src.portfolio_v2 import run_portfolio_example_v2

run_portfolio_example_v2(
    tickers=TICKERS.split(),
    start_date="2020-01-01",
    end_date="2024-01-01",
    bonmin_path="/content/bin/bonmin",
    n_points=60,
    run_backtest=True,
)
EOF
```

When the run finishes, the notebook will display two matplotlib figures: the
efficient frontier and the allocation-by-risk chart. Open up the output folder
within your drive in order to see the formulated figures.

## What changed in v2?
- `src/portfolio_v2.py` adds BONMIN-backed mixed-integer optimization with
  binary asset activation, sector exposure caps, and linking constraints.
- Risk scenario generation and an optional paper-trading comparison are built
  into the tutorial flow.
- The guided notebook now runs v2 by default; the original pipeline remains for
  backwards compatibility.

## Project Layout
```
bdm_fall2025_opt/
├── main.py                   # CLI entry point (routes to the v2 backend)
├── src/portfolio_v2.py       # NEW v2 pipeline (MIP, BONMIN, sectors, linking, scenarios, backtesting)
├── src/portfolio_pipeline.py # v1 pipeline retained for backward compatibility (v2 recommended)
├── README.md                 # You're reading it
├── requirements              # Python dependencies
└── .gitignore
```

## Notes on BONMIN/IPOPT via IDAES
- `idaes-pse` ships a helper (`idaes get-extensions`) that downloads BONMIN and
  IPOPT together and compiles them for the current platform.
- In Colab, the solver binaries typically land under `/content/bin/bonmin` and
  `/content/bin/ipopt`; on a local machine, they will be inside `~/.idaes/bin`
  unless you override the target folder.
- Use BONMIN for the mixed-integer tutorial; IPOPT remains available for the
  continuous-only (v1-style) model if you need it.
