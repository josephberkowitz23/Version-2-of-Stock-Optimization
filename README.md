# Mean-Variance Portfolio Optimization Pipeline

This repository downloads historical prices from Yahoo Finance, converts them to
monthly returns, sweeps a variance cap to trace the efficient frontier, and
visualizes both the frontier and the corresponding allocations.

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

**2) Add the solver** – download IPOPT (plus supporting binaries) to a local
`/content/bin` directory.
```python
!idaes get-extensions --to /content/bin
```

**3) Run the tutorial** – set your tickers and execute the main script, pointing
to the IPOPT binary you just downloaded.
```python
TICKERS = "AAPL MSFT NVDA AMZN GOOGL"

!python main.py \
    --tickers {TICKERS} \
    --start-date 2020-01-01 \
    --end-date 2024-01-01 \
    --ipopt-path /content/bin/ipopt
```

When the run finishes, the notebook will display two matplotlib figures: the
efficient frontier and the allocation-by-risk chart. Open up the output folder
within your drive in order to see the formulated figures.

## Project Layout
```
bdm_fall2025_opt/
├── main.py                  # CLI entry point
├── src/portfolio_pipeline.py# Data download, Pyomo model, plotting
├── README.md                # You're reading it
├── requirements.txt         # Python dependencies
└── .gitignore
```

## Notes on IPOPT via IDAES
- `idaes-pse` ships a helper (`idaes get-extensions`) that downloads IPOPT and
  compiles it for the current platform.
- In Colab, the solver binary typically lands under `/content/bin/ipopt`; on a
  local machine, it will be inside `~/.idaes/bin/ipopt` unless you override the
  target folder.
- Pass the exact path to `--ipopt-path` if it differs from the default defined
  in `src/portfolio_pipeline.py`.
