# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

1. Before writing any code, describe your approach and wait for approval. Always ask clarifying questions before writing any code if requirements are ambiguous.

2. If a task requires changes to more than 3 files, stop and break it into smaller tasks first.

3. After writing code, list what could break and suggest tests to cover it.

4. When there's a bug, start by writing a test that reproduces it, then fix it until the test passes.

5. Every time I correct you, add a new rule to the CLAUDE .md file so it never happens again.

## Project Overview

Portfolio risk analysis toolkit for analyzing investment portfolios. Uses Python with Pandas and yfinance for market data. Based on EDHEC risk management curriculum from https://github.com/paulhake/portfolio_analysis_edhc.

**Technology Stack**: Python, Pandas, NumPy, SciPy, Matplotlib, yfinance

**Data Source**: Yahoo Finance (daily stock prices and volume via yfinance)

## Core Architecture

### Risk Analysis Modules

Two main risk toolkit modules with overlapping but distinct implementations:

- **[edhec_risk_ph.py](edhec_risk_ph.py)**: Extended version (~950 lines)
  - Yahoo Finance integration via `get_from_yahoo(tickers, period='10y')`
  - `calculate_daily_returns()` handles multi-level DataFrame columns from yfinance
  - CIR interest rate model: `cir()`, `ann_to_inst()`, `inst_to_ann()`
  - Bond pricing: `bond_cash_flows()`, `bond_price()`, `macaulay_duration()`
  - Enhanced CPPI strategy with drawdown constraints
  - Complete fixed income analytics

- **[edhec_risk_kit_129.py](edhec_risk_kit_129.py)**: Original EDHEC course version (~763 lines)
  - More compact, focused on Ken French datasets
  - Data loaders expect `../data/` directory with CSV files
  - Core risk functions identical to edhec_risk_ph.py

### Common Functions Across Both Modules

**Risk Metrics:**
- `var_historic(r, level=5)`: Historic Value at Risk
- `var_gaussian(r, level=5, modified=False)`: Parametric VaR with Cornish-Fisher modification
- `cvar_historic(r, level=5)`: Conditional VaR (Expected Shortfall)
- `semi_deviation(r)` / `semideviation(r)`: Downside deviation
- `skewness(r)`, `kurtosis(r)`: Distribution moments
- `is_normal(r, level=0.01)`: Jarque-Bera normality test

**Portfolio Optimization:**
- `minimize_vol(target_return, er, cov)`: Min volatility for target return
- `msr(risk_free_rate, er, cov)`: Maximum Sharpe Ratio portfolio
- `gmv(cov)`: Global Minimum Volatility portfolio
- `portfolio_return(weights, returns)`: Calculate portfolio return
- `portfolio_vol(weights, cov)`: Calculate portfolio volatility

**Analysis & Visualization:**
- `plot_ef(n_points, er, cov, show_cml=False, risk_free_rate=0, show_ew=False, show_gmv=False)`: Plot efficient frontier
- `drawdown(rets)`: Calculate wealth index, peaks, and drawdowns
- `summary_stats(r, riskfree_rate=0.03)`: Comprehensive risk metrics table
- `annualize_rets(r, periods_per_year)`, `annualize_vol(r, periods_per_year)`
- `sharpe_ratio(r, riskfree_rate, periods_per_year)`

**CPPI Strategy:**
- `run_cppi(risky_r, safe_r=None, m=3, start=1000, floor=0.8, riskfree_rate=0.03, drawdown=None)`
- `show_cppi(n_scenarios=50, mu=0.07, sigma=0.15, m=3, floor=0., riskfree_rate=0.03, y_max=100)`
- Multiple allocator functions: `fixedmix_allocator`, `glidepath_allocator`, `floor_allocator`, `drawdown_allocator`

**Monte Carlo & Simulation:**
- `gbm(n_years=10, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_year=12, s0=100.0, prices=True)`: Geometric Brownian Motion

## Data Requirements

**Primary Data Source**: Yahoo Finance via yfinance package

- Use `erk.get_from_yahoo(tickers, period='10y')` to download historical price data
- Data is fetched in real-time; no local CSV files required
- Returns multi-level DataFrame with 'Close' and 'Volume' columns

### Data Organization

- **Input**: `stocks_list.csv` - List of ticker symbols to analyze (one per line with 'ticker' header)
- **Output**: `/data` folder contains downloaded price and returns data:
  - `portfolio_prices_5y.csv` - Daily closing prices for all tickers
  - `portfolio_returns_5y.csv` - Daily returns calculated from prices
- **Note**: CSV files in `/data` are excluded from git via `.gitignore`

## Development Commands

Currently uses Jupyter notebooks for analysis and prototyping. A UI component will be added in the future.

### Initial Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Project

```bash
# Activate virtual environment (if not already activated)
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Launch Jupyter Notebook
jupyter notebook

# Or use JupyterLab
jupyter lab
```

### Working with the Notebooks

- **[portfolio_data_download.ipynb](portfolio_data_download.ipynb)**: **PRIMARY WORKFLOW NOTEBOOK**
  - Downloads 5 years of daily price data from Yahoo Finance for stocks in `stocks_list.csv`
  - Calculates daily returns using `erk.calculate_daily_returns()`
  - Saves prices and returns to `/data` folder as CSV files
  - Plots cumulative returns for user-selected stocks
  - User can edit `selected_stocks` list to compare different tickers
  - Data quality checks for missing values
  - Outputs: `data/portfolio_prices_5y.csv` and `data/portfolio_returns_5y.csv`

- **[max drawdown.ipynb](max%20drawdown.ipynb)**:
  - Loads Fama-French data and calculates drawdowns
  - Demonstrates the `drawdown()` function implementation
  - Identifies maximum drawdown periods (e.g., -83% for SmallCap in 1932-05)

- **[risk_builder.ipynb](risk_builder.ipynb)**:
  - Statistical analysis: skewness, kurtosis, Jarque-Bera normality tests
  - Compares hedge fund indices vs normal distribution
  - Demonstrates that most financial returns are not normally distributed

- **[sharpe_portfolio.ipynb](sharpe_portfolio.ipynb)**:
  - Portfolio optimization using Ken French 30 industry portfolios
  - Efficient frontier plotting with Capital Market Line (CML)
  - Demonstrates `msr()`, `plot_ef()` with various overlays (EW, GMV, MSR)

## Common Workflow Patterns

### Download Data from Yahoo Finance

```python
import edhec_risk_ph as erk

# Download historical data
tickers = ['AAPL', 'MSFT', 'GOOGL']
prices = erk.get_from_yahoo(tickers, period='10y')

# Calculate daily returns
returns = erk.calculate_daily_returns(prices, column='Close')
```

### Portfolio Optimization

```python
import edhec_risk_ph as erk

# Load industry returns (or use your own return data)
ind = erk.get_ind_returns()
er = erk.annualize_rets(ind['1996':'2000'], 12)
cov = ind['1996':'2000'].cov()

# Find maximum Sharpe ratio portfolio
weights = erk.msr(risk_free_rate=0.03, er=er, cov=cov)

# Plot efficient frontier with overlays
erk.plot_ef(20, er, cov, show_cml=True, risk_free_rate=0.03,
            show_ew=True, show_gmv=True)
```

### Risk Analysis

```python
# Calculate comprehensive risk metrics
stats = erk.summary_stats(returns, riskfree_rate=0.03)
# Returns DataFrame with: Annualized Return, Vol, Skewness, Kurtosis,
# VaR, CVaR, Sharpe Ratio, Max Drawdown

# Drawdown analysis
dd_result = erk.drawdown(returns)
max_dd = dd_result['Drawdown'].min()
worst_date = dd_result['Drawdown'].idxmin()
```

### CPPI Strategy Backtesting

```python
# Run CPPI backtest
cppi_result = erk.run_cppi(
    risky_r=risky_returns,
    m=3,                    # multiplier
    start=1000,            # starting capital
    floor=0.8,             # 80% floor
    riskfree_rate=0.03,
    drawdown=0.25          # optional max drawdown constraint
)

# Access results
wealth_history = cppi_result["Wealth"]
risky_allocation = cppi_result["Risky Weight"]
```

## Important Implementation Details

### Data Format Conventions

- **Returns**: Decimal format (0.05 = 5%), not percentage
- **Index**: PeriodIndex with 'M' frequency for monthly data
- **Annualization**: Use `periods_per_year=12` for monthly, `252` for daily
- **Risk-free rate**: Annual rate as decimal (0.03 = 3%)

### Function Parameter Patterns

- Portfolio weights from optimizers (`msr`, `minimize_vol`, `gmv`) return numpy arrays ordered by input expected returns
- `plot_ef()` returns matplotlib axes object for overlaying additional plots
- CPPI functions can accept both Series and DataFrame inputs; DataFrames process each column independently
- Most risk functions (VaR, CVaR, etc.) automatically aggregate DataFrames column-wise using `.aggregate()`

### scipy.optimize Usage

Portfolio optimization uses `scipy.optimize.minimize` with:
- Method: `'SLSQP'` (Sequential Least Squares Programming)
- Constraints: weights sum to 1, optional target return constraint
- Bounds: (0.0, 1.0) for long-only portfolios

### Known Quirks

- `edhec_risk_kit_129.py` has bug on line 36: uses `is` instead of `==` for string comparison
- Both modules have hardcoded data paths; adjust as needed for your directory structure
- `get_total_market_index_returns()` computes cap-weighted market index from industry data

### Critical Bug Fix: Daily Returns Calculation

**NEVER use `.dropna()` after calculating returns with `.pct_change()`** without understanding the implications.

When calculating daily returns from price data:
```python
# WRONG - will drop any row with ANY NaN value in ANY column
daily_returns = close_prices.pct_change().dropna()

# CORRECT - only skip the first row (all NaN from pct_change)
daily_returns = close_prices.pct_change().iloc[1:]
```

**Why this matters:**
- Some tickers have shorter trading histories (e.g., SCHY started ~2021, TTD started ~2018, VIGI has gaps)
- These tickers will have NaN values for dates before they started trading
- `.dropna()` by default drops rows where ANY column has NaN (`how='any'` is default)
- This can **silently truncate your entire dataset** (e.g., from 10 years to 5 years)
- Keep the NaN values - they correctly represent dates before a ticker existed

**Real example from this project:**
- Dataset: 30 tickers over 10 years (2514 days: 2016-02-05 to 2026-02-04)
- 3 tickers (SCHY, TTD, VIGI) have limited history with NaN in early years
- Using `.dropna()`: Only 1197 rows remain (2021-04-30 to 2026-02-04) - **5 years lost!**
- Using `.iloc[1:]`: All 2513 rows preserved (2016-02-08 to 2026-02-04) - **full 10 years retained**

**If you must remove NaN:**
- Use `dropna(how='all')` to only drop rows where ALL columns are NaN
- Or use `dropna(subset=['AAPL', 'SPY'])` to only consider specific columns
- Or use `fillna(method='ffill')` to forward-fill missing values

