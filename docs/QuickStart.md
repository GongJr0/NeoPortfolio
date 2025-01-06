# Installation
NeoPortfolio can be installed through pip with the following command:
```bash
python -m pip install NeoPortfolio
```
You can visit the [GitHub repository](https://github.com/GongJr0/NeoPortfolio) or 
[PyPI page](https://pypi.org/project/NeoPortfolio/#files) to access the source code.


# Quick Start
NeoPortfolio offers optimization tools for pre-determined portfolios a search tool to find the
optimal portfolio from an index's constituents.
## MPT Optimization for Pre-Determined Portfolio
```python
from NeoPortfolio import Portfolio, Markowitz

# Create a portfolio object and pass it to the Markowitz class
portfolio = Portfolio(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'])
markowitz = Markowitz(
    portfolio=portfolio,
    market='^GSPC',  # S&P 500
    horizon= 21,
    lookback=252,
    rf_rate_pa=None, # Use 10-Year US Treasury yield if None
    api_key_path='path/to/api_key.env', # NewsAPI key
    api_key_var='YOUR_VAR_NAME'
)

# Run the optimization
weights, opt = markowitz.optimize_return(target_return=0.05,
                                         bounds=(0.05, 0.70),
                                         include_beta=True)
print(weights)

# Plot the efficient frontier
markowitz.efficient_frontier('return', n=500)
```

## Automatic Portfolio Selection
```python
from NeoPortfolio import nCrOptimize

ncr = nCrOptimize(
    market='^GSPC',  # S&P 500
    n=5,  # Number of stocks in the portfolio
    target_return=0.05,
    horizon=21,
    lookback=252,
    api_key_path='path/to/api_key.env', # NewsAPI key
    api_key_var='YOUR_VAR_NAME'
)

# Run the optimization
results = ncr.optimize_space(bounds=(0.05, 0.70))

results.best_portfolio(display=True)
```