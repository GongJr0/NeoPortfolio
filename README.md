# NeoPortfolio

## Introduction

This project aims to bring stock selection and portfolio optimization together while 
implementing modern features such as automated sentiment analysis and ML based stock return
prediction. The project is not final and this README will be updates as changes are 
introduced and more modules are added.

## Installation
You can start using the `NeoPortfolio` package after running the following command in your 
desired environment.

```bash
python -m pip install NeoPortfolio
```
#### PyTorch
If the `pip install` does not work, it is likely due to an incompatibility with the PyTorch
version pip attempts to install. In this case, you can install PyTorch manually by following
[PyTorch Installation Guide](https://pytorch.org/get-started/locally/) with the __*compute platform*__
set to __CPU__.
   
## Quick Start
The main goal of this project is to eliminate the step-by-step approach to portfolio
optimization and stock selection. In that spirit, methods and classes users need to know
are kept to a minimum.

The combination engine to select stock on the users behalf is currently in development,
therefore the first step is to create a portfolio of $n$ stocks.

```python
from NeoPortfolio import Portfolio
from NeoPortfolio import Markowitz

# Create a portfolio of 5 stocks
portfolio = Portfolio('AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA')

# Define the investment horizon and lookback period in days
horizon = 21  # 1 trading month
lookback = 252  # 1 trading year

# Define the index to use as the market, and the risk-free rate
market = '^GSPC'  # S&P 500
rf_rate = 0.5717  # 10 year treasury yield (USA, per annum)

# Create a Markowitz object
markowitz = Markowitz(portfolio,
                      market=market,
                      horizon=horizon,
                      lookback=lookback,
                      rf_rate_pa=rf_rate,
                      api_key_path='path/to/newsapi/key.env',
                      api_key_var='YOUR_VARNAME')

# Define optimization parameters for a target return of 10%
# Use optimize_volatility to pass a target volatility instead

target_return = 0.1
bounds = (0.05, 0.7)  # Set the bounds for the weights
with_beta = True  # Include beta in the optimization
additional_constraints = []  # Add additional constraints as a list if needed
# use scipy.optimize constraint format

# Run the optimization
weights, opt = markowitz.optimize_return(target_return,
                                         bounds=bounds,
                                         include_beta=with_beta,
                                         additional_constraints=additional_constraints,
                                         record=True)  # record the results in the portfolio object

# Print the results
print(f'Optimal weights:\n{weights}')
```

## Modules
As of now, the only user-facing modules are the `Portfolio` and `Markowitz` 
classes. The `Portfolio` class holds stock symbols and relevant information
regarding the optimization process. The `Markowitz` class is used to optimize
the portfolio weights and plot the efficient frontier. One important note is that
an instance of `Porfolio` will not populate the stock information, data, or 
statistics. The object must be passed to the `Markowitz` class, at which point
the information will be retrieved and necessary calculations will be made on
`Markowitz.__init__` without any additional input from the user.

## `Portfolio` Class
`Portfolio` is an extension to the standard `tuple` class. The arguments passed
on instantiation (stock symbols) will be stored in a tuple and can be accessed 
using numerical indices. Additionally, using stock symbols as string indices
will return relevant information about the stock.

#### Attributes
- `results`: A dictionary of dictionaries containing stock information.

        The first level of keys are metrics: 
        - ['weights', 'expected_returns', 'volatility', 'beta', 'sharpe_ratio', 'sentiment']

        The second level of keys are stock symbols as strings.
- `optimum_portfolio_info`: A dictionary containing summary information regarding the optimized portfolio.

        The keys are:
        - ['target_return', 'target_volatiltiy', `weights`, `risk_per_return`]

- `weights`: A dictionary of stock symbols and their respective weights in the portfolio.
- `tickers`: A `yfinance.Tickers` object containing initialized with stocks passed to `Portfolio`.
    
## `Markowitz` Class
`Markowitz` is the main class used to optimize the portfolio weights 
and plot the efficient frontier.

### `__init__` Parameters
- `portfolio`: A `Portfolio` object containing stock symbols.
- `market`: A string representing the index to use as the market.
- `horizon`: An integer representing the investment horizon in days.
- `lookback`: An integer representing the lookback period in days.
- `rf_rate_pa`: A float representing the risk-free rate per annum.

### Methods
- `optimize_return`: Optimize the portfolio weights for a target return.

        Parameters:
        - target_return: A float representing the target return.
        - bounds: A tuple representing the bounds for the weights.
        - with_beta: A bool representing whether to include beta in the optimization.
        - additional_constraints: A list of additional constraints to pass to the optimizer.
        - record: A bool representing whether to record the results in the portfolio object.

        Returns:
        - `weights`: A `dict` of stock symbols and their respective weights in the portfolio.
        - `opt`: A `scipy.optimize.OptimizeResult` object containing the optimization results.

<br></br>

- `optimize_volatility`: Optimize the portfolio weights for a target volatility.

        Parameters:
        - target_volatility: A float representing the target volatility.
        - bounds: A tuple representing the bounds for the weights.
        - include_beta: A bool representing whether to include beta in the optimization.
        - additional_constraints: A list of additional constraints to pass to the optimizer.
        - record: A bool representing whether to record the results in the portfolio object.

        Returns:
        - `weights`: A `dict` of stock symbols and their respective weights in the portfolio.
        - `opt`: A `scipy.optimize.OptimizeResult` object containing the optimization results.

<br></br>
- `efficient_frontier`: Plot the efficient frontier of a portfolio.

        Parameters:
        - target_input: A string literal ['return', 'volatility'] representing the target to optimize for.
        - n: An integer representing the number of points to plot.
        - save: A bool representing whether to save the plot as a .png file.

        Returns:
        - None

