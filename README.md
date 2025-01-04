# NeoPortfolio

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Modules](#modules)
    1. [`Portfolio` Class](#portfolio-class)
    2. [`Markowitz` Class](#markowitz-class)
        1. [`__init__` Parameters](#__init__-parameters)
        2. [Methods](#methods)
            1. [`optimize_return`](#optimize_return)
            2. [`optimize_volatility`](#optimize_volatility)
            3. [`efficient_frontier`](#efficient_frontier)
    4. [`nCrOptimize`](...)

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
## `nCrOptimize` Class
`nCrOptimize` is a combination based portfolio selection class that selects the portfolio with the lowest
volatility given a target return. Having an index specified as the market. The class compiles data on each
component of the index and creates a pool of candidate portfolios. Due to the nature of this optimization
iterating over all possible combinations is unfeasible and the number of possible combinations are controlled
through aggressive filtering formulas.

#### Filtering Methodology
Stocks are selected from two subsets being:
  - High Return
  - Low Volatility
 
The subsets are created by simply sorting the stocks by the respective metrics and applying a formula to determine
the ratio of stocks to select from each subset. The formula is constructed as follows:

- Assumptions:
   1. 0.7 is the average ratio of stocks to select from the high return subset.
   2. 5 is the component count of an average portfolio.

- Derived Terms:
  1. By extension of assumption 1, the ratio of low volatility stocks to select is derived to be $1-0.7=0.3$ on average.
  2. Risk-aversion is determined by the component count with the assumption that more components are expected from a safer investor.
  The factor that quantifies this is $\sigma_{tolerance} = \frac{n_{components} - 5}{5}$
- Formula:
  1. The ratio of high return stocks to select is given by:
     $r_{return} = \frac{0.7}{1+\sigma_{tolerance}}$
  
  2. The ratio of low volatility stocks to select is given by:
     $r_{volatility} = 1 - r_{return}$

Having the ratios, however, does not determine the total number of stocks to include in the combination
space. The total number $N$ is determined by the formula: $N = \min [N_{index},\\ (n_{components} \cdot \ln{N_{index})}],\\ \\ n \in [2, \infty)$

Finally, the combination space is created by retrieving the:
- Top $r_{return} \cdot N$ stocks from the high return subset.
- Top $r_{volatility} \cdot N$ stocks from the low volatility subset.

### `__init__` Parameters
- `market -> str`: The index to use as the market.
- `n -> int`: The number of components in the portfolio.
- `target_return -> float`: The target return for the portfolio.
- `horizon -> int`: The investment horizon in days.
- `lookback -> int`: The lookback period in days.
- `max_pool_size -> Optional[int]`: The maximum number of stocks to include in the combination pool.
- `api_key_path -> Optional[PathLike]`: The path to the .env file containing the NewsAPI key.
- `api_key_var -> Optional[str]`: The name of the variable in the .env file containing the NewsAPI key.

### Methods
- `optimize_space`: Optimize the combination space for the lowest volatility given a target return.

        Parameters:
        - bounds: A tuple representing the bounds for the weights.
        
        Returns:
        - `nCrReslut`: Wrapper object containing the results of the optimization.

## `nCrResult` Class
`nCrRsult` extends the `list` object and provides added functionality regarding the analysis and visualization
of the results of the `nCrOptimize` class. It is not intended as a user-facing class and is only returned by
`nCrOptimize.optimize_space`.

## Methods
- `max_return`: Get the maximum return of the portfolio.

        Parameters:
        - display: Prints a formatted HTML report if True.

        Returns:
        - dict: Restults of the highest return portfolio in the combination space.

- `min_volatility`: Get the minimum volatility of the portfolio.
    
            Parameters:
            - display: Prints a formatted HTML report if True.
    
            Returns:
            - dict: Restults of the lowest volatility portfolio in the combination space.

- `best_portfolio`: Get the best portfolio in the combination space.

        Parameters:
        - display: Prints a formatted HTML report if True.

        Returns:
        - dict: Restults of the best (max(p['return']/p['portfolio_variance'])) portfolio in the combination space.