# `Markowitz`
`Markowitz` perform optimization to the weights of a known portfolio through Modern Portfolio Theory (MPT). 
Added to the traditional MPT are Machine Learning models to predict stock returns and sentiment analysis adjust return
expectations based on media outlook. `Markowitz` relies on `ReturnPred`, `Sentiment`, and `Cache` modules on the backend.
None of the listed modules are user-facing.

## `__init__ Arguments`
- `portfolio -> Portfolio`: A `Portfolio` object containing stock symbols.
- `market -> IndexSymbol`: Market index symbol.
- `horizon -> Days`: Investment horizon in days.
- `lookback -> Days`: Historical data lookback period in days.
- `rf_rate_pa -> Optional[float]`: Risk-free rate of return per annum. (10 Year US Treasury yield if None)
- `api_key_path -> Optional[str | os.PathLike]`: Path to the .env file containing a NewsAPI key.
- `api_key_var -> Optional[str]`: Name of the environment variable containing the NewsAPI key.

## Attributes
- `portfolio -> Portfolio`: A `Portfolio` object containing stock symbols.
- `names -> dict[StockSymbol, str]`: Short names of the companies issuing the stocks.
- `market -> yfinance.Ticker`: Market index.
- `market_name -> str`: Name of the market index. (often times the symbol)
- `rf -> float`: Risk-free rate of return for the investment period. (not per annum)
- `raw_close -> pandas.DataFrame`: Historical closing prices of the stocks.
- `periodic_return -> pandas.DataFrame`: Periodic returns of the stocks.
- `expected_returns -> pandas.DataFrame`: Expected returns of the stocks.
- `volatility -> pandas.Series`: Volatility of the stocks.
- `market_returns -> pandas.Series`: Market returns.
- `market_volatility -> float`: Market volatility.
- `rm -> float`: Market expected return. (historical average)
- `cov_matrix -> pandas.DataFrame`: Covariance matrix of the stocks.
- `beta -> list[float]`: Beta values of the stocks.

## Methods
### `optimize_return` 
Optimize the portfolio for a target return.

__Parameters:__

- `target_return: float`: Target return for the portfolio.
- `additional_constraints: Optioonal[list]`: Additional constraints formatted to be passed to `scipy.optimize.minimize`.
- `bounds: tuple[float, float]`: Tuple containing the lower and upper bounds for the weights.
- `include_beta: bool`: Include beta values in the optimization.
- `record: bool`: Record the optimization results in the Portfolio object.

__Returns:__

- `tuple[dict, scipy.optimize.OptimizeResult]`: A tuple containing the optimized weights and the optimization result.

### `optimize_volatility`
Optimize the portfolio for a target volatility.

__Parameters:__

- `target_volatility: float`: Target volatility for the portfolio.
- `additional_constraints: Optioonal[list]`: Additional constraints formatted to be passed to `scipy.optimize.minimize`.
- `bounds: tuple[float, float]`: Tuple containing the lower and upper bounds for the weights.
- `include_beta: bool`: Include beta values in the optimization.
- `record: bool`: Record the optimization results in the Portfolio object.

__Returns:__

- `tuple[dict, scipy.optimize.OptimizeResult]`: A tuple containing the optimized weights and the optimization result.

### `efficient_frontier`
Plot the efficient frontier.

__Parameters:__
- `target_input: Literal['return', 'volatility']`: The target input for the efficient frontier.
- `n: int`: Number of points to plot on the efficient frontier.
- `save: bool`: Save the plot as a .png file.

__Returns:__
- `None`