# `nCrOptimize`
`nCrOptimize` is a class that performs optimization on a search space of nCr combinations of stocks from a market index.
It employs aggressive elimination processes to reduce the members of the combination space to keep compute times reasonable.
`nCrOptimize` is programmed to cache results on each iteration to retain information from interrupted runs. `nCrOptimize` 
relies on `nCrEngine`, `ReturnPred`, `Sentiment`, and `Cache` modules on the backend. 
None of the listed modules are user-facing.

## `__init__ Arguments`
- `market -> IndexSymbol`: Market index symbol.
- `n -> int`: Number of stocks in the portfolio.
- `target_return -> float`: Target return for the portfolio.
- `horizon -> Days`: Investment horizon in days.
- `lookback -> Days`: Historical data lookback period in days.
- `max_pool_size -> Optional[int]`: Maximum number of combinations to consider.
- `api_key_path -> Optional[str | os.PathLike]`: Path to the .env file containing a NewsAPI key.
- `api_key_var -> Optional[str]`: Name of the environment variable containing the NewsAPI key.

## Attributes
- `api_key_path -> Optional[str | os.PathLike]`: Path to the .env file containing a NewsAPI key.
- `key_var -> Optional[str]`: Name of the environment variable containing the NewsAPI key.
- `portfolio_cache -> PortfolioCache`: `Cache.PortfolioCache` object.
- `portfolios -> list[Portfolio]`: List of `Portfolio` objects created from combinations.
- `market_returns -> pandas.Series`: Market returns.
- `rf_rate -> float`: Risk-free rate of return for the investment period. (not per annum)

## Methods
### `optimize_space`
Optimize the search space of nCr combinations.

__Parameters:__
- `bounds: tuple[float, float]`: Tuple containing the lower and upper bounds for the weights.

__Returns:__
- `nCrResult`: An object containing the results of the optimization.