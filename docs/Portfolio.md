# `Portfolio`

`Portfolio` is an extension to the standard `tuple` class. The arguments passed
on instantiation (stock symbols) will be stored in a tuple and can be accessed 
using numerical indices. Additionally, using stock symbols as string indices
will return relevant information about the stock.

## `__init__` Arguments
- *args: Stock symbols to be added to the portfolio.

## Attributes
- `results`: A nested `dict` containing stock information.
    - __First level keys:__ 'weights', 'expected_returns', 'volatility', 'beta', 'sharpe_ratio', 'sentiment' 

    - __Second level keys:__ Stock symbols
  
-`optimum_portfolio_info`: A dictionary containing summary information regarding the optimized portfolio.
    - __Keys:__ 'target_return', 'target_volatility', 'weights', 'risk_per_return'

-`weights`: A dictionary of stock symbols and their respective weights in the portfolio.

-`tickers`: A `yfinance.Tickers` object containing initialized with stocks passed to `Portfolio`.
