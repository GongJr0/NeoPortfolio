# `BtEngine`
`BtEngine` is a backtesting module designed to accept close prices outputted by `yfinance.Ticker` 
or `yfinance.Tickers`. It employs a strategy stored in `BtStrategy` and iterates over the historical
to test the performance of a portfolio when actively trading.

## `__init__` Arguments

- `portfolio_close -> pd.DataFrame`: Close prices returned from `yfinance`
- `strategy -> BtStrategy`: BtStrategy object to handle signal generation
- `hi_lo -> tuple[pandas.DataFrame, pandas.DataFrame]`: Tuple of high and low price data from `yfinance`
- `vol -> pandas.DataFrame`: Volume data from `yfinance`

## Attributes

- `cash -> float`: Liquid assets (initial balance = 100,000)
- `max_trade_proportion -> float`: Maximum trade amount as a percentage of total holdings (default = 0.33)
- `price_data -> pandas.DataFrame`: Close prices from `yfinance`
- `dt_index: pandas.DateIndex`: Default index of the close price data
- `hi -> pandas.DataFrame`: High prices from `yfinance`
- `lo -> pandas.DataFrame`: Low prices from `yfinance`
- `vol -> pandas.DataFrame`: Volume data from `yfinance`
- `sma_period -> int`: Window for Short-Moving-Average (SMA)
- `lma_period -> int`: Window for Long-Moving-Average (LMA)
- `strat: BtStrategy`: Strategy container for signal generation
- `holdings: dict[str, float]`: Holdings for each stock (in shares)
- `buy -> dict[str, list[tuple[str, float]]]`: dict of each period (keys) where a buy signal was produced and a list tuples with format (stock_that_produced_buy_sig, strat_score)
- `sell -> dict[str, list[tuple[str, float]]]`: dict of each period (keys) where a sell signal was produced and a list tuples with format (stock_that_produced_buy_sig, strat_score)
- `total_buys -> int`: Count of total buy signals
- `total_sells -> int`: Count of total sell signals

## Methods

### `run`
Run the engine to perform iterative backtesting with the pre-determined strategy

__Params:__
- None

__Returns:__

- `dict[str, float | dict]`: dict containing post-run portfolio value in USD, total liquid assets,
and asset distribution as a dict of stocks and their respective holdings at the end of the run.

### `plot_history`
Display a `matplotlib.pyplot` figure that contains key information regarding the performance of the selected
backtesting strategy and portfolio.

__Params:__
- None

__Returns:__
- None