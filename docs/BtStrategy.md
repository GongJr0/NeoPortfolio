# `BtStrategy`
`BtStrategy` is a strategy container developed for use inside a `BtEngine` object. It defines trading strategies 
and outputs scores based on the input data, selected strategy, and configurable thresholds.

## `__init__` Arguments
- `strat: Literal['crossover', 'rsi_ma', 'rsi_ewma']`: strategy to use when running a backtest

## Attributes
- `strat -> str`: strategy to use on backtesting
- `signal_scaler -> Callable`: Signal magnitude mapper function to use depending on the selected strategy
- `buy_threshold -> int`: Buying signal threshold for RSI strategies
- `sell_threshold -> int`: Selling signal threshold for RSI strategies

## Methods
### `set_thresholds`
sets the thresholds given a single value using `buy_threshold = threshold; sell_threshold = (100-threshold)` to
keep a uniform range of signal generation space on both ends of RSI.

__Params:__
- `threshold: int`: threshold to determine buying and selling regions

__Returns:__
- None