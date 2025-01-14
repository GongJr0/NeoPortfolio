# `BtStrategy`
`BtStrategy` is a strategy container developed for use inside a `BtEngine` object. It defines trading strategies 
and outputs scores based on the input data, selected strategy, and configurable thresholds.

## Available Strategies

### Crossover
Crossover is a strategy where two moving averages of close prices compared. The Short Moving Average (SMA) is used to capture
recent trends in price changes while the Long Moving Average (LMA) records the long-term movement of the close price.

Crossover generates a buy or sell signal when a change in the price movement trend is detected. The trend comparison
is made by comparing the SMA and LMA of the current period to the previous one in a relational setting:

- __Buy Signal:__ $SMA_i \geq LMA_i \ \text{and } SMA_{i-1} \leq LMA_{i-1}$ (On periods where SMA crosses over LMA)
- __Sell Signal:__ $SMA_i \leq LMA_i \ \text{and } SMA_{i-1} \geq LMA_{i-1}$ (On periods where LMA crosses over SMA)

Hence, the strategy is named "Crossover" in reference to the relations it aims to capture. Signal magnitude is not 
measured in this strategy and all trades will be made with `BtEngine.max_trade_proportion` (0.33) of the asset traded.

### Relative Strength Index
Relative Strength Index (RSI) is a common strategy that, again, takes moving averages as its basis. However, RSI relies
on the average change of close prices instead of the absolute price. Relative Strength (RS) is calculated and scaled to 
the range $[0, 100]$ to get the RSI score. RS and its rescaling is done as follows:

$$RS=\frac{\text{Average Gains}}{\text{Average Losses}}$$
$$RSI = 100-\frac{100}{1+RS}$$

Where $\text{Average Gains}$ and $\text{Average Losses}$ are the moving averages of positive and negative changes in price
respectively. A buy signal is typically produced when $RSI \leq 30$ and a sell is often placed at $RSI \geq 70$.
`BtStrategy` offers __Linear, Exponential, and Logistic Scaling__ for RSI strategies with Logistic Scaling being the default
and best performing option. (per our testing)

### Fibonacci Retracement
...

### Ichimoku Cloud
...

## `__init__` Arguments
- `strat: Literal['crossover', 'rsi_ma', 'rsi_ewma', 'fib_retracement', 'ichimoku_cloud']`: strategy to use when running 
a backtest

## Attributes
- `strat -> str`: Strategy to use on backtesting
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