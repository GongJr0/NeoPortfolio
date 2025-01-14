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
Fibonacci Retracement is a strategy relying on drawing brackets between the high and low prices from a period
brackets are defined as percentiles in the $[Lo_{i-t:i}, Hi_{i-t:i}]$ range and the levels are drawn from the Fibonacci
sequence. ($FibLevels = \{0.236, 0.382, 0.618, 0.786\}$) Signals are generated as follows:

$$\mathrm{FibSignal}(P_i)=
\begin{cases}
Buy & \text{if} \ P_i \leq P_{38.2%} \\
Hold & \text{if} \ P_i \in (P_{38.2%}, \ P_{61.8%}) \\
Sell & \text{if} \ P_i \geq P_{61.8%}
\end{cases]
$$

Similar to RSI, signal magnitude is achieved by mapping calculated Fib percentile of the current price to range $[0, 1]$
using on of the __Linear, Exponential, and Logistic Scaling__ functions implemented. Again, the default and best
performing function is determined to be the Logistic Scaling. (per our testing)

### Ichimoku Cloud
Ichimoku Cloud is a strategy built on the moving averages of the range between high and low prices. Having the base 
formula:

$$\mathrm{Span}(t) = \frac{Hi_{i-t: t} - Lo_{i-t: t}}{2}$$

Multiple moving averages are calculated with differing inputs as $t$.

- __Tenkan-sen ($=\mathrm{Span}(9)$):__ Captures the average range in the shortest rolling window
- __Kijun-sen ($=\mathrm{Span}(26)$):__ Captures the average range in the mid-range rolling window
- __Senkou Span B ($=\mathrm{Line}(52)$):__ Captures the average range in the longest rolling window
- __Senkou Span A ($\frac{\text{Tenkan-sen} + \text{Kijun-sen}}{2}$):__ Captures the range between short and medium 
rolling spans

Signals are generated as follows:

$$\mathrm{Ichimoku} =
\begin{cases}
Buy & \text{if} \ \text{Tenkan-sen > Kijun-sen} \text{ and} P_i > \max (Senkou_A, \ Senkou_B)
Sell & \text{if} \ \text{Tenkan-sen < Kijun-sen} \text{ and} P_i < \min (Senkou_A, \ Senkou_B)
Hold & \text{if} \ \text{Any other outcome}
\end{cases]
$$

Ichimoku cloud signals do not generate scores with known bounds or midpoints, making signal scaling
difficult. The approach taken in generating a signal magnitude for this strategy relies on clipping the final
output to the range $[0, 1]$ with the assumption that magnitudes over 1 indicate a significantly strong signal
and magnitudes less than 0 similarly indicate a significantly weak signal.

Scaling is handled by the following functions:

$$\mathrm{CloudBoundary} = 
\begin{cases}
Senkou_A & \text{if } \ Signal = 1 \\
Senkou_B & \text{if } \ Signal = -1
\end{cases}$$

$$\text{Raw Magnitude} = \frac{|P_i - CloudBoundary|}{Hi_i - Lo_i}$$

$$Magnitude = 
\begin{cases}
1 & \text{if } \ \text{Raw Magnitude} \geq 1 \\
\text{Raw Magnitude} & \text{if } \ \text{Raw Magnitude} \in (0, 1) \\
0 & \text{if } \ \text{Raw Magnitude} \leq 0
\end{cases}$$

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