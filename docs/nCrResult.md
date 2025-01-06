# `nCrResult`
`nCrResult` extends the standard `list` class and contains the results of the optimization performed by `nCrOptimize`
for each combination of stocks. It provides methods to explore the optimized combination space and automatically create
HTML reports for the best portfolios, outputted by `Ipython.display.HTML` and `Ipython.display.display`.

## Methods
### `best_portfolio`
Display the best portfolio (determined by $score = \frac{\mu_{expected}}{\sigma^2}$)

__Parameters:__
- `display: bool`: Create an HTML report and display it if True. Else return the results as a dictionary.

__Returns:__
- `Optional[dict]`: A dictionary containing the best portfolio's information.

### `max_return`
Display the portfolio with the highest expected return.

__Parameters:__
- `display: bool`: Create an HTML report and display it if True. Else return the results as a dictionary.

__Returns:__
- `Optional[dict]`: A dictionary containing the portfolio with the highest expected return's information.

### `min_volatility`
Display the portfolio with the lowest volatility.

__Parameters:__
- `display: bool`: Create an HTML report and display it if True. Else return the results as a dictionary.

__Returns:__
- `Optional[dict]`: A dictionary containing the portfolio with the lowest volatility's information.
