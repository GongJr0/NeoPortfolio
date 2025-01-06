# NeoPortfolio
NeoPortfolio is a tool created to address the backwards-looking approach of the [Modern Portfolio Theory](https://en.wikipedia.org/wiki/Modern_portfolio_theory) (MPT).
The tool combines traditional MPT with Machine Learning (ML) models to predict stock returns and sentiment analysis to adjust return expectations based on media outlook.

## Why not just use MPT?
There are multiple shortcomings of MPT that swayed the analysts and investors to develop multiple alternative approaches.
The work of Harry Markowitz, although worthy of the Nobel Prize in Economics, is not widely used due to inconveniences
and assumptions that are not always true.

### Fully historical return expectations
MPT takes a fully historical approach to predicting the future and does it through simple averages. Therefore, the returns
calculated in the traditional MPT are not always accurate and can be heavily biased by the past. 

Predicting the future does rely heavily on decomposing and analysing past data, the question is how to do it better.
NeoPortfolio uses ML models to predict the future returns of stocks through lagged features. The process is designed to be
simple and time-efficient to allow iterative optimization of multiple portfolios. By nature, an ML model (`RandomForestRegressor` in this case)
attempts to deduce the underlying patterns in the rate of change of returns instead of simply calculating an average. Given
the model performance is satisfactory, (which is checked within the functions) the predictions will most likely be more
accurate than the historical averages.

In the case of lacking model performance, NeoPortfolio uses a fallback method of calculating the Exponentially Weighted Moving Average (EWMA) instead
of the historical average. EWMA reduces the impact of past data and significantly reduces the possibility of a biased return expectation caused by past events.

Above all, the return expectations are adjusted in accordance to the medias sentiment regarding the stock. FinBERT, a pre-trained Deep-Learning
NLP model, is used to calculate sentiment scores for articles retrieved from NewsAPI. This last step ensures that the expectations of the public
are incorporated into the expectations of the models.

### How would you optimize a non-existent portfolio with MPT? 
The main question NeoPortfolio aims to answer is not even addressed by MPT. MPT expects the investor to have a portfolio
in his mind before starting the process of optimizing the weights. NeoPortfolio, however, provides a tool to identify the
best portfolio (for a target return) to construct from the constituents of a market index. Using a combination engine and an
iterative optimization algorithm, NeoPortfolio can find the weights __and__ stocks to get the lowest possible risk for a given 
return.

## What NeoPortfolio doesn't fix
### Fluctuating Portfolio Weights
If you have some experience with MPT you'll know that weights of the same portfolio can vary extremely on a day-to-day basis.
The reason behind this is that MPT always returns the best possible portfolio. Naturally, a rational investor following
MPT is then expected to reconstruct their portfolio every day to maintain the optimal weights. This is almost never applicable
to a real-life scenario.

NeoPortfolio chooses not to address this in the spirit of always returning the best possible option to the investors.
The best choice based on mathematical processes is not always the best choice for each individual investor. Attempting to
assume the specific needs and conditions of each investor would certainly lead to outputs that are far from optimal. Therefore,
NopPortfolio sticks to the tried and true method of giving investors the mathematical solution and having them adjust the
weights to their needs.