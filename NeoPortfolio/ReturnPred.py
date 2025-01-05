# ML imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV

# Data imports
import pandas as pd
import numpy as np

# Typing imports
from typing import Union
from .CustomTypes import Days


class ReturnPred:
    """Create expected return predictions from historical prices"""

    def __init__(self, data: pd.DataFrame, inv_horizon: Days = 21):
        self.model = RandomForestRegressor()
        self.data = data
        self.inv_horizon = inv_horizon

    def split_stocks(self):
        """Split a DataFrame of multiple stocks into individual DataFrame"""
        out = []
        for col in self.data.columns:
            out.append(self.data[col])
        return out

    def add_lagged_features(self, stock_data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Create lagged features for a stock's historical data and set the target as the future price.

        Parameters:
            stock_data (Union[pd.DataFrame, pd.Series]): A time-series of stock prices or returns.

        Returns:
            pd.DataFrame: A DataFrame with lagged features and the original target column (future price).
        """
        if isinstance(stock_data, pd.Series):
            stock_data = stock_data.to_frame(name="future_price")

        assert stock_data.shape[1] == 1, "stock_data must have exactly one column"

        # Calculate lag component based on inv_horizon
        lag_component = max(4, int(np.floor(np.sqrt(self.inv_horizon))))

        feature_set = pd.DataFrame(index=stock_data.index)

        # Create lagged features
        for i in range(1, lag_component + 1):  # Include lag_component in the loop
            feature_set[f"lag_{i}"] = stock_data.shift(i).values.ravel()

        # Calculate the future price at i+inv_horizon
        future_price = stock_data.shift(-self.inv_horizon)

        # Combine features and target (future price), dropping rows with NaNs due to lagging
        data = pd.concat([feature_set, future_price], axis=1)

        today_data = data.iloc[-1] # Last days data for return prediction (future_price is NaN)
        today_data = today_data.to_frame().dropna().T

        data = data.dropna(how="any")
        return data, today_data


    def train(self, stock_data: Union[pd.DataFrame, pd.Series], comb: bool = False) -> dict:
        """Train the model on a stock's historical returns. Hyperparameters are not tuned in order to keep the runtime
        low. In case of unacceptable accuracy, the fallback method is an historical EWMA with a span equal to the
        investment horizon (in days). This is an iteration on the traditional return calculation of using the mean return
        over the data period.

        :param stock_data: A time-series of stock prices or returns.
        :param comb: Boolean to indicate if the portfolio is part of a combination space from `nCrEngine`.
        """

        success = False
        confidence = 0
        data, today_data = self.add_lagged_features(stock_data)

        X = data.drop(columns=data.columns[-1])
        y = data[data.columns[-1]]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Simple Grid-Search
        if not comb:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'max_features': ['sqrt', 'log2'],
                'bootstrap': [True]
            }
            gscv = GridSearchCV(self.model, param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
            gscv.fit(X_train, y_train)

            self.model = gscv.best_estimator_
        elif comb:
            self.model.fit(X_train, y_train)

        pred = self.model.predict(X_test)
        target_error = np.var(y_test)

        if mean_squared_error(y_test, pred) < target_error:
            success = True

        confidence = np.round(1 - mean_absolute_percentage_error(y_test, pred), 4)

        # Return prediction for the period r_{t:t+inv_horizon}
        expected_price = np.round(self.model.predict(today_data)[0], 2)
        expected_return = np.round((expected_price - stock_data.iloc[-1]) / stock_data.iloc[-1], 4)

        if not success:
            period_returns = (stock_data - stock_data.shift(self.inv_horizon)) / stock_data.shift(self.inv_horizon)

            span = np.round(np.sqrt(self.inv_horizon), 0) * 2
            expected_return = np.round(period_returns.ewm(span=span).mean().iloc[-1], 4)
            expected_price = np.round(stock_data.iloc[-1] * (1 + expected_return), 2)

        return {'success': success,
                'confidence': confidence,
                'expected_price': expected_price,
                'expected_return': expected_return}


    def all_stocks_pred(self, comb: bool = False) -> dict:
        """Return predictions for all stocks in the data"""
        stocks = self.split_stocks()
        return {stock.name: self.train(stock, comb) for stock in stocks}