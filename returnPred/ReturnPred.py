# ML imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Data imports
import pandas as pd
import numpy as np

# Typing imports
from typing import Union

class ReturnPred:
    """Create expected return predictions from historical prices"""

    def __init__(self, data: pd.DataFrame, inv_horizon: int = 21):
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


    def train(self, stock_data: Union[pd.DataFrame, pd.Series]) -> dict:
        """Train the model on a stock's historical returns. Hyperparameters are not tuned in order to keep the runtime
        low. In case of unacceptable accuracy, the fallback method is an historical EWMA with a span equal to the
        investment horizon (in days). This is an iteration on the traditional return calculation of using the mean return
        over the data period."""

        success = False
        confidence = 0
        data, last_pred = self.add_lagged_features(stock_data)

        X = data.drop(columns=data.columns[-1])
        y = data[data.columns[-1]]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        self.model.fit(X_train, y_train)
        pred = self.model.predict(X_test)
        target_error = np.var(y_test)

        if mean_squared_error(y_test, pred) < target_error:
            success = True


        pred = self.model.predict(X_test)
        confidence = (1 - mean_absolute_percentage_error(y_test, pred)).round(4)

        #pred last day
        expected_price = self.model.predict(last_pred)[0].round(2)
        expected_return = ((expected_price - stock_data.iloc[-1]) / stock_data.iloc[-1]).round(4)

        # if not success:
        #     period_returns = (stock_data - stock_data.shift(self.inv_horizon)) / stock_data.shift(self.inv_horizon)
        #     expected_return = period_returns.ewm(span=self.inv_horizon).mean().iloc[-1].round(4)
        #     expected_price = (stock_data.iloc[-1] * (1 + expected_return)).round(2)

        return {'success': success,
                'confidence': confidence,
                'expected_price': expected_price,
                'expected_return': expected_return}


    def all_stocks_pred(self) -> dict:
        """Return predictions for all stocks in the data"""
        stocks = self.split_stocks()
        return {stock.name: self.train(stock) for stock in stocks}