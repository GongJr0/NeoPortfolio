from typing import NewType, Literal, Tuple, Union
from IPython.display import HTML
import pandas as pd

StockSymbol = str
StockDataSubset = Tuple[Literal['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]

Days = int


class nCrResult(list):
    """
    Class to store the result of a nCrEngine calculation.
    """
    def __init__(self, *args):
        super().__init__(*args)

    @staticmethod
    def beautify_portfolio(portfolio_dict):
        portfolio = portfolio_dict['portfolio'].split(' - ')
        weights = portfolio_dict['weights']
        expected_returns = portfolio_dict['expected_returns']
        cov_matrix = portfolio_dict['cov_matrix']
        betas = portfolio_dict['betas']

        # Create a DataFrame for the portfolio and weights
        portfolio_df = pd.DataFrame({
            "Weight": weights,
            "Expected Return": expected_returns,
            "Beta": betas
        }, index=portfolio)

        # Generate HTML for the table
        portfolio_html = portfolio_df.to_html(float_format="%.4f")
        cov = pd.DataFrame(cov_matrix, index=portfolio, columns=portfolio).to_html(float_format="%.4f")

        # Create additional HTML content for metrics
        summary_html = f"""
            <h2>Portfolio Analysis</h2>
            <h3>Summary Metrics</h3>
            <ul>
                <li><strong>Expected Portfolio Return:</strong> {portfolio_dict['return'] * 100:.4f}%</li>
                <li><strong>Portfolio Variance:</strong> {portfolio_dict['portfolio_variance'] * 100:.4f}%</li>
            </ul>
            <h3>Portfolio Composition</h3>
            {portfolio_html}
            
            <h3>Covariance Matrix</h3>
            {cov}
            """

        # Combine all HTML content
        full_html = summary_html

        return HTML(full_html)

    def _best_portfolio(self) -> HTML | dict:
        return max(
            self,
            key=lambda x: x['return'] / x['portfolio_variance']
        )

    def _max_return(self, display: bool = False) -> dict:
        if display:
            return self.beautify_portfolio(self.max_return())
        return max(
            self,
            key=lambda x: x['return']
        )

    def _min_volatility(self) -> dict:
        return min(
            self,
            key=lambda x: x['portfolio_variance']
        )

    def max_return(self, display: bool = False) -> dict | HTML:
        """
        Get the maximum return from the result.
        """
        if display:
            return self.beautify_portfolio(self._max_return())
        return self._max_return()

    def min_volatility(self, display: bool = False) -> dict | HTML:
        """
        Get the minimum volatility from the result.
        """
        if display:
            return self.beautify_portfolio(self._min_volatility())
        return self._min_volatility()

    def best_portfolio(self, display: bool = False) -> dict | HTML:
        """
        Get the best portfolio from the result.
        """
        if display:
            return self.beautify_portfolio(self._best_portfolio())
        return self._best_portfolio()