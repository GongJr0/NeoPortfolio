import numpy as np
from IPython.display import HTML, display
import pandas as pd
import plotly.express as px

class nCrResult(list):
    """
    Class to store the result of a nCrEngine calculation.
    """

    def __init__(self, *args, rf_rate: float):
        super().__init__(*args)
        self.horizon_rf_rate = rf_rate


    def _beautify_portfolio(self, portfolio_dict):
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

        # Create plotly scatter of all portfolios
        all_portfolios = {
            'Portfolio': [x['portfolio'] for x in self],
            'Return': [x['return'] for x in self],
            'Variance': [x['portfolio_variance'] for x in self]
        }

        df = pd.DataFrame(all_portfolios)
        df.set_index('Portfolio', inplace=True)

        df['Sharpe Ratio'] = (df['Return'] - self.horizon_rf_rate) / np.sqrt(df['Variance'])

        max_return = df.loc[df['Return'].idxmax(), ['Return', 'Variance', 'Sharpe Ratio']]
        min_volatility = df.loc[df['Variance'].idxmin(), ['Return', 'Variance', 'Sharpe Ratio']]
        best_portfolio = df.loc[(df['Return'] / df['Variance']).idxmax(), ['Return', 'Variance', 'Sharpe Ratio']]

        fig = px.scatter(df,
                         x='Variance',
                         y='Return',
                         color='Sharpe Ratio',
                         color_continuous_scale='Viridis',
                         title='Optimized Portfolios')

        # Add traces for each key portfolio with proper colors and symbols
        fig.add_trace(
            px.scatter(
                x=[max_return['Variance']], y=[max_return['Return']],
                size=[10], symbol=[f'Max Return: {max_return.name}'], labels={'x': 'Variance', 'y': 'Return'},
            ).data[0]
        )
        fig.data[-1].marker.color = 'red'
        fig.data[-1].marker.size = 10

        fig.add_trace(
            px.scatter(
                x=[min_volatility['Variance']], y=[min_volatility['Return']],
                size=[10], symbol=[f'Min Volatility: {min_volatility.name}'], labels={'x': 'Variance', 'y': 'Return'},
            ).data[0]
        )
        fig.data[-1].marker.color = 'green'
        fig.data[-1].marker.size = 10

        fig.add_trace(
            px.scatter(
                x=[best_portfolio['Variance']], y=[best_portfolio['Return']],
                size=[10], symbol=[f'Best Portfolio: {best_portfolio.name}'], labels={'x': 'Variance', 'y': 'Return'},
            ).data[0]
        )
        fig.data[-1].marker.color = 'violet'
        fig.data[-1].marker.size = 10

        # Update layout to improve legend placement and color scale
        fig.update_layout(
            xaxis_title='Variance',
            yaxis_title='Return',
            title='Optimized Portfolios',
            coloraxis_colorbar=dict(
                title='Sharpe Ratio',
                tickvals=[min(df['Sharpe Ratio']), max(df['Sharpe Ratio'])],
                ticktext=[f'{min(df["Sharpe Ratio"]):.2f}', f'{max(df["Sharpe Ratio"]):.2f}'],
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.1,
                xanchor="right",
                x=0.5
            )
        )

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
            
            <h3>Optimized Combination Space</h3>
            """

        display(HTML(summary_html))
        fig.show()

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
            return self._beautify_portfolio(self._max_return())
        return self._max_return()

    def min_volatility(self, display: bool = False) -> dict | HTML:
        """
        Get the minimum volatility from the result.
        """
        if display:
            return self._beautify_portfolio(self._min_volatility())
        return self._min_volatility()

    def best_portfolio(self, display: bool = False) -> dict | HTML:
        """
        Get the best portfolio from the result.
        """
        if display:
            return self._beautify_portfolio(self._best_portfolio())
        return self._best_portfolio()