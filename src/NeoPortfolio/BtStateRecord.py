from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Generator
import pandas as pd


@dataclass
class State:
    """Represents the state of the backtest at a point in time"""
    iteration: int
    cash: float
    holdings: Dict[str, float]
    signals: Dict[str, tuple[float, float]]  # (signal, magnitude)
    portfolio_value: float  # Total portfolio value including cash and holdings


class BacktestRecorder:
    def __init__(self):
        self.states: List[State] = []

    def record(self, iteration: int, cash: float, holdings: dict, signals: dict,
               current_prices: pd.Series) -> None:
        # Calculate total portfolio value
        holdings_value = sum(
            holdings[stock] * current_prices[stock]
            for stock in holdings
        )
        portfolio_value = cash + holdings_value

        state = State(
            iteration=iteration,
            cash=cash,
            holdings=holdings.copy(),
            signals=signals.copy(),
            portfolio_value=portfolio_value
        )
        self.states.append(state)

    def get_history(self) -> pd.DataFrame:
        """Convert recorded states to a DataFrame for analysis"""
        records = []
        for state in self.states:
            record = {
                'iteration': state.iteration,
                'cash': state.cash,
                'portfolio_value': state.portfolio_value,
                **{f'holdings_{k}': v for k, v in state.holdings.items()},
                **{f'signal_{k}': v[0] for k, v in state.signals.items()},
                **{f'magnitude_{k}': v[1] for k, v in state.signals.items()}
            }
            records.append(record)
        df = pd.DataFrame(records)
        df.set_index('iteration', inplace=True)
        df['profit'] = df['portfolio_value'].diff()
        return df


@contextmanager
def record_state() -> Generator[BacktestRecorder, None, None]:
    recorder = BacktestRecorder()
    try:
        yield recorder
    finally:
        pass
