from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import pandas as pd


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI)."""
    series = pd.Series(series)
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


class SmaCross(Strategy):
    n1 = 10
    n2 = 20

    def init(self):
        close = self.data.Close
        self.ma1 = self.I(SMA, close, self.n1)
        self.ma2 = self.I(SMA, close, self.n2)

    def next(self):
        if crossover(self.ma1, self.ma2):
            self.buy()
        elif crossover(self.ma2, self.ma1):
            self.sell()


class RSIStrategy(Strategy):
    rsi_period = 14
    overbought = 70
    oversold = 30

    def init(self):
        close = self.data.Close
        self.rsi = self.I(rsi, close, self.rsi_period)

    def next(self):
        if self.position:
            if self.rsi[-1] > self.overbought:
                self.position.close()
        else:
            if self.rsi[-1] < self.oversold:
                self.buy()


def run_sma_crossover(data: pd.DataFrame, short_window: int, long_window: int,
                      cash: float = 10000, commission: float = 0.002):
    class CustomSma(SmaCross):
        n1 = short_window
        n2 = long_window

    bt = Backtest(data, CustomSma, cash=cash, commission=commission)
    stats = bt.run()
    return bt, stats


def run_rsi_strategy(
    data: pd.DataFrame,
    rsi_period: int,
    overbought: int,
    oversold: int,
    cash: float = 10000,
    commission: float = 0.002,
):
    period = rsi_period
    ob = overbought
    os_ = oversold

    class CustomRSI(RSIStrategy):
        rsi_period = period
        overbought = ob
        oversold = os_

    bt = Backtest(data, CustomRSI, cash=cash, commission=commission)
    stats = bt.run()
    return bt, stats
