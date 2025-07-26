import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from quant_trading_system.risk.risk_manager import RiskManager


def test_calculate_position_size():
    rm = RiskManager(equity=10000)
    qty = rm.calculate_position_size(100, 90, risk_percentage=0.01)
    assert qty == 10
