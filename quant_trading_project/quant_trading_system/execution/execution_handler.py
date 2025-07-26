# quant_trading_system/execution/execution_handler.py

import os
import sys

# --- Path Fix ---
# Add the project root to the Python path
# This is necessary to ensure that the script can find the other modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ----------------

# Import the new alpaca-py library components
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Import the RiskManager using an absolute path from the project root
from quant_trading_system.risk.risk_manager import RiskManager

class ExecutionHandler:
    """
    A module for handling the connection to a brokerage and executing trades.

    This class is now UPGRADED to use the modern `alpaca-py` library,
    resolving dependency conflicts and providing a more robust interface.
    """

    def __init__(self, paper_trading=True):
        """
        Initializes the ExecutionHandler using alpaca-py.
        """
        # Import config here to avoid circular imports
        from quant_trading_system.utils.config import config
        
        self.api_key = os.getenv('APCA_API_KEY_ID', config.APCA_API_KEY_ID)
        self.secret_key = os.getenv('APCA_API_SECRET_KEY', config.APCA_API_SECRET_KEY)
        self.paper_trading = paper_trading

        if not self.api_key or not self.secret_key or self.api_key == 'YOUR_KEY_ID_HERE':
            raise ValueError("API keys not found. Please set APCA_API_KEY_ID and APCA_API_SECRET_KEY environment variables or update config.py.")

        print("Connecting to Alpaca via modern `alpaca-py` library...")
        try:
            # The new TradingClient handles paper/live trading automatically
            self.client = TradingClient(self.api_key, self.secret_key, paper=self.paper_trading)
            self.account = self.client.get_account()
            print(f"Successfully connected. Account status: {self.account.status}")
        except Exception as e:
            print(f"Failed to connect to Alpaca: {e}")
            self.client = None
            self.account = None

    def get_account_info(self):
        """
        Retrieves and prints key information about the trading account.
        """
        if self.account:
            print(f"Account Number: {self.account.account_number}")
            print(f"Equity: ${self.account.equity}")
            print(f"Buying Power: ${self.account.buying_power}")
        else:
            print("Not connected to an account.")
        return self.account

    def place_bracket_order(self, symbol, side, entry_price, risk_manager, risk_percentage, take_profit_price, stop_loss_price):
        """
        Places a risk-managed bracket order using the new alpaca-py syntax.
        """
        if not self.client:
            print("Cannot place order, not connected to API.")
            return None

        # 1. Calculate Position Size using the RiskManager
        qty = risk_manager.calculate_position_size(
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            risk_percentage=risk_percentage
        )

        if qty <= 0:
            print("Position size is 0. No order will be placed.")
            return None

        # 2. Create the Bracket Order Request object
        print(f"\nPlacing BRACKET order: {side.upper()} {qty} shares of {symbol}...")
        
        # Convert side string to OrderSide enum
        order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL

        # Create individual requests
        market_order_data = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY
        )

        take_profit_data = TakeProfitRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            limit_price=take_profit_price
        )

        stop_loss_data = StopLossRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            stop_price=stop_loss_price
        )

        # 3. Submit the order
        try:
            order = self.client.submit_order(order_data=market_order_data)
            print(f"Market order submitted successfully. Order ID: {order.id}")

            # Place take profit order
            take_profit_order = self.client.submit_order(order_data=take_profit_data)
            print(f"Take profit order submitted successfully. Order ID: {take_profit_order.id}")

            # Place stop loss order
            stop_loss_order = self.client.submit_order(order_data=stop_loss_data)
            print(f"Stop loss order submitted successfully. Order ID: {stop_loss_order.id}")

            return order
        except Exception as e:
            print(f"An error occurred while placing the bracket order: {e}")
            return None

# Note: The `if __name__ == '__main__':` block has been removed to make this a pure library module.
