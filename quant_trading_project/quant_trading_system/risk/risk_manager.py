# quant_trading_system/risk/risk_manager.py

import math

class RiskManager:
    """
    A module for handling position sizing based on a defined risk strategy.

    This class fulfills the core logic of Deliverable 5.2 of the project plan.
    It provides a method to calculate the appropriate position size for a trade
    based on the account's equity and the risk parameters of the trade itself.
    """

    def __init__(self, equity):
        """
        Initializes the RiskManager.

        Args:
            equity (float): The total equity of the trading account.
        """
        if equity <= 0:
            raise ValueError("Account equity must be positive.")
        self.equity = equity
        print(f"Risk Manager initialized with account equity: ${self.equity:,.2f}")

    def calculate_position_size(self, entry_price, stop_loss_price, risk_percentage=0.01):
        """
        Calculates the number of shares to trade based on a fixed fractional
        risk strategy.

        Args:
            entry_price (float): The intended entry price of the asset.
            stop_loss_price (float): The price at which the stop-loss will be set.
            risk_percentage (float, optional): The percentage of total equity to risk
                                               on this single trade. Defaults to 0.01 (1%).

        Returns:
            int: The calculated number of shares (quantity). Returns 0 if risk is invalid.
        """
        if risk_percentage <= 0 or risk_percentage >= 1:
            raise ValueError("Risk percentage must be between 0 and 1.")

        # Determine the risk per share
        risk_per_share = abs(entry_price - stop_loss_price)
        if risk_per_share <= 0:
            print("Warning: Entry price and stop-loss price are the same. Cannot calculate position size.")
            return 0

        # Determine the total dollar amount to risk
        dollar_risk_amount = self.equity * risk_percentage

        # Calculate the number of shares
        position_size = dollar_risk_amount / risk_per_share
        
        # Return the floor of the position size to avoid fractional shares
        # and ensure we do not risk more than intended.
        calculated_qty = math.floor(position_size)
        
        print(f"Risk Calculation:")
        print(f"  - Total Equity: ${self.equity:,.2f}")
        print(f"  - Risk Percentage: {risk_percentage:.2%}")
        print(f"  - Dollar Risk: ${dollar_risk_amount:,.2f}")
        print(f"  - Risk per Share: ${risk_per_share:,.2f}")
        print(f"  - Calculated Position Size (Qty): {calculated_qty}")

        return calculated_qty

# --- Example Usage ---
if __name__ == '__main__':
    # Assume we have a $100,000 trading account
    account_equity = 100000.0
    risk_manager = RiskManager(equity=account_equity)

    # --- Scenario 1: Trading a stock ---
    print("\n" + "="*50)
    print("Scenario 1: Trading AAPL")
    print("="*50)
    aapl_entry = 170.00
    aapl_stop_loss = 165.00 # $5 risk per share
    
    # We want to risk 1.5% of our account on this trade
    aapl_qty = risk_manager.calculate_position_size(
        entry_price=aapl_entry,
        stop_loss_price=aapl_stop_loss,
        risk_percentage=0.015
    )
    print(f"--> Recommended AAPL shares to buy: {aapl_qty}")

    # --- Scenario 2: Trading a more volatile stock ---
    print("\n" + "="*50)
    print("Scenario 2: Trading TSLA (wider stop)")
    print("="*50)
    tsla_entry = 250.00
    tsla_stop_loss = 230.00 # $20 risk per share

    # We risk the same 1.5% of our account
    tsla_qty = risk_manager.calculate_position_size(
        entry_price=tsla_entry,
        stop_loss_price=tsla_stop_loss,
        risk_percentage=0.015
    )
    print(f"--> Recommended TSLA shares to buy: {tsla_qty}")
    print("\nNote how the position size is smaller for the more volatile stock to keep the dollar risk constant.")
