import unittest
import pandas as pd
import numpy as np
from ML_MODEL import transform_api_data, predict_signals

class TestTradingSignalsWithJump(unittest.TestCase):
    def setUp(self):
        # Create 20 days of synthetic data with a price jump:
        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        # First 10 days: slowly increasing from 100 to 105
        close_prices_first = np.linspace(100, 105, 10)
        # Next 10 days: jump from 115 to 125 (a clear jump from the previous segment)
        close_prices_second = np.linspace(115, 125, 10)
        close_prices = np.concatenate([close_prices_first, close_prices_second])
        
        # Construct the synthetic DataFrame simulating PySimFin output.
        data = {
            "Date": dates,
            "Last Closing Price": close_prices,
            "Highest Price": close_prices + 1,  # simple relation
            "Lowest Price": close_prices - 1,
            "Opening Price": close_prices,
            "Trading Volume": np.random.randint(1000000, 2000000, size=20),
            "Dividend Paid": [0.0] * 20,
            "Common Shares Outstanding": [1000000000] * 20,
            "ticker": ["AAPL"] * 20
        }
        self.raw_df = pd.DataFrame(data)
    
    def test_buy_signal_with_jump(self):
        # Transform the raw API data
        transformed = transform_api_data(self.raw_df)
        print("=== Transformed Data Sample ===")
        print(transformed.head(15))
        
        # Generate signals using your ML model
        signals = predict_signals(transformed)
        print("\n=== Signals Sample ===")
        print(signals.head(20))
        
        # Analyze the distribution of actions
        actions = signals["Action"].value_counts().to_dict()
        print("\n=== Action Distribution ===")
        print(actions)
        
        # Check summary stats for Buy Probability
        bp_stats = signals["Buy Probability"].describe()
        print("\n=== Buy Probability Statistics ===")
        print(bp_stats)
        
        # Assert that there's at least one BUY signal (if the jump causes a BUY)
        self.assertIn("BUY", actions, "No BUY signals generated even with a jump in price data.")

if __name__ == '__main__':
    unittest.main()
