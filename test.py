import logging
import datetime
from datetime import timedelta
from pysimfin import PySimFin
import pandas as pd
from ML_MODEL import predict_signals, transform_api_data

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# SimFin API key (replace with the updated one)
api_key = "e1c75cc5-3bca-4b0c-b847-6447bd4ed901"
simfin = PySimFin(api_key)

# Limited test parameters to avoid API rate limits
tickers = ["AAPL", "MSFT"]  # Small subset of tickers
strategies = ["Strategy 1: Buy-and-Hold"]
fiscal_years = [2024]  # Single fiscal year for minimal testing
fiscal_periods = ["Q4"]  # Single fiscal period
today = datetime.date.today()

# Test with a single 30-day date range
date_ranges = [(today - timedelta(days=30), today)]

results = []

for ticker in tickers:
    for start_date, end_date in date_ranges:
        for strategy in strategies:
            for fy in fiscal_years:
                for period in fiscal_periods:
                    try:
                        # Fetch share prices
                        share_df = simfin.get_share_prices(
                            ticker=ticker,
                            start=start_date.strftime("%Y-%m-%d"),
                            end=end_date.strftime("%Y-%m-%d")
                        )

                        # Fetch financial statements
                        stmts = simfin.get_financial_statement(
                            ticker=ticker,
                            fyear=fy,
                            period=period,
                            statements="PL,BS,CF,DERIVED"
                        )

                        # Fetch trading signals
                        trade_date = end_date - timedelta(days=1)
                        trade_df = simfin.get_share_prices(
                            ticker=ticker,
                            start=trade_date.strftime("%Y-%m-%d"),
                            end=trade_date.strftime("%Y-%m-%d")
                        )

                        if not trade_df.empty:
                            transformed = transform_api_data(trade_df)
                            signals = predict_signals(transformed)

                            if not signals.empty:
                                action = signals.iloc[0]["Action"]
                                prob = signals.iloc[0]["Buy Probability"]
                                results.append((ticker, trade_date, action, prob))

                    except Exception as e:
                        logging.error(f"Error encountered for {ticker}: {e}")

# Display results
import ace_tools as tools
results_df = pd.DataFrame(results, columns=["Ticker", "Date", "Predicted Action", "Buy Probability"])
tools.display_dataframe_to_user(name="Test Results", dataframe=results_df)
