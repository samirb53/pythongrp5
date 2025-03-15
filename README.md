# Trading System Web Application

Welcome to our Trading System Web Application! This project combines an ETL process, a machine learning model, a Python API wrapper for SimFin, and a Streamlit-based web interface to deliver real-time stock data, financial statements, trading signals, and a backtesting simulator.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Required CSV Files](#required-csv-files)
- [Usage](#usage)
- [Deployment](#deployment)
- [License](#license)
- [Contact](#contact)

---

## Overview

This application provides:
1. **Share Prices:** Fetch daily share price data for selected tickers.
2. **Financial Statements:** Retrieve quarterly or annual statements from SimFin.
3. **ML Model Predictions:** Predict next-day price movements (BUY, SELL, HOLD).
4. **Trading Strategy & Signals:** Suggest actions based on model output.
5. **Optional Backtesting:** Simulate how a strategy would have performed historically.

The system uses an ETL script to clean and prepare data, an XGBoost-based machine learning model to predict stock movements, and a Streamlit multipage interface for user interaction.

---

## Project Structure
my_trading_app/ ├── app.py # Main Streamlit entry point ├── pysimfin.py # API wrapper for SimFin ├── ETL.py # ETL script for data cleaning & preprocessing ├── ML_MODEL.py # Machine learning model training & prediction ├── requirements.txt # Dependencies ├── README.md # This file ├── pages/ │ ├── 1_Home.py # Home/Overview page │ ├── 2_Go_Live.py # Main interactive page for prices, financials, signals │ └── 3_Backtesting.py # Optional backtesting simulator page └── best_trading_model.pkl # Saved ML model (generated by ML_MODEL.py)


- **`app.py`**: Launches the Streamlit app.
- **`pages/`**: Contains multipage files for the app (Home, Go Live, Backtesting, etc.).
- **`ETL.py`**: Cleans and processes raw CSV data into training/test sets.
- **`ML_MODEL.py`**: Trains the XGBoost model and provides a prediction function.
- **`pysimfin.py`**: Python wrapper for the SimFin API.
- **`requirements.txt`**: Lists all Python dependencies.

---

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   
2. Install dependencies
   ```bash
   pip install -r requirements.txt

3. Obtain CSV Files

For this application to work, it is key that the end user downloads the following .csv files from SimFin:

us-income-quarterly
us-shareprices-daily
us-companies
Place these files in the same directory as ETL.py (or adjust paths in ETL.py accordingly).

4. Run ETL
   ```bash
   python ETL.py

5. Train the ML Model
   ```bash
   python ML_MODEL.py

Required CSV Files
us-income-quarterly.csv
Quarterly income statement data from SimFin.

us-shareprices-daily.csv
Daily share price data for U.S. stocks from SimFin.

us-companies.csv
Company metadata (tickers, industries, etc.) from SimFin.

Make sure these files are in the correct folder (the same folder as ETL.py by default), or modify the file paths in ETL.py if needed.

Usage
Once you have run the ETL and trained the model:

streamlit run app.py

This command will:

Launch the Streamlit application in your browser.
Provide multipage navigation (Home, Go Live, optional Backtesting, etc.).
Let you select tickers, date ranges, and view predictions/trading signals.

Features
Price Data: View share prices for a selected ticker and date range, plus a line chart of closing prices.
Financial Statements: Fetch the most recent or selected period’s statements (Income, Balance, Cash Flow, Derived).
Trading Signals: Next-day BUY/SELL/HOLD predictions, displayed in a visually appealing format.
Optional Backtesting: Evaluate a trading strategy historically to see how it would have performed.

Deployment
To deploy on Streamlit Cloud:

Push your code to a public GitHub repository.
Sign in to Streamlit Cloud and click New app.
Select your repository, branch, and app.py as the main file.
Click Deploy.
Share the generated URL!
License
You can specify a license here (e.g., MIT, Apache 2.0). If you haven’t decided, you can remove this section.

Contributors – [Samir Barakat, Joy Zhong, Nour Sewilam, Thomas Renwick, Pedro Alejandro Medellín]
For major issues, please open an issue on the GitHub repository.


### Explanation of Key Sections

- **Overview**: Explains the app’s purpose and key functionalities.  
- **Project Structure**: Provides a quick map of important files.  
- **Installation**: Detailed steps for cloning, installing dependencies, and preparing data.  
- **Required CSV Files**: Emphasizes that users must have `us-income-quarterly`, `us-shareprices-daily`, and `us-companies` from SimFin.  
- **Usage**: Tells users how to run the ETL, train the model, and launch the Streamlit app.  
- **Deployment**: Summarizes how to deploy on Streamlit Cloud.  
- **License** & **Contact**: Common sections in a README for open-source projects.

Feel free to tailor the text, rename sections, or adjust any details to match your project’s specifics. This should provide a solid foundation for a professional, user-friendly `README.md`.
