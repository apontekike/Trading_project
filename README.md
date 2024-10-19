Automated Stock Trading Bot

This project involves creating an automated stock trading bot that trades stocks using two strategies:

K-Nearest Neighbors (KNN) Model: A machine learning-based approach that uses technical indicators to make daily buy and sell decisions.
Momentum Strategy: Based on the concepts from the book Quantitative Momentum, this strategy makes monthly trades using momentum indicators.
Data Source
The bot pulls stock data from Yahoo Finance, focusing on tickers from the S&P 500 and S&P MidCap 400 indices, which are provided in the CSV file Ticker.csv.

Goals
The goal of this project is to implement these strategies and perform daily paper trades to test the bot’s performance and efficiency.

Project Structure
1. Data Collection
Source: Yahoo Finance.
Tickers: Located in the file Ticker.csv (S&P 500 and S&P MidCap 400).
Steps:

Use Python libraries such as yfinance to download historical stock data (daily close prices, volume, and other technical indicators).
Prepare the data by cleaning and structuring it for modeling and strategy implementation.

2. KNN Model for Daily Trading
Purpose: To make daily buy/sell decisions based on technical indicators.
Steps:

Feature Engineering: Calculate technical indicators such as moving averages (SMA/EMA), RSI, MACD, and others relevant to stock price movements.
Model Training: Implement and train a KNN model using historical data. Use indicators as features and classify actions (buy, hold, sell) as targets.
Daily Predictions: Based on the trained model, make daily predictions for each stock in the pool and simulate buy/sell actions.

3. Momentum Strategy for Monthly Trading
Purpose: To implement a momentum-based strategy with monthly frequency, inspired by Quantitative Momentum.
Steps:

Momentum Calculation: Use monthly data to calculate momentum indicators such as the 12-month momentum or price strength.
Ranking: Rank the stocks based on momentum values and select the top/bottom performers.
Monthly Rebalancing: Execute buy/sell trades based on the strategy at the end of each month.

4. Paper Trading
Simulation: Use a paper trading environment to simulate trades.
Daily Execution: Run the KNN-based strategy daily.
Monthly Execution: Run the momentum-based strategy at the end of each month.
Logging: Track each trade’s outcome and overall portfolio performance.

6. Evaluation
Metrics: Calculate key performance metrics like return on investment (ROI), Sharpe ratio, and drawdown.
Backtesting: Backtest both strategies over historical data to evaluate their performance over time.

Working Progress

