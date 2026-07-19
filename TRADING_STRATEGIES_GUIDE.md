# Trading Strategies Execution Guide

## Overview

The Investment Risk Profiler now includes a comprehensive **Trading Strategies** module that allows you to:
- Get real-time trading signals from multiple quantitative strategies
- Backtest strategies on historical data
- Analyze strategy performance with detailed metrics

## How to Execute Trading Strategies

### Step 1: Start the Application

Run the application using the provided startup script:

**Windows:**
```bash
start.bat
```

**Linux/Mac:**
```bash
chmod +x start.sh
./start.sh
```

This will start:
- Backend Flask server on `http://localhost:5000`
- Frontend React server on `http://localhost:3000`

### Step 2: Access Trading Strategies Tab

1. Open your browser and navigate to `http://localhost:3000`
2. Click on the **"Trading Strategies"** tab at the top

### Step 3: Choose an Action

You have two main options:

#### A. Get Current Trading Signals
- **Purpose**: Get real-time buy/sell/hold signals based on current market conditions
- **Use Case**: Decide what to do now with a stock

#### B. Backtest Strategy
- **Purpose**: Test how a strategy would have performed on historical data
- **Use Case**: Evaluate strategy performance before using real money

### Step 4: Select a Strategy

Choose from 4 available strategies:

#### 1. Trend Following (Momentum)
- **What it does**: Identifies upward trends and buys, downward trends and sells
- **Indicators**: Moving averages (50/200 day), ADX
- **Best for**: Stocks with clear trends
- **Example**: AAPL, MSFT, AMZN

#### 2. Mean Reversion (Pairs Trading)
- **What it does**: Buys undervalued pairs, sells overvalued pairs
- **Requirements**: Two correlated stocks (pair ticker needed)
- **Best for**: Stock pairs with historical correlation
- **Example**: AAPL-MSFT, GOOGL-MSFT

#### 3. Technical Indicators
- **What it does**: Combines RSI, MACD, and Bollinger Bands
- **Best for**: Active traders looking for entry/exit points
- **Example**: Any liquid stock

#### 4. Calendar Spread
- **What it does**: Trades futures contract spreads
- **Requirements**: Two contract months (pair ticker needed)
- **Best for**: Futures traders
- **Example**: ES futures contracts

### Step 5: Enter Stock Information

1. **Stock Ticker** (Required): Enter the stock symbol
   - US stocks: `AAPL`, `MSFT`, `GOOGL`
   - Indian stocks: `RELIANCE.NS`, `TCS.NS`
   - Any valid Yahoo Finance ticker

2. **Pair/Contract Ticker** (Required for Pairs & Calendar Spread):
   - For pairs: Enter the second stock (e.g., `MSFT` when trading AAPL-MSFT)
   - For calendar spread: Enter the next month contract

3. **Initial Capital** (For backtests only):
   - Default: $100,000
   - Adjust to match your trading capital

### Step 6: View Results

#### For Signals:
- **Current Signal**: BUY, SELL, or HOLD
- **Current Price**: Latest stock price
- **Position Size**: Recommended number of shares or hedge ratio
- **Recommendation**: Actionable trading advice

#### For Backtests:
- **Returns**: Total return percentage
- **Trade Statistics**: Number of trades and win rate
- **Risk Metrics**: Max drawdown and Sharpe ratio
- **Trade History**: Detailed list of all trades with P&L

## Example Use Cases

### Example 1: Get Trading Signal for Apple Stock

1. **Action**: Get Current Signals
2. **Strategy**: Technical Indicators
3. **Stock Ticker**: `AAPL`
4. **Click**: "Get Signals"

**Expected Output:**
- Signal: BUY/SELL/HOLD
- Current price: $XXX
- Recommended shares: XXX

### Example 2: Backtest Trend Following on Microsoft

1. **Action**: Backtest Strategy
2. **Strategy**: Trend Following
3. **Stock Ticker**: `MSFT`
4. **Initial Capital**: 100000
5. **Click**: "Run Backtest"

**Expected Output:**
- Total return: XX%
- Win rate: XX%
- Sharpe ratio: X.XX
- List of all trades

### Example 3: Pairs Trading AAPL and MSFT

1. **Action**: Get Current Signals
2. **Strategy**: Mean Reversion
3. **Stock Ticker**: `AAPL`
4. **Pair Ticker**: `MSFT`
5. **Click**: "Get Signals"

**Expected Output:**
- Signal for the pair
- Hedge ratio (e.g., "1:0.79")
- Recommended shares for each stock

## API Endpoints (For Developers)

If you want to integrate trading strategies into your own application:

### 1. Get Available Strategies
```
GET /api/trading-strategies
```

### 2. Get Trading Signals
```
POST /api/strategy-signals
Body: {
  "ticker": "AAPL",
  "strategy_type": "technical",
  "parameters": {},
  "pair_ticker": null  // optional for pairs trading
}
```

### 3. Backtest Strategy
```
POST /api/backtest-strategy
Body: {
  "ticker": "AAPL",
  "strategy_type": "trend_following",
  "initial_capital": 100000,
  "parameters": {},
  "pair_ticker": null  // optional for pairs trading
}
```

## Understanding the Results

### Signal Colors
- **Green (BUY)**: Strong buy signal - bullish conditions
- **Red (SELL)**: Sell signal - bearish conditions
- **Orange (HOLD)**: No clear signal - wait

### Backtest Metrics

**Returns:**
- Total % return over the backtest period
- Initial vs final capital

**Win Rate:**
- Percentage of profitable trades
- Higher is generally better (>50% is good)

**Sharpe Ratio:**
- Risk-adjusted return
- Higher is better (>1 is decent, >2 is excellent)

**Max Drawdown:**
- Largest peak-to-trough decline
- Lower is better (represents risk)

**Avg Win vs Avg Loss:**
- Average profit per winning trade
- Average loss per losing trade
- Win amount should exceed loss amount

## Tips for Best Results

1. **For Signals**:
   - Use current market hours for most accurate signals
   - Combine multiple strategies for confirmation
   - Check fundamental analysis before acting on signals

2. **For Backtests**:
   - Use at least 1 year of data (default is 2 years)
   - Consider commission costs (default 0.1%)
   - Compare multiple strategies side-by-side

3. **Strategy Selection**:
   - **Trend Following**: Best in trending markets
   - **Mean Reversion**: Best in ranging/volatile markets
   - **Technical**: Most versatile, good for most stocks
   - **Calendar Spread**: Advanced, for futures traders

4. **Risk Management**:
   - Never risk more than 2% per trade
   - Use stop losses (position sizing helps)
   - Diversify across strategies and assets

## Troubleshooting

### "Insufficient historical data"
- Use a different stock with more history
- The stock may have limited public trading history

### "Could not fetch data"
- Check internet connection
- Verify ticker symbol is correct
- Try a different exchange suffix (e.g., .NS for Indian stocks)

### "Pair data required" error
- For Mean Reversion and Calendar Spread, you must provide a pair ticker
- Make sure both tickers are valid

### No signals generated
- The strategy may not have detected trading conditions
- Try a different stock or strategy
- Technical indicators may need more data to generate signals

## Next Steps

1. **Paper Trade**: Test strategies with paper trading before real money
2. **Combine Strategies**: Use multiple signals for confirmation
3. **Monitor Performance**: Track your actual results vs backtest
4. **Optimize Parameters**: Adjust strategy parameters based on results
5. **Consider Risk**: Always consider your risk tolerance

## Need Help?

- Check the backend logs in the terminal running `app.py`
- Review strategy documentation in `backend/modules/trading_strategies.py`
- Test with well-known stocks first (AAPL, MSFT, etc.)

## Important Disclaimers

⚠️ **For Educational Purposes**: This system is for educational and research purposes only.

⚠️ **Not Financial Advice**: Signals and backtests are not financial advice.

⚠️ **Past Performance**: Past performance doesn't guarantee future results.

⚠️ **Risk Warning**: Trading involves risk. Never invest more than you can afford to lose.

Happy Trading! 📈

