# How to Execute Trading Strategies - Quick Start

## 🚀 Quick Start (3 Steps)

### Step 1: Start the Application
```bash
# Windows
start.bat

# Linux/Mac
./start.sh
```

### Step 2: Open Browser
Navigate to `http://localhost:3000`

### Step 3: Use Trading Strategies Tab
1. Click **"Trading Strategies"** tab
2. Choose an action: **Get Signals** or **Backtest**
3. Select a strategy
4. Enter ticker symbol (e.g., `AAPL`)
5. Click the action button
6. View results!

---

## 📊 What You Can Do

### Option A: Get Current Trading Signals
**Purpose**: Get real-time BUY/SELL/HOLD signals

**Example**:
- Strategy: **Technical Indicators**
- Ticker: **AAPL**
- Click: **Get Signals**

**Output**: 
- Current signal (BUY/SELL/HOLD)
- Recommended position size
- Current price

### Option B: Backtest Strategy Performance
**Purpose**: Test strategy on historical data

**Example**:
- Action: **Backtest Strategy**
- Strategy: **Trend Following**
- Ticker: **MSFT**
- Capital: 100000
- Click: **Run Backtest**

**Output**:
- Total return %
- Win rate
- Sharpe ratio
- Trade history

---

## 🎯 Available Strategies

| Strategy | Best For | Requirements |
|----------|----------|--------------|
| **Trend Following** | Trending stocks | Stock ticker only |
| **Technical Indicators** | Active trading | Stock ticker only |
| **Mean Reversion** | Correlated pairs | Stock + Pair ticker |
| **Calendar Spread** | Futures trading | Near + Next month |

---

## 📝 Examples

### Example 1: Simple Signal for Apple
```
Action: Get Current Signals
Strategy: Technical Indicators
Ticker: AAPL
→ Click "Get Signals"
```

### Example 2: Backtest Microsoft
```
Action: Backtest Strategy
Strategy: Trend Following
Ticker: MSFT
Capital: 100000
→ Click "Run Backtest"
```

### Example 3: Pairs Trading
```
Action: Get Current Signals
Strategy: Mean Reversion
Ticker: AAPL
Pair Ticker: MSFT
→ Click "Get Signals"
```

---

## 🔗 For More Details

See **[TRADING_STRATEGIES_GUIDE.md](TRADING_STRATEGIES_GUIDE.md)** for:
- Detailed feature explanations
- Advanced usage examples
- API documentation
- Troubleshooting guide
- Risk management tips

---

## ⚠️ Important

**For Educational Purposes Only**
- Not financial advice
- Test before using real money
- Past performance ≠ future results
- Trading involves risk

---

## 🆘 Need Help?

**Common Issues:**

1. **"Insufficient data"**: Try a different stock
2. **"Could not fetch"**: Check ticker symbol
3. **"Pair required"**: Add second ticker for pairs
4. **No signals**: Try different stock/strategy

**Check:**
- Backend running on port 5000?
- Frontend running on port 3000?
- Internet connection active?
- Valid ticker symbol?

---

**That's it! Start executing trading strategies now! 📈**

