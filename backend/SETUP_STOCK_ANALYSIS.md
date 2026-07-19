# Stock Risk Analysis Setup

This document explains the comprehensive stock risk analysis system built for analyzing equity stocks.

## Architecture Overview

The system performs multi-dimensional analysis on stocks using:

1. **Technical Analysis** - RSI, MACD, Moving Averages, Bollinger Bands
2. **Fundamental Analysis** - P/E, P/B, ROE, Profit Margins, Debt ratios
3. **Sentiment Analysis** - News sentiment scoring
4. **Volumetric Analysis** - Volume trends, OBV, Price-Volume confirmation
5. **Sector Comparison** - Performance vs sector peers and market index

## Module Structure

```
backend/
├── modules/
│   ├── data_fetcher.py          # Fetches stock data from Yahoo Finance
│   ├── technical_analysis.py    # Technical indicators and scoring
│   ├── fundamental_analysis.py  # Fundamental metrics and scoring
│   ├── sentiment_analysis.py    # News sentiment analysis
│   ├── volumetric_analysis.py  # Volume pattern analysis
│   ├── sector_comparison.py     # Sector and market comparison
│   └── stock_analyzer.py        # Main orchestrator
└── app.py                       # API endpoints
```

## Installation

### 1. Install Python Dependencies

```bash
cd backend
pip install -r requirements.txt
```

Required packages:
- Flask (API framework)
- yfinance (stock data)
- pandas, numpy (data processing)
- flask-cors (CORS support)

### 2. Install Frontend Dependencies

```bash
cd frontend
npm install
```

## API Endpoints

### POST /api/analyze-stock

Analyzes a stock and returns comprehensive risk score.

**Request:**
```json
{
  "ticker": "AAPL"
}
```

**Response:**
```json
{
  "success": true,
  "ticker": "AAPL",
  "overall_score": {
    "score": 72.5,
    "breakdown": {
      "technical": 75.0,
      "fundamental": 80.0,
      "sentiment": 65.0,
      "volumetric": 70.0,
      "sector": 68.0
    }
  },
  "technical_analysis": {...},
  "fundamental_analysis": {...},
  "sentiment_analysis": {...},
  "volumetric_analysis": {...},
  "sector_comparison": {...},
  "recommendations": [...],
  "risk_level": "Low-Medium Risk / Buy"
}
```

## Usage

### Starting the Services

```bash
# In root directory
./start.bat  # Windows
# or
bash start.sh  # Linux/Mac
```

Or manually:

**Terminal 1:**
```bash
cd backend
python app.py
```

**Terminal 2:**
```bash
cd frontend
npm start
```

### Analyzing a Stock

1. Open `http://localhost:3000`
2. Click on "Stock Analysis" tab
3. Enter stock ticker (e.g., AAPL, MSFT, TSLA)
4. Click "Analyze Stock"
5. View comprehensive analysis results

## Analysis Breakdown

### Technical Analysis (25% weight)
- **RSI** - Relative Strength Index
- **MACD** - Moving Average Convergence Divergence
- **Moving Averages** - SMA 20, SMA 50
- **Bollinger Bands** - Volatility analysis
- **Support/Resistance** levels

### Fundamental Analysis (30% weight)
- **Valuation**: P/E Ratio, P/B Ratio
- **Profitability**: ROE, Profit Margins
- **Financial Health**: Debt-to-Equity, Current Ratio
- **Growth**: Earnings Growth

### Sentiment Analysis (15% weight)
- News sentiment scoring
- Keyword extraction
- Confidence levels
- Positive/negative indicators

### Volumetric Analysis (15% weight)
- Volume trends
- On-Balance Volume (OBV)
- Price-Volume confirmation
- Volume patterns

### Sector Comparison (15% weight)
- Performance vs sector
- Performance vs market index
- Outperformance metrics

## Scoring System

- **75-100**: Strong Buy (Excellent fundamentals and trends)
- **65-74**: Buy (Good overall indicators)
- **45-64**: Hold (Mixed signals)
- **25-44**: Caution (Multiple negative indicators)
- **0-24**: Avoid/Sell (Strong negative indicators)

## Ticker Format

- US Stocks: `AAPL`, `MSFT`, `GOOGL`
- Indian Stocks: `RELIANCE.NS`, `TCS.NS`, `INFY.NS`
- Other exchanges: `SHELL.L`, `TSLA`

## Extending the System

### Adding New Indicators

1. Create module in `backend/modules/`
2. Implement analysis logic
3. Return score (0-100)
4. Add to `stock_analyzer.py` weighted calculation

### Custom Data Sources

Modify `data_fetcher.py` to integrate:
- Alpha Vantage API
- Bloomberg API
- Custom databases

### Advanced Features

- Machine learning predictions
- Real-time alerts
- Portfolio optimization
- Historical performance tracking

## Troubleshooting

### Common Issues

1. **"No data available"**
   - Check ticker symbol is correct
   - Verify internet connection
   - Try a different ticker

2. **Slow analysis**
   - Normal for first analysis (data fetching)
   - Consider caching
   - Reduce analysis period

3. **API rate limits**
   - Implement request throttling
   - Use alternative data sources
   - Cache results

## Examples

### Example Stock Tickers

```python
# US Stocks
"AAPL"    # Apple
"MSFT"    # Microsoft
"GOOGL"   # Google
"TSLA"    # Tesla

# Indian Stocks
"RELIANCE.NS"  # Reliance Industries
"TCS.NS"      # Tata Consultancy Services
"INFY.NS"     # Infosys
"HDFCBANK.NS" # HDFC Bank
```

## Performance Considerations

- First analysis: ~5-10 seconds (data fetching)
- Cached analysis: ~1-2 seconds
- Data freshness: ~15 minutes for real-time data

## Future Enhancements

- [ ] Real-time streaming analysis
- [ ] Machine learning predictions
- [ ] Multi-stock portfolio analysis
- [ ] Backtesting capabilities
- [ ] Custom indicator builder
- [ ] Export to PDF/Excel
- [ ] Email/SMS alerts

