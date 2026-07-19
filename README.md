# Investment Risk Profiler

A full-stack application that helps users assess their investment risk profile based on their financial goals and preferences.

## Project Structure

```
investment-risk-profiler/
├── backend/                 # Flask API server
│   ├── app.py              # Main Flask application
│   ├── requirements.txt    # Python dependencies
│   └── streamlit_app.py    # Streamlit alternative interface
├── frontend/               # React frontend application
│   ├── package.json        # Node.js dependencies
│   ├── public/             # Static files
│   └── src/                # React source code
├── data/                   # Data files
└── docs/                   # Documentation
```

## Setup Instructions

### Backend Setup

1. **Navigate to the backend directory:**
   ```bash
   cd backend
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - Windows:
     ```bash
     source venv/Scripts/activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the Flask server:**
   ```bash
   python app.py
   ```

   The backend will start on `http://localhost:5000`

### Frontend Setup

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install Node.js dependencies:**
   ```bash
   npm install
   ```

3. **Start the React development server:**
   ```bash
   npm start
   ```

   The frontend will start on `http://localhost:3000`

## How the Connection Works

### Backend API Endpoints

- `POST /api/profile` - Calculate risk profile
- `GET /api/health` - Health check endpoint
- `POST /api/analyze-stock` - Comprehensive stock analysis
- `GET /api/trading-strategies` - Get available trading strategies
- `POST /api/strategy-signals` - Get trading signals
- `POST /api/backtest-strategy` - Backtest trading strategies

### Frontend-Backend Communication

The frontend communicates with the backend through:

1. **Proxy Configuration**: The `package.json` includes a proxy setting that forwards API calls from the frontend to the backend:
   ```json
   "proxy": "http://localhost:5000"
   ```

2. **API Calls**: The React app uses Axios to make HTTP requests to the backend:
   ```javascript
   const response = await axios.post('/api/profile', formData);
   ```

3. **CORS Support**: The backend includes Flask-CORS to allow cross-origin requests from the frontend.

## Running Both Services

### Option 1: Separate Terminals

1. **Terminal 1 - Backend:**
   ```bash
   cd backend
   python app.py
   ```

2. **Terminal 2 - Frontend:**
   ```bash
   cd frontend
   npm start
   ```

### Option 2: Using a Script

You can create a script to run both services simultaneously. Create a `start.bat` (Windows) or `start.sh` (macOS/Linux) file in the root directory:

**Windows (start.bat):**
```batch
@echo off
start cmd /k "cd backend && python app.py"
start cmd /k "cd frontend && npm start"
```

**macOS/Linux (start.sh):**
```bash
#!/bin/bash
cd backend && python app.py &
cd frontend && npm start &
```

## Available Features

### 1. Stock Analysis
Analyze stocks using multiple dimensions:
- Technical Analysis (RSI, MACD, Moving Averages)
- Fundamental Analysis (P/E, ROE, Profitability)
- Sentiment Analysis (News sentiment)
- Volumetric Analysis (Volume patterns)
- Sector Comparison (vs Market and Peers)

### 2. Investor Profiling
Assess your investment risk profile:
- Based on investment goals and horizon
- Risk appetite assessment
- Personalized recommendations

### 3. Trading Strategies (NEW!)
Execute and backtest quantitative trading strategies:
- **Trend Following** - Momentum-based trading
- **Mean Reversion** - Pairs trading
- **Technical Indicators** - RSI, MACD, Bollinger Bands
- **Calendar Spread** - Futures spread trading

See [TRADING_STRATEGIES_GUIDE.md](TRADING_STRATEGIES_GUIDE.md) for detailed instructions.

## Testing the Connection

1. Open your browser and go to `http://localhost:3000`
2. Choose a tab: Stock Analysis, Investor Profiling, or Trading Strategies
3. Fill out the form and submit
4. The frontend will send the data to the backend API
5. The backend will process and return results
6. The results will be displayed on the frontend

## API Response Format

The backend returns JSON responses in this format:

```json
{
  "success": true,
  "risk_score": 7,
  "risk_level": "High",
  "recommendations": [
    "Higher allocation to equity investments",
    "Consider direct equity investments",
    "Aggressive mutual funds and ETFs"
  ]
}
```

## Troubleshooting

### Common Issues

1. **Port already in use:**
   - Backend: Change port in `app.py` (line with `app.run()`)
   - Frontend: React will automatically suggest an alternative port

2. **CORS errors:**
   - Ensure the backend is running with CORS enabled
   - Check that the proxy setting in `package.json` is correct

3. **Module not found errors:**
   - Backend: Make sure all dependencies are installed (`pip install -r requirements.txt`)
   - Frontend: Make sure all dependencies are installed (`npm install`)

4. **API connection issues:**
   - Verify both servers are running
   - Check browser console for error messages
   - Ensure the backend is accessible at `http://localhost:5000`

## Development

### Adding New Features

1. **Backend API endpoints:** Add new routes in `backend/app.py`
2. **Frontend components:** Create new React components in `frontend/src/`
3. **Styling:** Modify CSS files in `frontend/src/`

### Data Flow

1. User fills form on frontend
2. Frontend sends POST request to `/api/profile`
3. Backend processes data and calculates risk score
4. Backend returns JSON response with results
5. Frontend displays results to user

This architecture provides a clean separation between the frontend UI and backend logic, making it easy to maintain and extend.