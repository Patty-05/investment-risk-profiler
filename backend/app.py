# NOTE: torch must be imported before MKL/OpenMP libraries (numpy, scikit-learn,
# xgboost). On Windows, loading torch after them triggers a DLL init conflict
# (WinError 1114 loading c10.dll) that silently disables FinBERT sentiment.
# Guarded so the app still runs if torch isn't installed (sentiment falls back
# to the keyword engine).
try:
    import torch  # noqa: F401  (import for side-effect: DLL load order)
except Exception:
    pass

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
import traceback
from dotenv import load_dotenv
from modules.seasonal_analysis import IndianSeasonalAnalyzer
from datetime import datetime, timedelta
from modules.stock_analyzer import StockRiskAnalyzer
from modules.trading_strategies import StrategySelector, Backtester, RiskManager
from modules.data_fetcher import StockDataFetcher
from services import war_service
import pandas as pd

from datetime import datetime, timedelta


# ✅ ADD THIS
app = Flask(__name__)
CORS(app)

load_dotenv()


# Cache
_seasonal_cache = {
    "data": None,
    "fetched_at": None,
    "ttl_hours": 6
}

# ✅ NOW route will work


# ✅ NEW: JSON serializer that handles numpy types and datetime objects
def sanitize_for_json(obj):
    import numpy as np
    import pandas as pd

    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [sanitize_for_json(i) for i in obj]

    elif isinstance(obj, tuple):
        return tuple(sanitize_for_json(i) for i in obj)

    # 🔥 Handle pandas
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict()

    elif isinstance(obj, pd.Series):
        return obj.to_dict()

    # 🔥 Handle numpy types
    elif isinstance(obj, np.bool_):
        return bool(obj)

    elif isinstance(obj, (np.integer,)):
        return int(obj)

    elif isinstance(obj, (np.floating,)):
        return float(obj)

    elif pd.isna(obj):
        return None

    return obj

@app.route('/api/analyze-stock', methods=['POST'])
def analyze_stock():
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'Request body is required'}), 400

        ticker = data.get('ticker', '').upper().strip()
        if not ticker:
            return jsonify({'success': False, 'error': 'Stock ticker is required'}), 400

        # Investor risk profile — personalises the portfolio allocation.
        # Optional; the optimizer applies sensible 'medium' defaults if absent.
        risk_appetite = data.get('risk_appetite') or data.get('riskAppetite')
        horizon = data.get('horizon')

        print(f"Analyzing stock: {ticker} (risk_appetite={risk_appetite}, horizon={horizon})")

        fetcher = StockDataFetcher(ticker)
        if not fetcher.validate_ticker():
            return jsonify({'success': False, 'error': 'Invalid or unsupported stock ticker'}), 400

        # ✅ FIX: Analyzer reuses the fetcher — avoids double network call
        analyzer = StockRiskAnalyzer(ticker, risk_appetite=risk_appetite, horizon=horizon)
        results = analyzer.analyze()

        if not results.get('success'):
            return jsonify(results), 400

        return jsonify({
            "success": True,
            "data": sanitize_for_json(results)
        })

    except Exception as e:
        print(f"Error in stock analysis: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': f'Failed to analyze stock: {str(e)}'}), 500


@app.route('/api/profile', methods=['POST'])
def profile_investor():
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'Request body is required'}), 400

        print(f"Received data: {data}")

        goal = data.get('goal', '')
        risk_appetite = data.get('riskAppetite', '')

        try:
            horizon = int(data.get('horizon', 0))
            income = int(data.get('income', 0))
        except (TypeError, ValueError):
            return jsonify({'success': False, 'error': 'horizon and income must be numbers'}), 400

        # ✅ FIX: horizon must be at least 1 (0 is not a valid investment horizon)
        if horizon < 1:
            return jsonify({'success': False, 'error': 'Investment horizon must be at least 1 year'}), 400
        if income < 0:
            return jsonify({'success': False, 'error': 'Income cannot be negative'}), 400
        if not goal:
            return jsonify({'success': False, 'error': 'Investment goal is required'}), 400
        if not risk_appetite:
            return jsonify({'success': False, 'error': 'Risk appetite is required'}), 400

        risk_score = calculate_risk_score(goal, horizon, income, risk_appetite)

        return jsonify({
            'success': True,
            'risk_score': risk_score,
            'risk_level': get_risk_level(risk_score),
            'recommendations': get_recommendations(risk_score)
        })

    except Exception as e:
        print(f"Error in profile: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Backend is running'})


@app.route('/api/conflicts', methods=['GET'])
def get_conflicts():
    """Curated geopolitical conflict watchlist for the War Room."""
    try:
        return jsonify({'success': True, 'conflicts': war_service.list_conflicts()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/war-analysis', methods=['POST'])
def war_analysis():
    """
    India-specific geopolitical impact analysis. Provide either:
      - conflict_id   (from the curated watchlist), or
      - scenario_text (a free-text 'what if' scenario).
    """
    try:
        data = request.json or {}
        result = war_service.analyze(
            conflict_id=data.get('conflict_id'),
            scenario_text=data.get('scenario_text'),
        )
        status = 200 if result.get('success') else 400
        return jsonify(sanitize_for_json(result)), status
    except Exception as e:
        print(f"Error in war analysis: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/trading-strategies', methods=['GET'])
def get_trading_strategies():
    strategies = StrategySelector.get_available_strategies()
    return jsonify({'success': True, 'strategies': strategies})


@app.route('/api/backtest-strategy', methods=['POST'])
def backtest_strategy():
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'Request body is required'}), 400

        ticker = data.get('ticker', '').upper().strip()
        strategy_type = data.get('strategy_type', '').lower()
        parameters = data.get('parameters', {})
        pair_ticker = data.get('pair_ticker', None)

        # ✅ FIX: Validate initial_capital
        try:
            initial_capital = float(data.get('initial_capital', 100000))
            if initial_capital <= 0:
                raise ValueError
        except (TypeError, ValueError):
            return jsonify({'success': False, 'error': 'initial_capital must be a positive number'}), 400

        if not ticker:
            return jsonify({'success': False, 'error': 'Stock ticker is required'}), 400
        if not strategy_type:
            return jsonify({'success': False, 'error': 'Strategy type is required'}), 400

        fetcher = StockDataFetcher(ticker, period='2y')
        hist_data = fetcher.get_historical_data()

        if hist_data is None or len(hist_data) < 100:
            return jsonify({'success': False, 'error': 'Insufficient historical data for backtesting'}), 400

        if not isinstance(hist_data, pd.DataFrame):
            hist_data = pd.DataFrame(hist_data)

        pair_data = None
        if strategy_type in ['mean_reversion', 'pairs', 'calendar_spread']:
            if not pair_ticker:
                return jsonify({'success': False, 'error': f'Pair ticker required for {strategy_type} strategy'}), 400
            pair_fetcher = StockDataFetcher(pair_ticker, period='2y')
            pair_data = pair_fetcher.get_historical_data()
            if pair_data is None:
                return jsonify({'success': False, 'error': 'Could not fetch pair data'}), 400
            if not isinstance(pair_data, pd.DataFrame):
                pair_data = pd.DataFrame(pair_data)

        strategy = StrategySelector.create_strategy(
            strategy_type=strategy_type,
            data=hist_data,
            pair_data=pair_data,
            parameters=parameters
        )

        backtester = Backtester(strategy, initial_capital=initial_capital)
        results = backtester.run_backtest()

        # ✅ FIX: Sanitize trades list — dates and numpy types break jsonify
        safe_trades = sanitize_for_json(
            results['trades'][:20] if len(results['trades']) > 20 else results['trades']
        )

        formatted_results = {
            'success': True,
            'strategy_type': strategy_type,
            'ticker': ticker,
            'backtest_results': {
                'initial_capital': initial_capital,
                'final_capital': results['final_capital'],
                'total_return_pct': round(float(results['total_return']), 2),
                'total_trades': int(results['total_trades']),
                'winning_trades': int(results['winning_trades']),
                'losing_trades': int(results['losing_trades']),
                'win_rate_pct': round(float(results['win_rate']), 2),
                'avg_win': round(float(results['avg_win']), 2),
                'avg_loss': round(float(results['avg_loss']), 2),
                'max_drawdown_pct': round(float(results['max_drawdown']), 2),
                'sharpe_ratio': round(float(results['sharpe_ratio']), 2)
            },
            'trades': safe_trades
        }

        return jsonify(formatted_results)

    except Exception as e:
        print(f"Error in backtest: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': f'Failed to backtest strategy: {str(e)}'}), 500


@app.route('/api/strategy-signals', methods=['POST'])
def get_strategy_signals():
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'Request body is required'}), 400

        ticker = data.get('ticker', '').upper().strip()
        strategy_type = data.get('strategy_type', '').lower()
        parameters = data.get('parameters', {})
        pair_ticker = data.get('pair_ticker', None)

        if not ticker:
            return jsonify({'success': False, 'error': 'Stock ticker is required'}), 400
        if not strategy_type:
            return jsonify({'success': False, 'error': 'Strategy type is required'}), 400

        fetcher = StockDataFetcher(ticker, period='1y')
        hist_data = fetcher.get_historical_data()

        if hist_data is None or len(hist_data) < 50:
            return jsonify({'success': False, 'error': 'Insufficient historical data'}), 400

        if not isinstance(hist_data, pd.DataFrame):
            hist_data = pd.DataFrame(hist_data)

        pair_data = None
        if strategy_type in ['mean_reversion', 'pairs', 'calendar_spread']:
            if pair_ticker:
                pair_fetcher = StockDataFetcher(pair_ticker, period='1y')
                pair_data = pair_fetcher.get_historical_data()
                if pair_data is not None and not isinstance(pair_data, pd.DataFrame):
                    pair_data = pd.DataFrame(pair_data)

        strategy = StrategySelector.create_strategy(
            strategy_type=strategy_type,
            data=hist_data,
            pair_data=pair_data,
            parameters=parameters
        )

        signals = strategy.generate_signals()
        latest_signal = signals.iloc[-1]
        latest_price = hist_data['Close'].iloc[-1]

        account_value = data.get('account_value', 100000)
        position_size = None
        if latest_signal['signal'] != 0:
            try:
                if strategy_type == 'mean_reversion' and pair_data is not None:
                    pair_price = pair_data['Close'].iloc[-1]
                    position_size = strategy.calculate_position_size(
                        'BUY' if latest_signal['signal'] > 0 else 'SELL',
                        latest_price,
                        account_value,
                        pair_price
                    )
                else:
                    position_size = strategy.calculate_position_size(
                        'BUY' if latest_signal['signal'] > 0 else 'SELL',
                        latest_price,
                        account_value
                    )
            except Exception as pos_err:
                # ✅ FIX: Log instead of silently swallowing
                print(f"Position size calculation failed: {pos_err}")
                position_size = None

        signal_val = int(latest_signal['signal'])
        signal_text = 'BUY' if signal_val > 0 else ('SELL' if signal_val < 0 else 'HOLD')

        return jsonify({
            'success': True,
            'ticker': ticker,
            'strategy_type': strategy_type,
            'current_signal': {
                'signal': signal_val,
                'position': int(latest_signal['position']),
                'signal_text': signal_text,
                'current_price': float(latest_price),
                'date': str(hist_data.index[-1])
            },
            'position_size': sanitize_for_json(position_size),
            'recommendation': signal_text.capitalize()
        })

    except Exception as e:
        print(f"Error getting signals: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': f'Failed to get signals: {str(e)}'}), 500
    
# ─────────────────────────────────────────────────────────────────────────────
# ADD TO backend/app.py
#
# Step 1 — add this import near the top (with other module imports):
#     from modules.seasonal_analysis import IndianSeasonalAnalyzer
#
# Step 2 — add this cache dict at module level (outside any function):
#     _seasonal_cache = {"data": None, "fetched_at": None, "ttl_hours": 6}
#
# Step 3 — paste the route below into app.py alongside other @app.route blocks
# ─────────────────────────────────────────────────────────────────────────────

_seasonal_cache = {
    "data":       None,
    "fetched_at": None,
    "ttl_hours":  6
}


@app.route('/api/seasonal-analysis', methods=['GET'])
def seasonal_analysis():
    try:
        force_refresh = request.args.get('refresh', 'false').lower() == 'true'

        # Return cached data if still fresh
        if (not force_refresh
                and _seasonal_cache["data"] is not None
                and _seasonal_cache["fetched_at"] is not None):
            age = datetime.now() - _seasonal_cache["fetched_at"]
            if age < timedelta(hours=_seasonal_cache["ttl_hours"]):
                print(f"📦 Returning cached seasonal data (age: {int(age.total_seconds() / 60)}min)")
                return jsonify({
                    "success": True,
                    "data":    sanitize_for_json(_seasonal_cache["data"]),
                    "cached":  True,
                    "cache_age_minutes": int(age.total_seconds() / 60)
                })

        print("🌿 Fetching fresh seasonal analysis...")
        analyzer = IndianSeasonalAnalyzer(years=10)
        result   = analyzer.analyze()          # ✅ correct method name

        if not result.get("success"):
            return jsonify({"success": False, "error": "Seasonal analysis failed"}), 500

        _seasonal_cache["data"]       = result
        _seasonal_cache["fetched_at"] = datetime.now()

        return jsonify({
            "success": True,
            "data":    sanitize_for_json(result),
            "cached":  False
        })

    except Exception as e:
        import traceback
        print(f"❌ Seasonal analysis error:\n{traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500


def calculate_risk_score(goal, horizon, income, risk_appetite):
    score = 0

    # Goal scoring
    goal_map = {'Wealth Building': 3, 'Retirement': 2, 'Education': 1}
    score += goal_map.get(goal, 1)

    # ✅ FIX: More granular horizon bands
    if horizon >= 20:
        score += 3
    elif horizon >= 10:
        score += 2
    elif horizon >= 5:
        score += 1
    else:
        score += 0  # Short-term = no bonus

    # Income scoring
    if income > 1000000:
        score += 3
    elif income > 500000:
        score += 2
    else:
        score += 1

    # Risk appetite scoring
    appetite_map = {'High': 3, 'Medium': 2, 'Low': 1}
    score += appetite_map.get(risk_appetite, 1)

    # ✅ FIX: Max is now 12 (3+3+3+3), not 10
    return min(score, 12)


def get_risk_level(score):
    # ✅ Updated thresholds to match new max of 12
    if score <= 4:
        return 'Low'
    elif score <= 8:
        return 'Medium'
    else:
        return 'High'


def get_recommendations(score):
    if score <= 4:
        return [
            'Consider conservative investments like FDs and government bonds',
            'Focus on capital preservation',
            'Consider debt mutual funds'
        ]
    elif score <= 8:
        return [
            'Balanced portfolio with mix of equity and debt',
            'Consider hybrid mutual funds',
            'Systematic Investment Plans (SIPs) in equity funds'
        ]
    else:
        return [
            'Higher allocation to equity investments',
            'Consider direct equity investments',
            'Aggressive mutual funds and ETFs'
        ]

def clean_data(obj):
    """Convert numpy/pandas types to native Python types"""
    import numpy as np

    if isinstance(obj, dict):
        return {k: clean_data(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_data(i) for i in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    else:
        return obj


if __name__ == '__main__':
    app.run(debug=True)