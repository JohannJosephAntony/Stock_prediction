Advanced AI Stock Prediction Model - Frontend
This is the frontend component of an advanced AI stock prediction application. It provides a user-friendly interface to view stock predictions, manage a personal watchlist, analyze company fundamentals, and stay updated with market news.

✨ Features
Multi-Model Predictions: Get 1-day, 7-day, and 30-day stock price predictions using various AI model types (mocked for frontend demonstration).

Interactive Charts: Visualize historical stock prices, MACD (Moving Average Convergence Divergence), and RSI (Relative Strength Index) for technical analysis.

Dynamic Analysis Tab: View key financial metrics, a company profile, and a conceptual industry comparison for a selected stock. This tab starts empty and populates only after a stock symbol has been successfully processed in the Prediction tab.

Personal Watchlist: Add stocks to your custom watchlist with a specified quantity. Each item tracks its symbol, quantity, average buy price, current value, and percentage change.

Watchlist Management: Easily remove stocks from your watchlist using the "Remove" button next to each item. The watchlist starts empty.

Conceptual Watchlist Performance: See a conceptual performance chart for your entire watchlist.

Real-time News Feed: Stay informed with the latest market news powered by the TradingView widget, now displayed at a larger, more prominent size.

User-Friendly Interface: Clean, responsive design built with Tailwind CSS for optimal viewing on various devices.

💻 Technologies Used
HTML5: Structure of the web page.

CSS (Tailwind CSS): For modern and responsive styling.

JavaScript (ES6+): For interactive elements, tab switching, data handling, and chart rendering, including custom modal for quantity input.

Chart.js: A popular JavaScript library for creating interactive charts.

TradingView Widget: An embedded widget for displaying real-time financial news, configured for increased size.

Conceptual Backend (Flask/Python): This frontend is designed to interact with a backend server (e.g., a Flask application written in Python) to fetch actual prediction data and historical stock information. (Note: The backend code is included as app.py in your repository, and its expected functionality is described below.)

🚀 Setup Instructions
To run this application:

Clone the Repository:

git clone <your-repo-url>
cd <your-repo-name>

Frontend Setup:

Open the index.html file using any modern web browser (e.g., Chrome, Firefox, Edge, Safari). You can usually do this by double-clicking the file.

Backend Setup (if you want to use prediction features):

Ensure you have Python and pip installed.

Install Flask and Flask-CORS:

pip install Flask Flask-Cors

Run the backend server from your terminal in the directory where app.py is located:

python app.py

The server should start on http://127.0.0.1:5000.

Backend (app.py - Reference from your Repository)
Your app.py handles the API endpoint for stock predictions. Here's a reminder of its conceptual structure and what it's designed to do:

# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import datetime
import yfinance as yf
import pandas as pd
import numpy as np # For numerical operations, useful for feature engineering

# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app) # Enable CORS for cross-origin requests from your frontend

# --- Helper Functions for Technical Indicators ---

def calculate_ema(data, span):
    """Calculates Exponential Moving Average."""
    return data.ewm(span=span, adjust=False).mean()

def calculate_macd(data_series, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculates MACD, Signal Line, and MACD Histogram.
    Returns: A dictionary with MACD components or None and an error message.
    """
    if len(data_series) < max(slow_period, fast_period) + signal_period:
        return None, "Not enough data for MACD calculation."

    ema_fast = calculate_ema(data_series, fast_period)
    ema_slow = calculate_ema(data_series, slow_period)
    macd = ema_fast - ema_slow
    signal_line = calculate_ema(macd, signal_period)
    macd_histogram = macd - signal_line

    return {
        "macd": macd.fillna(0).tolist(),
        "signal_line": signal_line.fillna(0).tolist(),
        "histogram": macd_histogram.fillna(0).tolist()
    }, None

def calculate_rsi(data_series, period=14):
    """
    Calculates the Relative Strength Index (RSI).
    Returns: A list of RSI values or None and an error message.
    """
    if len(data_series) < period * 2: # Need enough data for initial average calculation
        return None, "Not enough data for RSI calculation."

    delta = data_series.diff()
    gain = delta.where(delta > 0, 0).fillna(0) # Fill NaN from diff with 0
    loss = -delta.where(delta < 0, 0).fillna(0) # Fill NaN from diff with 0

    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()

    # Avoid division by zero for RS
    rs = avg_gain / avg_loss.replace(0, np.nan) # Replace 0 with NaN to avoid /0, then handle NaN in fillna
    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(0).tolist(), None # Fill NaN (including from division by zero) with 0, convert to list


def calculate_support_resistance(data_series, period=30):
    """
    Calculates simple support and resistance levels based on historical data.
    Support: Lowest closing price in the given period.
    Resistance: Highest closing price in the given period.
    """
    if len(data_series) < period:
        # If data is less than period, use available data
        support = data_series.min()
        resistance = data_series.max()
    else:
        support = data_series.tail(period).min()
        resistance = data_series.tail(period).max()
    return round(support, 2), round(resistance, 2)

# --- New: Exponential Growth Index (Conceptual) ---
def calculate_egi(data_series, period=20):
    """
    Calculates a conceptual Exponential Growth Index (EGI).
    This is a simplified approach:
    EGI = (Latest Price / EMA(period)) - 1
    A positive EGI suggests price is above its exponential average (growth trend).
    A negative EGI suggests price is below its exponential average (decline trend).
    """
    if len(data_series) < period:
        return np.nan, "Not enough data for EGI calculation."

    ema_val = calculate_ema(data_series, period).iloc[-1]
    if ema_val == 0: # Avoid division by zero
        return 0.0, None
    egi = ((data_series.iloc[-1] / ema_val) - 1) * 100 # Percentage difference
    return round(egi, 2), None

# --- Feature Engineering (Conceptual for ML Models) ---
def create_features(df):
    """
    Conceptual feature engineering function.
    In a real model, you'd create more sophisticated features here
    based on the historical stock data (df).
    These features would be the inputs to your trained ML models.
    """
    # Example features:
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['Daily_Change'] = df['Close'].diff()
    df['Volume_Change'] = df['Volume'].diff()

    # Calculate RSI
    rsi_list, rsi_error = calculate_rsi(df['Close'])
    if rsi_error:
        df['RSI_val'] = np.nan
        print(f"Warning: {rsi_error} for RSI calculation in features.")
    else:
        # Align RSI list back to DataFrame index for proper feature creation
        # RSI output starts from when it becomes valid, so it might be shorter than df
        df['RSI_val'] = pd.Series(rsi_list, index=df.index[len(df) - len(rsi_list):])
        df['RSI_val'] = df['RSI_val'].reindex(df.index, fill_value=np.nan)


    # Calculate EGI
    egi_val, egi_error = calculate_egi(df['Close'])
    if egi_error:
        df['EGI_val'] = np.nan
        print(f"Warning: {egi_error} for EGI calculation in features.")
    else:
        df['EGI_val'] = egi_val # This is a single value, will be the same for all rows after ffill

    # Fill NaN values created by rolling windows or diff for feature columns
    # Using .ffill() then .bfill() handles NaNs at both ends of the series
    # Ensure fillna is applied to the newly created feature columns only or the whole df after creation
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)


    # Select only the features that your model expects as input
    # Added 'EGI_val' to features
    features_df = df[['Close', 'Volume', 'SMA_5', 'SMA_10', 'Daily_Change', 'Volume_Change', 'RSI_val', 'EGI_val']]

    # Return the latest row of features, which would be used for prediction
    return features_df.iloc[-1].to_dict() # Return as a dictionary for easier handling

# --- Mock Prediction Functions (Placeholders for Real ML Models) ---
# In a real scenario, you would load your pre-trained models here
# For example:
# import joblib
# xgboost_model = joblib.load('path/to/your/xgboost_model.pkl')
# sklearn_model = joblib.load('path/to/your/sklearn_model.pkl')
# import torch
# from your_model_definition import LSTMModel # if you defined it in a separate file
# pytorch_model = LSTMModel(...) # Initialize with correct parameters
# pytorch_model.load_state_dict(torch.load('path/to/your/pytorch_model.pt'))
# pytorch_model.eval() # Set to evaluation mode for PyTorch

def predict_xgboost_mock(features_dict, days_ahead, latest_price):
    """
    MOCK prediction simulating an XGBoost model.
    """
    base_modifier = (features_dict.get('SMA_5', latest_price) - features_dict.get('SMA_10', latest_price)) / latest_price
    egi_impact = features_dict.get('EGI_val', 0) / 1000 # Small impact from EGI
    prediction = latest_price * (1 + base_modifier * 0.1 + egi_impact + (random.random() - 0.5) * (0.02 + days_ahead * 0.005))
    return round(prediction, 2)

def predict_sklearn_mock(features_dict, days_ahead, latest_price):
    """
    MOCK prediction simulating a scikit-learn model (e.g., Linear Regression, RandomForest).
    """
    daily_change_effect = features_dict.get('Daily_Change', 0) * (days_ahead / 5.0)
    egi_impact = features_dict.get('EGI_val', 0) / 500 # Slightly larger impact from EGI
    prediction = latest_price + daily_change_effect + egi_impact + (random.random() - 0.5) * (latest_price * 0.01 + days_ahead * 0.002)
    return round(prediction, 2)

def predict_lightgbm_mock(features_dict, days_ahead, latest_price):
    """
    MOCK prediction simulating a LightGBM model.
    """
    volume_effect = (features_dict.get('Volume_Change', 0) / features_dict.get('Volume', 1)) * 0.001
    egi_impact = features_dict.get('EGI_val', 0) / 750 # Medium impact from EGI
    prediction = latest_price * (1 + volume_effect + egi_impact + (random.random() - 0.5) * (0.015 + days_ahead * 0.004))
    return round(prediction, 2)

def predict_pytorch_mock(features_dict, days_ahead, latest_price):
    """
    MOCK prediction simulating a PyTorch (e.g., LSTM/RNN) model.
    """
    noise = (random.random() - 0.5) * (latest_price * 0.02)
    rsi_effect = (features_dict.get('RSI_val', 50) - 50) / 100 * (latest_price * 0.01)
    egi_effect = features_dict.get('EGI_val', 0) / 200 # Larger impact from EGI for "deep learning"
    pattern = np.sin(days_ahead / 7 * np.pi) * (latest_price * 0.005)
    prediction = latest_price + pattern + noise + rsi_effect + egi_effect + (random.uniform(-0.005, 0.005) * days_ahead * latest_price)
    return round(prediction, 2)

# --- Main Prediction Logic ---
def get_stock_data_and_prediction(symbol):
    """
    Fetches historical stock data using yfinance, calculates indicators,
    performs conceptual feature engineering, and generates mock predictions
    for 1, 7, and 30 days using various ML model placeholders.
    """
    try:
        ticker = yf.Ticker(symbol)
        # Fetch data for the last 120 days to ensure enough data for indicators and features
        # Ensure we get enough data for all rolling windows and indicators
        hist_df = ticker.history(period="120d") # Using 'hist_df' to indicate it's a DataFrame

        if hist_df.empty:
            return None, "No data found for the given symbol. Please check the symbol and try again."

        # Get the latest closing price
        latest_price = hist_df['Close'].iloc[-1]

        # Create features for the latest data point
        # Ensure 'hist_df' has enough rows for feature creation (e.g., for SMA_10)
        if len(hist_df) < 20: # Minimum for EGI, ensure enough history
            return None, "Not enough historical data (need at least 20 days) for robust feature engineering and indicator calculation."

        features_latest_dict = create_features(hist_df.copy()) # Pass a copy to avoid modifying original DataFrame

        # --- Generate Mock Predictions for 1, 7, and 30 days from each "model" ---
        predictions = {
            "xgboost": {},
            "sklearn": {},
            "lightgbm": {},
            "pytorch": {}
        }

        for days_ahead in [1, 7, 30]:
            # Each 'model' type gets the same latest features and latest price
            predictions["xgboost"][f"{days_ahead}day"] = predict_xgboost_mock(features_latest_dict, days_ahead, latest_price)
            predictions["sklearn"][f"{days_ahead}day"] = predict_sklearn_mock(features_latest_dict, days_ahead, latest_price)
            predictions["lightgbm"][f"{days_ahead}day"] = predict_lightgbm_mock(features_latest_dict, days_ahead, latest_price)
            predictions["pytorch"][f"{days_ahead}day"] = predict_pytorch_mock(features_latest_dict, days_ahead, latest_price)

        # Calculate support and resistance based on the last 30 closing prices
        support, resistance = calculate_support_resistance(hist_df['Close'], period=30)

        # Calculate MACD
        macd_data, macd_error = calculate_macd(hist_df['Close'])
        if macd_error:
            print(f"MACD Error for {symbol}: {macd_error}") # Log error, but proceed if possible
            macd_data = {"macd": [], "signal_line": [], "histogram": []} # Provide empty lists

        # Calculate RSI
        rsi_data, rsi_error = calculate_rsi(hist_df['Close'])
        if rsi_error:
            print(f"RSI Error for {symbol}: {rsi_error}") # Log error, but proceed if possible
            rsi_data = [] # Provide empty list

        # Prepare historical data for charting (dates and close prices)
        historical_data = []
        # Ensure dates correspond to MACD/RSI data points as they align
        chart_dates = hist_df.index.strftime("%Y-%m-%d").tolist()

        for index, row in hist_df.iterrows():
            historical_data.append({
                "date": index.strftime("%Y-%m-%d"),
                "price": round(row['Close'], 2)
            })

        # Determine the next prediction dates
        today = datetime.date.today()
        prediction_date_1day = (today + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        prediction_date_7day = (today + datetime.timedelta(days=7)).strftime("%Y-%m-%d")
        prediction_date_30day = (today + datetime.timedelta(days=30)).strftime("%Y-%m-%d")

        return {
            "symbol": symbol,
            "latest_price": round(latest_price, 2),
            "predictions": predictions, # Contains all model predictions for all days
            "prediction_date_1day": prediction_date_1day,
            "prediction_date_7day": prediction_date_7day,
            "prediction_date_30day": prediction_date_30day,
            "historical_data": historical_data,
            "chart_dates": chart_dates,
            "macd": macd_data,
            "rsi": rsi_data,
            "support": support,
            "resistance": resistance,
            "message": f"Data from yfinance, conceptual ML predictions and indicators for {symbol}"
        }, None

    except Exception as e:
        # Catch more general exceptions during yfinance fetch or processing
        return None, f"An unexpected error occurred while processing stock data: {str(e)}"

# --- Flask Routes ---
@app.route('/')
def home():
    """
    Home route for the Flask app.
    """
    return "Stock Prediction API. Use /predict to get predictions."

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to get stock predictions.
    Expects a JSON payload with 'symbol'.
    """
    data = request.get_json()
    if not data or 'symbol' not in data:
        return jsonify({"error": "Please provide a 'symbol' in the request body."}), 400

    symbol = data['symbol'].upper()
    prediction_data, error = get_stock_data_and_prediction(symbol)

    if error:
        return jsonify({"error": f"Failed to get stock data: {error}"}), 500
    else:
        return jsonify(prediction_data)

# --- Run the Flask App ---
if __name__ == '__main__':
    # To run this Flask app:
    # 1. Save this code as 'app.py' in your 'backend' folder (e.g., TRADING_BOT/backend/app.py)
    # 2. Open your terminal or command prompt
    # 3. Navigate to the 'backend' directory: cd TRADING_BOT/backend
    # 4. Install necessary Python libraries (if you haven't already):
    #    pip install Flask Flask-CORS yfinance pandas numpy
    # 5. Run the app: python app.py
    # The app will typically run on http://127.0.0.1:5000/
    app.run(debug=True)

💡 Usage
Prediction Tab (📈):

Enter a stock symbol (e.g., AAPL, GOOGL, TSLA).

Select an AI Model Type from the dropdown.

Click "Get Predictions & Charts" to see historical data, predicted prices, and technical indicator charts (MACD and RSI).

After getting a prediction, click "Add to Watchlist" to add the stock to your personal watchlist. A small modal will appear to ask for the quantity of shares.

Watchlist Tab (⭐):

View all stocks you've added to your watchlist, along with the quantity, average buy price, current value, and percentage change.

Each item now has a "Remove" button to easily delete it from your watchlist.

Initially, the watchlist will be empty until you add stocks.

A chart at the bottom shows the conceptual performance of your entire watchlist.

Analysis Tab (📊):

Initially, this tab will show a placeholder message.

Once you successfully get a prediction for a stock from the "Prediction" tab, this tab will automatically populate with mock financial metrics (P/E Ratio, EPS, Market Cap, etc.) and a company profile for that specific stock. It will no longer show default Apple data.

News Tab (📰):

This tab displays a comprehensive news feed powered by the TradingView widget, providing market headlines. The widget has been configured to take up more vertical space for a better viewing experience.

⚠️ Important Notes
Mock Data: The predictions, financial metrics in the Analysis tab, and watchlist performance are currently based on mock data within the frontend's JavaScript and the provided app.py backend for demonstration purposes. To get real-time, accurate predictions and financial data, you would need to integrate with real financial data APIs and potentially train more sophisticated machine learning models.

Backend Connection: The "Get Predictions & Charts" button relies on your app.py backend running on http://127.0.0.1:5000. If app.py is not running, this feature will show an error, but the rest of the frontend (tabs, watchlist management, news) will still be functional with its mock data where applicable.
