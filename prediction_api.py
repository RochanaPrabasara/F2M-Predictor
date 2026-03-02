from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from prophet import Prophet

app = Flask(__name__)
CORS(app)  # Enable CORS for Node.js to access

# Load models on startup
print("Loading models...")
prophet_models = joblib.load('prophet_models.pkl')
rf_models = joblib.load('rf_models.pkl')
le_commodity = joblib.load('commodity_encoder.pkl')
le_region = joblib.load('region_encoder.pkl')
df_clean = joblib.load('cleaned_data.pkl')
print("✓ Models loaded successfully!")

# Features used in Random Forest
rf_features = ['Temperature (°C)', 'Rainfall (mm)', 'Humidity (%)', 
               'Crop Yield Impact Score', 'Month', 'Week', 'Quarter',
               'Price_Lag_1Week', 'Price_Lag_2Week', 'Price_Lag_4Week',
               'Price_MA_4Week', 'Commodity_Encoded', 'Region_Encoded']

rf_features = [f for f in rf_features if f in df_clean.columns]

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(rf_models),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict vegetable prices
    
    Request body:
    {
        "commodity": "Carrot",
        "region": "Nuwara Eliya",
        "weeks_ahead": 4,
        "weather": {
            "temperature": 25,
            "rainfall": 100,
            "humidity": 75,
            "crop_yield_impact": 5
        }
    }
    """
    try:
        data = request.get_json()
        
        commodity = data.get('commodity')
        region = data.get('region')
        weeks_ahead = data.get('weeks_ahead', 4)
        weather_data = data.get('weather', None)
        
        # Validate inputs
        if not commodity or not region:
            return jsonify({
                'success': False,
                'error': 'commodity and region are required'
            }), 400
        
        if weeks_ahead < 1 or weeks_ahead > 4:
            return jsonify({
                'success': False,
                'error': 'weeks_ahead must be between 1 and 4'
            }), 400
        
        # Make prediction
        model_key = f"{commodity}_{region}"
        
        if model_key not in prophet_models or model_key not in rf_models:
            return jsonify({
                'success': False,
                'error': f'No model available for {commodity} in {region}'
            }), 404
        
        # Get models
        prophet_model = prophet_models[model_key]
        rf_model = rf_models[model_key]
        
        # Get historical data for this commodity-region
        historical_data = df_clean[
            (df_clean['Commodity'] == commodity) & 
            (df_clean['Region'] == region)
        ].copy()
        
        # Get last date
        last_date = historical_data['Date'].max()
        
        # Create future dates
        future_dates = pd.date_range(
            start=last_date + timedelta(days=7), 
            periods=weeks_ahead, 
            freq='W'
        )
        
        predictions = []
        
        # Keep track of recent prices (start with last 4 historical prices)
        recent_prices = historical_data['Price'].tail(4).tolist()
        
        for i, future_date in enumerate(future_dates):
            week_num = i + 1
            
            # Prophet prediction
            future_df = pd.DataFrame({'ds': [future_date]})
            prophet_pred = prophet_model.predict(future_df)['yhat'].values[0]
            
            # Prepare weather features
            if weather_data is None:
                recent_data = df_clean[
                    (df_clean['Commodity'] == commodity) & 
                    (df_clean['Region'] == region)
                ].tail(4)
                
                weather_features = {
                    'Temperature (°C)': recent_data['Temperature (°C)'].mean() if 'Temperature (°C)' in recent_data.columns else 25,
                    'Rainfall (mm)': recent_data['Rainfall (mm)'].mean() if 'Rainfall (mm)' in recent_data.columns else 100,
                    'Humidity (%)': recent_data['Humidity (%)'].mean() if 'Humidity (%)' in recent_data.columns else 75,
                    'Crop Yield Impact Score': recent_data['Crop Yield Impact Score'].mean() if 'Crop Yield Impact Score' in recent_data.columns else 5,
                }
            else:
                weather_features = {
                    'Temperature (°C)': weather_data.get('temperature', 25),
                    'Rainfall (mm)': weather_data.get('rainfall', 100),
                    'Humidity (%)': weather_data.get('humidity', 75),
                    'Crop Yield Impact Score': weather_data.get('crop_yield_impact', 5),
                }
            
            # Create RF feature vector using UPDATED lag prices
            X_future = pd.DataFrame([{
                'Temperature (°C)': weather_features['Temperature (°C)'],
                'Rainfall (mm)': weather_features['Rainfall (mm)'],
                'Humidity (%)': weather_features['Humidity (%)'],
                'Crop Yield Impact Score': weather_features['Crop Yield Impact Score'],
                'Month': future_date.month,
                'Week': future_date.isocalendar()[1],
                'Quarter': (future_date.month - 1) // 3 + 1,
                'Price_Lag_1Week': recent_prices[-1] if len(recent_prices) >= 1 else prophet_pred,
                'Price_Lag_2Week': recent_prices[-2] if len(recent_prices) >= 2 else prophet_pred,
                'Price_Lag_4Week': recent_prices[-4] if len(recent_prices) >= 4 else prophet_pred,
                'Price_MA_4Week': np.mean(recent_prices[-4:]) if len(recent_prices) >= 4 else prophet_pred,
                'Commodity_Encoded': le_commodity.transform([commodity])[0],
                'Region_Encoded': le_region.transform([region])[0]
            }])
            
            X_future = X_future[rf_features]
            
            # RF prediction
            rf_pred = rf_model.predict(X_future)[0]
            
            # Hybrid prediction (60% Prophet, 40% RF)
            hybrid_pred = 0.6 * prophet_pred + 0.4 * rf_pred
            
            predictions.append({
                'week': week_num,
                'date': future_date.strftime('%Y-%m-%d'),
                'predicted_price': round(hybrid_pred, 2),
                'confidence_range': {
                    'low': round(hybrid_pred * 0.9, 2),
                    'high': round(hybrid_pred * 1.1, 2)
                }
            })
            
            # UPDATE: Add this week's hybrid prediction to recent_prices for next iteration
            recent_prices.append(hybrid_pred)
            # Keep only last 4 prices
            if len(recent_prices) > 4:
                recent_prices.pop(0)
        
        return jsonify({
            'success': True,
            'commodity': commodity,
            'region': region,
            'predictions': predictions,
            'current_date': datetime.now().strftime('%Y-%m-%d')
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/available-models', methods=['GET'])
def available_models():
    """Get list of available commodity-region combinations"""
    models = []
    for key in rf_models.keys():
        commodity, region = key.rsplit('_', 1)
        models.append({
            'commodity': commodity,
            'region': region
        })
    
    return jsonify({
        'success': True,
        'models': models,
        'total': len(models)
    })

@app.route('/commodities', methods=['GET'])
def get_commodities():
    """Get list of all commodities"""
    commodities = df_clean['Commodity'].unique().tolist()
    return jsonify({
        'success': True,
        'commodities': commodities
    })

@app.route('/regions', methods=['GET'])
def get_regions():
    """Get list of all regions"""
    regions = df_clean['Region'].unique().tolist()
    return jsonify({
        'success': True,
        'regions': regions
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)