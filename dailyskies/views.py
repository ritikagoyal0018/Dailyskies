from django.shortcuts import render
import requests
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from datetime import datetime, timedelta
import pickle
import pytz

API_Key = 'a8fc3cc4cb073a415a61ffa4aee465be'
Base_URL = 'https://api.openweathermap.org/data/2.5/weather?'

def correct_city_name(user_input):
    return user_input.title()

def get_current_weather(city):
    url = f"{Base_URL}q={city}&appid={API_Key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        main = data.get('main', {})
        weather = data.get('weather', [{}])[0]
        wind = data.get('wind', {})
        return {
            'city': data.get('name', ''),
            'temperature': main.get('temp', 300),
            'feels_like': main.get('feels_like', 300),
            'temp_min': main.get('temp_min', 295),
            'temp_max': main.get('temp_max', 305),
            'humidity': main.get('humidity', 50),
            'description': weather.get('description', 'Clear sky'),
            'country': data['sys'].get('country', ''),
            'wind_gust_dir': wind.get('deg', 0),
            'wind_gust_speed': wind.get('speed', 0),
            'pressure': main.get('pressure', 1013),
            'visibility': data.get('visibility', 10000)
        }
    return None

def load_models():
    def load_pkl(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    rain_model = load_pkl(os.path.join(base_dir, 'models', 'rain_model.pkl'))
    temp_model = load_pkl(os.path.join(base_dir, 'models', 'temp_model.pkl'))
    hum_model = load_pkl(os.path.join(base_dir, 'models', 'humidity_model.pkl'))
    scaler = load_pkl(os.path.join(base_dir, 'models', 'scaler.pkl'))
    le = load_pkl(os.path.join(base_dir, 'models', 'wind_dir_encoder.pkl'))
    return rain_model, temp_model, hum_model, scaler, le

def to_celsius(kelvin):
    return float(kelvin) - 273.15 if kelvin is not None else None

def weatherview(request):
    city = 'Delhi'
    if request.method == 'POST':
        city = correct_city_name(request.POST.get('city', 'Delhi'))

    current_weather = get_current_weather(city)
    if not current_weather:
        return render(request, 'weather.html', {'error': 'City not found or API error.'})

    rain_model, temp_model, hum_model, scaler, le = load_models()

    wind_deg = current_weather['wind_gust_dir'] % 360
    compass_points = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                      'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    wind_direction = compass_points[int((wind_deg + 11.25) / 22.5) % 16]
    compass_dir_encoded = le.transform([wind_direction])[0] if wind_direction in le.classes_ else 0

    feature_order = ['MinTemp', 'MaxTemp', 'Temp', 'WindGustDir', 'WindGustSpeed', 'Humidity', 'Pressure']
    row = [
        to_celsius(current_weather['temp_min']),
        to_celsius(current_weather['temp_max']),
        to_celsius(current_weather['temperature']),
        compass_dir_encoded,
        current_weather['wind_gust_speed'],
        current_weather['humidity'],
        current_weather['pressure']
    ]
    df_input = pd.DataFrame([row], columns=feature_order)

    rain_pred = rain_model.predict(df_input)[0]

    # Predict temperature for next 7 steps
    temp_features = row.copy()
    temp_preds = []
    for _ in range(7):
        input_array = np.array([temp_features])
        input_array = scaler.transform(input_array)
        pred = temp_model.predict(input_array)[0]
        temp_preds.append(pred)
        temp_features[2] = pred

    # Predict humidity for next 7 steps
    hum_features = row.copy()
    hum_preds = []
    for _ in range(7):
        input_array = np.array([hum_features])
        input_array = scaler.transform(input_array)
        pred = hum_model.predict(input_array)[0]
        hum_preds.append(pred)
        hum_features[5] = pred

    future_rain = ['Yes' if rain_pred == 1 else 'No'] * 7

    now = datetime.now(pytz.timezone('Asia/Kolkata'))
    future_times = [now + timedelta(hours=i+1) for i in range(7)]
    future_data = list(zip(future_times, temp_preds, hum_preds, future_rain))

    context = {
        'location': current_weather['city'],
        'country': current_weather['country'],
        'temperature': to_celsius(current_weather['temperature']),
        'feels_like': to_celsius(current_weather['feels_like']),
        'temp_min': to_celsius(current_weather['temp_min']),
        'temp_max': to_celsius(current_weather['temp_max']),
        'humidity': current_weather['humidity'],
        'description': current_weather['description'],
        'time': now.strftime('%H:%M %p'),
        'date': now.strftime('%B %d, %Y'),
        'wind': wind_direction,
        'wind_speed': current_weather['wind_gust_speed'],
        'pressure': current_weather['pressure'],
        'visibility': current_weather['visibility'],
        'rain_prediction': 'Yes' if rain_pred == 1 else 'No',
        'future_data': future_data
    }

    return render(request, 'weather.html', context)
