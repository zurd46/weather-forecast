import requests
import pandas as pd
import numpy as np
import torch
import gradio as gr
import joblib
import onnxruntime as ort
from sklearn.preprocessing import MinMaxScaler

# OpenWeatherMap API-Schlüssel (ersetzen Sie durch Ihren eigenen Schlüssel)
API_KEY = 'your-api-key'

# Funktion zum Abrufen der aktuellen Temperatur von Luzern
def get_current_temperature(city='Luzern', country='CH'):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},{country}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data['main']['temp']
    else:
        raise ValueError(f"Fehler beim Abrufen der Wetterdaten: {response.status_code}")

# Funktion zum Abrufen der aktuellen Wetterdaten von Luzern
def get_current_weather(city='Luzern', country='CH'):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},{country}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {
            'temp': data['main']['temp'],
            'pressure': data['main']['pressure'],
            'humidity': data['main']['humidity'],
            'clouds': data['clouds']['all'],
            'wind_speed': data['wind']['speed'],
        }
    else:
        raise ValueError(f"Fehler beim Abrufen der Wetterdaten: {response.status_code}")

# Funktion zur Vorhersage mit ONNX-Modellen
def make_prediction(sequence, model_path):
    input_size = sequence.shape[1]
    sequence = scaler.transform(sequence.reshape(-1, input_size))
    sequence = np.tile(sequence, (24, 1)).astype(np.float32)  # Sicherstellen, dass die Sequenz die richtige Länge hat
    sequence = np.expand_dims(sequence, axis=0)  # Hinzufügen einer Batch-Dimension
    
    session = ort.InferenceSession(model_path)
    inputs = {session.get_inputs()[0].name: sequence}
    pred = session.run(None, inputs)[0]
    
    pred = scaler.inverse_transform(pred.reshape(-1, input_size))
    return pred

# Lade den Scaler
scaler = joblib.load('model/scaler.save')

# Gradio-Interface
def forecast_weather():
    # Erhalte die aktuellen Wetterdaten von Luzern
    current_weather = get_current_weather()
    
    # Erstelle die Eingabesequenz
    sequence = np.array([[
        current_weather['temp'],
        current_weather['pressure'],
        current_weather['humidity'],
        current_weather['clouds'],
        current_weather['wind_speed']
    ]])
    
    # Vorhersagen
    pred_3h = make_prediction(sequence, 'model/lstm_model_3h.onnx')[-1]
    pred_6h = make_prediction(sequence, 'model/lstm_model_6h.onnx')[-1]
    pred_12h = make_prediction(sequence, 'model/lstm_model_12h.onnx')[-1]
    pred_24h = make_prediction(sequence, 'model/lstm_model_24h.onnx')[-1]
    
    # Aktuelle Werte
    current_temp = current_weather['temp']

    # Ergebnis formatieren
    results = {
        "3 Stunden Vorhersage": {
            "Temperatur": f'{current_temp + pred_3h[0]:.2f} Grad Celsius',
            "Druck": f'{pred_3h[1]:.2f} hPa',
            "Luftfeuchtigkeit": f'{pred_3h[2]:.2f} %',
            "Bewölkung": f'{pred_3h[3]:.2f} %',
            "Windgeschwindigkeit": f'{pred_3h[4]:.2f} m/s'
        },
        "6 Stunden Vorhersage": {
            "Temperatur": f'{current_temp + pred_6h[0]:.2f} Grad Celsius',
            "Druck": f'{pred_6h[1]:.2f} hPa',
            "Luftfeuchtigkeit": f'{pred_6h[2]:.2f} %',
            "Bewölkung": f'{pred_6h[3]:.2f} %',
            "Windgeschwindigkeit": f'{pred_6h[4]:.2f} m/s'
        },
        "12 Stunden Vorhersage": {
            "Temperatur": f'{current_temp + pred_12h[0]:.2f} Grad Celsius',
            "Druck": f'{pred_12h[1]:.2f} hPa',
            "Luftfeuchtigkeit": f'{pred_12h[2]:.2f} %',
            "Bewölkung": f'{pred_12h[3]:.2f} %',
            "Windgeschwindigkeit": f'{pred_12h[4]:.2f} m/s'
        },
        "24 Stunden Vorhersage": {
            "Temperatur": f'{current_temp + pred_24h[0]:.2f} Grad Celsius',
            "Druck": f'{pred_24h[1]:.2f} hPa',
            "Luftfeuchtigkeit": f'{pred_24h[2]:.2f} %',
            "Bewölkung": f'{pred_24h[3]:.2f} %',
            "Windgeschwindigkeit": f'{pred_24h[4]:.2f} m/s'
        }
    }
    return results

gr.Interface(
    fn=forecast_weather, 
    inputs=[], 
    outputs=gr.JSON(label="Vorhersagen"), 
    title="Wettervorhersage", 
    description="Vorhersage des Wetters für die nächsten 3, 6, 12 und 24 Stunden."
).launch()

# Ausgabe der Vorhersagen mit der aktuellen Temperatur
print(f'Vorhersage für 3 Stunden: Temperatur: {current_temp + pred_3h[0]:.2f} Grad Celsius, Druck: {pred_3h[1]:.2f} hPa, Luftfeuchtigkeit: {pred_3h[2]:.2f} %, Bewölkung: {pred_3h[3]:.2f} %, Windgeschwindigkeit: {pred_3h[4]:.2f} m/s')
print(f'Vorhersage für 6 Stunden: Temperatur: {current_temp + pred_6h[0]:.2f} Grad Celsius, Druck: {pred_6h[1]:.2f} hPa, Luftfeuchtigkeit: {pred_6h[2]:.2f} %, Bewölkung: {pred_6h[3]:.2f} %, Windgeschwindigkeit: {pred_6h[4]:.2f} m/s')
print(f'Vorhersage für 12 Stunden: Temperatur: {current_temp + pred_12h[0]:.2f} Grad Celsius, Druck: {pred_12h[1]:.2f} hPa, Luftfeuchtigkeit: {pred_12h[2]:.2f} %, Bewölkung: {pred_12h[3]:.2f} %, Windgeschwindigkeit: {pred_12h[4]:.2f} m/s')
print(f'Vorhersage für 24 Stunden: Temperatur: {current_temp + pred_24h[0]:.2f} Grad Celsius, Druck: {pred_24h[1]:.2f} hPa, Luftfeuchtigkeit: {pred_24h[2]:.2f} %, Bewölkung: {pred_24h[3]:.2f} %, Windgeschwindigkeit: {pred_24h[4]:.2f} m/s')
