import pandas as pd
import numpy as np
import torch
import gradio as gr
import joblib
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
import torch.optim as optim

# Modelldefinition
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size * 24)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = out.view(out.size(0), 24, -1)
        return out

# Lade das trainierte Modell
input_size = 10
hidden_size = 50
output_size = 10
num_layers = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel(input_size, hidden_size, output_size, num_layers).to(device)
model.load_state_dict(torch.load('lstm-weather.pth', map_location=device))
model.eval()

# Lade LabelEncoder und Scaler
label_encoder = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')

# Funktion zur Vorhersage
def make_prediction(sequence):
    sequence = np.array(sequence).reshape(-1, 10)
    sequence[:, :-1] = scaler.transform(sequence[:, :-1])
    sequence = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(sequence)
    prediction = prediction.squeeze(0).cpu().numpy()
    prediction[:, :-1] = scaler.inverse_transform(prediction[:, :-1])
    weather_descriptions = label_encoder.inverse_transform(prediction[:, -1].astype(int))
    return prediction, weather_descriptions

# Gradio-Interface
def forecast_weather(temp, pressure, humidity, temp_min, temp_max, feels_like, clouds, wind_speed, wind_deg, weather_desc):
    # Erstelle die Eingabesequenz
    weather_desc_encoded = label_encoder.transform([weather_desc])[0]
    sequence = [[temp, pressure, humidity, temp_min, temp_max, feels_like, clouds, wind_speed, wind_deg, weather_desc_encoded]]

    # Vorhersagen
    pred_6h, weather_6h = make_prediction(sequence * 6)
    pred_12h, weather_12h = make_prediction(sequence * 12)
    pred_24h, weather_24h = make_prediction(sequence * 24)

    # Ergebnis formatieren
    results = {
        "6 Stunden Vorhersage": f'{pred_6h}, Wetter: {weather_6h}',
        "12 Stunden Vorhersage": f'{pred_12h}, Wetter: {weather_12h}',
        "24 Stunden Vorhersage": f'{pred_24h}, Wetter: {weather_24h}'
    }
    return results

inputs = [
    gr.inputs.Number(label="Temperatur"),
    gr.inputs.Number(label="Druck"),
    gr.inputs.Number(label="Luftfeuchtigkeit"),
    gr.inputs.Number(label="Min Temperatur"),
    gr.inputs.Number(label="Max Temperatur"),
    gr.inputs.Number(label="Gefühlte Temperatur"),
    gr.inputs.Number(label="Bewölkung (%)"),
    gr.inputs.Number(label="Windgeschwindigkeit"),
    gr.inputs.Number(label="Windrichtung (Grad)"),
    gr.inputs.Textbox(label="Wetterbeschreibung")
]

outputs = gr.outputs.JSON(label="Vorhersagen")

gr.Interface(fn=forecast_weather, inputs=inputs, outputs=outputs, title="Wettervorhersage", description="Vorhersage des Wetters für die nächsten 6, 12 und 24 Stunden.").launch()
