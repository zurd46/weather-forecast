import requests
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
import torch.optim as optim
import os
import jsonlines
import joblib

# OpenWeatherMap API-Schlüssel (ersetzen Sie durch Ihren eigenen Schlüssel)
API_KEY = 'your-api.key'

# Funktion zum Abrufen der aktuellen Temperatur von Luzern
def get_current_temperature(city='Luzern', country='CH'):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},{country}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data['main']['temp']
    else:
        raise ValueError(f"Fehler beim Abrufen der Wetterdaten: {response.status_code}")

# Funktion zum Lesen von JSONL-Dateien
def read_jsonl(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Die Datei {file_path} existiert nicht.")
    
    try:
        data = []
        with jsonlines.open(file_path) as reader:
            for obj in reader:
                data.append(obj)
        print(f"Daten von {file_path} erfolgreich geladen.")
    except Exception as e:
        raise ValueError(f"Fehler beim Lesen der Datei {file_path}: {e}")
    
    return data

# Lade und extrahiere relevante Daten aus JSONL
def load_data(file_path):
    data = read_jsonl(file_path)
    # Extrahiere relevante Features
    extracted_data = []
    for entry in data:
        main_info = entry.get('main', {})
        clouds_info = entry.get('clouds', {})
        wind_info = entry.get('wind', {})
        extracted_data.append([
            main_info.get('temp', 0),
            main_info.get('pressure', 0),
            main_info.get('humidity', 0),
            clouds_info.get('all', 0),
            wind_info.get('speed', 0)
        ])
    
    return np.array(extracted_data)

# Normalisieren der Daten
def normalize_data(data):
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data)
    return data_normalized, scaler

# Erstellen von Sequenzen
def create_sequences(data, seq_length, pred_steps):
    X, y = [], []
    for i in range(len(data) - seq_length - pred_steps + 1):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length + pred_steps - 1] - data[i + seq_length - 1])  # Differenz zum aktuellen Wert
    return np.array(X), np.array(y)

# Laden der Daten
file_path = './data/luzern.jsonl'
data = load_data(file_path)

# Normalisieren der Daten
data_normalized, scaler = normalize_data(data)

# Festlegen der Sequenzlänge und Erstellen von Sequenzen für verschiedene Vorhersageschritte
seq_length = 24  # Sequenzlänge von 24 Stunden

# Überprüfen, ob genügend Daten vorhanden sind, um mindestens eine Sequenz zu erstellen
if len(data_normalized) < seq_length + 24:
    raise ValueError(f"Nicht genügend Datenpunkte, um eine Sequenz von Länge {seq_length} und Vorhersageschritte zu erstellen. "
                     f"Erforderlich: {seq_length + 24}, Vorhanden: {len(data_normalized)}")

X_3h, y_3h = create_sequences(data_normalized, seq_length, 3)
X_6h, y_6h = create_sequences(data_normalized, seq_length, 6)
X_12h, y_12h = create_sequences(data_normalized, seq_length, 12)
X_24h, y_24h = create_sequences(data_normalized, seq_length, 24)

# In Tensoren umwandeln
X_3h = torch.tensor(X_3h, dtype=torch.float32)
y_3h = torch.tensor(y_3h, dtype=torch.float32)
X_6h = torch.tensor(X_6h, dtype=torch.float32)
y_6h = torch.tensor(y_6h, dtype=torch.float32)
X_12h = torch.tensor(X_12h, dtype=torch.float32)
y_12h = torch.tensor(y_12h, dtype=torch.float32)
X_24h = torch.tensor(X_24h, dtype=torch.float32)
y_24h = torch.tensor(y_24h, dtype=torch.float32)

# Debug: Ausgabe der Anzahl der Sequenzen
print(f'Anzahl der Sequenzen für 3 Stunden: {len(X_3h)}')
print(f'Anzahl der Sequenzen für 6 Stunden: {len(X_6h)}')
print(f'Anzahl der Sequenzen für 12 Stunden: {len(X_12h)}')
print(f'Anzahl der Sequenzen für 24 Stunden: {len(X_24h)}')

# Erstellen von Datasets und DataLoadern
def create_data_loaders(X, y, batch_size=32):
    dataset = TensorDataset(X, y)
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

train_loader_3h, test_loader_3h = create_data_loaders(X_3h, y_3h)
train_loader_6h, test_loader_6h = create_data_loaders(X_6h, y_6h)
train_loader_12h, test_loader_12h = create_data_loaders(X_12h, y_12h)
train_loader_24h, test_loader_24h = create_data_loaders(X_24h, y_24h)

# Modelldefinition
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Hyperparameter und Geräteauswahl
input_size = X_3h.shape[2]  # Anzahl der Features in den Sequenzen
hidden_size = 50
output_size = y_3h.shape[1]  # Anzahl der Output-Features
num_layers = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Funktion zum Trainieren des Modells
def train_model(train_loader, num_epochs=25, max_steps=3000):
    model = LSTMModel(input_size, hidden_size, output_size, num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    total_steps = 0
    for epoch in range(num_epochs):
        if total_steps >= max_steps:
            break
        model.train()
        epoch_loss = 0.0
        for i, (sequences, targets) in enumerate(train_loader):
            sequences, targets = sequences.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            total_steps += 1
            if total_steps >= max_steps:
                break

            print(f'Step [{total_steps}], Loss: {loss.item():.4f}')

        epoch_loss /= (i + 1)
        print(f'Epoch [{epoch+1}/{num_epochs}], Verlust: {epoch_loss:.4f}')
    
    return model

# Trainieren der Modelle für verschiedene Vorhersagezeiträume
model_3h = train_model(train_loader_3h)
model_6h = train_model(train_loader_6h)
model_12h = train_model(train_loader_12h)
model_24h = train_model(train_loader_24h)

# Funktion zur Vorhersage und Rücktransformation
def make_predictions(model, input_sequence, scaler, current_values):
    model.eval()
    with torch.no_grad():
        input_sequence = input_sequence.unsqueeze(0).to(device)
        predicted_sequence = model(input_sequence).cpu().numpy()
        predicted_sequence = scaler.inverse_transform(predicted_sequence)
        return current_values + predicted_sequence[0]  # Addition der aktuellen Werte

# Beispielvorhersagen für 3h, 6h, 12h und 24h
input_sequence = X_3h[0]  # Beispiel-Eingabesequenz

# Aktuelle Temperatur von Luzern abrufen
current_temp = get_current_temperature()
print(f'Aktuelle Temperatur in Luzern: {current_temp:.2f} Grad Celsius')

# Extrahiere die aktuellen Werte
current_values = data[-1]  # Aktuelle Werte

# Vorhersagen mit den Differenzwerten machen und die aktuelle Temperatur hinzufügen
prediction_3h = make_predictions(model_3h, input_sequence, scaler, current_values)
prediction_6h = make_predictions(model_6h, input_sequence, scaler, current_values)
prediction_12h = make_predictions(model_12h, input_sequence, scaler, current_values)
prediction_24h = make_predictions(model_24h, input_sequence, scaler, current_values)

# Ausgabe der Vorhersagen mit der aktuellen Temperatur
print(f'Vorhersage für 3 Stunden: Temperatur: {current_temp + prediction_3h[0]:.2f} Grad Celsius, Druck: {prediction_3h[1]:.2f} hPa, Luftfeuchtigkeit: {prediction_3h[2]:.2f} %, Bewölkung: {prediction_3h[3]:.2f} %, Windgeschwindigkeit: {prediction_3h[4]:.2f} m/s')
print(f'Vorhersage für 6 Stunden: Temperatur: {current_temp + prediction_6h[0]:.2f} Grad Celsius, Druck: {prediction_6h[1]:.2f} hPa, Luftfeuchtigkeit: {prediction_6h[2]:.2f} %, Bewölkung: {prediction_6h[3]:.2f} %, Windgeschwindigkeit: {prediction_6h[4]:.2f} m/s')
print(f'Vorhersage für 12 Stunden: Temperatur: {current_temp + prediction_12h[0]:.2f} Grad Celsius, Druck: {prediction_12h[1]:.2f} hPa, Luftfeuchtigkeit: {prediction_12h[2]:.2f} %, Bewölkung: {prediction_12h[3]:.2f} %, Windgeschwindigkeit: {prediction_12h[4]:.2f} m/s')
print(f'Vorhersage für 24 Stunden: Temperatur: {current_temp + prediction_24h[0]:.2f} Grad Celsius, Druck: {prediction_24h[1]:.2f} hPa, Luftfeuchtigkeit: {prediction_24h[2]:.2f} %, Bewölkung: {prediction_24h[3]:.2f} %, Windgeschwindigkeit: {prediction_24h[4]:.2f} m/s')

# Erstellen des Verzeichnisses für das Modell, falls es nicht existiert
model_dir = 'model'
os.makedirs(model_dir, exist_ok=True)

# Speichern der Modelle als ONNX
onnx_model_path_3h = os.path.join(model_dir, 'lstm_model_3h.onnx')
dummy_input = torch.randn(1, seq_length, input_size, device=device)
torch.onnx.export(model_3h, dummy_input, onnx_model_path_3h, input_names=['input'], output_names=['output'], dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

onnx_model_path_6h = os.path.join(model_dir, 'lstm_model_6h.onnx')
torch.onnx.export(model_6h, dummy_input, onnx_model_path_6h, input_names=['input'], output_names=['output'], dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

onnx_model_path_12h = os.path.join(model_dir, 'lstm_model_12h.onnx')
torch.onnx.export(model_12h, dummy_input, onnx_model_path_12h, input_names=['input'], output_names=['output'], dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

onnx_model_path_24h = os.path.join(model_dir, 'lstm_model_24h.onnx')
torch.onnx.export(model_24h, dummy_input, onnx_model_path_24h, input_names=['input'], output_names=['output'], dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

print(f'Modelle erfolgreich als ONNX gespeichert.')

# Speichern des Scalers für die zukünftige Verwendung
scaler_file = os.path.join(model_dir, 'scaler.save')
joblib.dump(scaler, scaler_file)
print(f'Scaler erfolgreich gespeichert als {scaler_file}.')
