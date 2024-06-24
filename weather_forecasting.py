import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
import torch.optim as optim
import os
import safetensors.torch

# Funktion zum Lesen von JSONL-Dateien
def read_jsonl(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Die Datei {file_path} existiert nicht.")
    
    try:
        df = pd.read_json(file_path, lines=True)
    except ValueError as e:
        raise ValueError(f"Fehler beim Lesen der Datei {file_path}: {e}")
    
    return df

# Daten laden
df = read_jsonl('./data/luzern.jsonl')

# Wähle relevante Spalten aus und erstelle DataFrame
data = df[['main.temp', 'main.pressure', 'main.humidity']].values

# Normalisieren der Daten
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)

# Daten in Sequenzen umwandeln
def create_sequences(data, seq_length, pred_length=24):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length - pred_length):
        seq = data[i:i + seq_length]
        label = data[i + seq_length:i + seq_length + pred_length]  # Vorhersage für die nächsten 24 Stunden
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

seq_length = 24
sequences, labels = create_sequences(data_normalized, seq_length)

# In Tensoren umwandeln
sequences = torch.tensor(sequences, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.float32)

# Erstellen von Dataset und DataLoader
dataset = TensorDataset(sequences, labels)
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Modelldefinition
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size * 24)  # Multipliziere mit 24 für 24-Stunden-Vorhersage
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = out.view(out.size(0), 24, -1)  # Forme die Ausgabe um
        return out

input_size = 3  # Anzahl der Merkmale (Temp, Druck, Luftfeuchtigkeit)
hidden_size = 50
output_size = 3  # Vorhersage von Temp, Druck und Luftfeuchtigkeit
num_layers = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel(input_size, hidden_size, output_size, num_layers).to(device)

# Hyperparameter
num_epochs = 100
learning_rate = 0.001

# Loss und Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for sequences, labels in train_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    epoch_loss /= len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# Modell speichern
torch.save(model.state_dict(), './lstm-weather.pth')

# Modell laden und evaluieren
model.load_state_dict(torch.load('./lstm-weather.pth'))
model.eval()

# Evaluierung auf Testdaten
test_loss = 0.0
with torch.no_grad():
    for sequences, labels in test_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

test_loss /= len(test_loader)
print(f'Test Loss: {test_loss:.4f}')

# Beispielhafte Vorhersage
with torch.no_grad():
    example_data = sequences[0].unsqueeze(0).to(device)  # Beispielsequenz
    prediction = model(example_data)
    print(f'Predicted: {prediction.cpu().numpy()}, Actual: {labels[0].cpu().numpy()}')

# Speichern des Modells als safetensors
safetensors.torch.save_file(model.state_dict(), 'lstm-weather.safetensors')
