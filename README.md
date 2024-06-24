# Weather Forecasting with LSTM

This project demonstrates the use of a Long Short-Term Memory (LSTM) neural network to forecast weather data, including temperature, pressure, humidity, and weather description (e.g., sunny, cloudy) for the next 6, 12, and 24 hours based on historical data.

## Project Structure

- `data/luzern.jsonl`: The JSONL file containing historical weather data.
- `weather_forecasting.py`: The main Python script for training and evaluating the LSTM model.
- `label_encoder.pkl`: Saved LabelEncoder for the weather descriptions.
- `scaler.pkl`: Saved MinMaxScaler for normalizing the data.
- `lstm-weather.pth`: Saved PyTorch model.
- `lstm-weather.safetensors`: Saved model in safetensors format.

## Dependencies

- pandas
- numpy
- scikit-learn
- torch
- safetensors
- joblib

You can install the required dependencies using the following command:

```bash
pip install pandas numpy scikit-learn torch safetensors joblib
