# Weather Forecasting with LSTM

This project demonstrates the use of a Long Short-Term Memory (LSTM) neural network to forecast weather data, including temperature, pressure, humidity, and weather description (e.g., sunny, cloudy) for the next 3, 6, 12, and 24 hours based on historical data.

## Project Structure

- `data/`: The JSONL files containing historical weather data.
- `lstm.py`: The main Python script for training and evaluating the LSTM model.
- `/model`: Folder for the trained model (onnx)

## Dependencies

- pandas
- numpy
- scikit-learn
- torch
- safetensors
- joblib
- gradio

You can install the required dependencies using the following command:

```bash
pip install pandas numpy scikit-learn torch safetensors joblib gradio
