# Stock Price Prediction using LSTM

## Overview

This project predicts stock prices using a Long Short-Term Memory (LSTM) neural network. The model is built with TensorFlow/Keras and trained on historical stock data. The predictions and actual prices are saved to an Excel file, along with a plot of the stock's closing prices.

## Project Structure

- `preprocess_data()`: Preprocesses stock data for a given company.
- `build_model()`: Constructs and compiles the LSTM model.
- `train_model()`: Trains the LSTM model.
- `predict_stock()`: Predicts stock prices and calculates performance metrics.
- `plot_graph()`: Plots the actual vs. predicted stock prices.
- `main()`: Coordinates the data preprocessing, model training, prediction, and result saving.

## Data Source
https://www.kaggle.com/datasets/rohitjain454/all-stocks-5yr/data

## Requirements

To install the necessary packages, run:

```bash
git clone https://github.com/danieljames96/ML-Projects.git
cd disease-prediction-using-ML
pip install -r requirements.txt
```

## Implementation
- Run the file `diseasePred.ipynb` to understand how the model works
- Run the file `diseasePred.py` to feed custom input data as a csv file `Symptoms.csv` present in the `/dataset` folder.
- The results generated in the `/output` directory in the form of two excel files `model_results.xslx` and `predictions.xlsx`.