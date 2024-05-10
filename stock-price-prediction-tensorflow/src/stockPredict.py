import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Setting environment variable to avoid TF OneDNN error
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from openpyxl import Workbook
from openpyxl.drawing.image import Image
import warnings
warnings.filterwarnings("ignore")

def preprocess_data(data_dir, company):
    """
    Preprocesses the stock data for a given company.
    
    Parameters:
    data_dir (str or Path): Path to the CSV file containing stock data.
    company (str): The company name to filter data for.
    
    Returns:
    tuple: Processed training and test sets, scaler, and original data slices.
    """
    data = pd.read_csv(data_dir)
    data['date'] = pd.to_datetime(data['date'])
    company_data = data[data['Name'] == company]

    close_data = company_data.filter(['close'])
    dataset = close_data.values
    training = int(np.ceil(len(dataset) * .95))

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[:training, :]
    x_train, y_train = [], []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    test_data = scaled_data[training - 60:, :]
    x_test, y_test = [], dataset[training:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    return x_train, y_train, x_test, y_test, scaler, company_data[training:], company_data[:training]

def build_model(x_train):
    """
    Builds and compiles an LSTM model.
    
    Parameters:
    x_train (ndarray): Training feature set.
    
    Returns:
    keras.Sequential: Compiled LSTM model.
    """
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(units=64, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(keras.layers.LSTM(units=64))
    model.add(keras.layers.Dense(32))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1))
    return model

def train_model(model, x_train, y_train):
    """
    Trains the LSTM model on the training data.
    
    Parameters:
    model (keras.Sequential): The LSTM model.
    x_train (ndarray): Training feature set.
    y_train (ndarray): Training labels.
    """
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(x_train, y_train, epochs=20, verbose=2)

def predict_stock(model, x_test, scaler, y_test):
    """
    Predicts stock prices using the trained model and calculates performance metrics.
    
    Parameters:
    model (keras.Sequential): Trained LSTM model.
    x_test (ndarray): Test feature set.
    scaler (MinMaxScaler): Scaler used for data normalization.
    y_test (ndarray): Actual test labels.
    
    Returns:
    tuple: Predictions, mean squared error (MSE), and root mean squared error (RMSE).
    """
    predictions = model.predict(x_test, verbose=2)
    predictions = scaler.inverse_transform(predictions)
    mse = np.mean((predictions - y_test) ** 2)
    rmse = np.sqrt(mse)
    return predictions, mse, rmse

def plot_graph(company, test_data, train_data, output_path, predictions):
    """
    Plots the stock close price and saves the plot as an image.
    
    Parameters:
    company (str): The company name.
    test_data (DataFrame): Test data with actual close prices.
    train_data (DataFrame): Training data with actual close prices.
    output_path (str or Path): Path to save the plot image.
    predictions (ndarray): Predicted stock prices.
    """
    test_data['predictions'] = predictions

    plt.figure(figsize=(10, 8))
    plt.plot(train_data['date'], train_data['close'], label='Train')
    plt.plot(test_data['date'], test_data['close'], label='Test')
    plt.plot(test_data['date'], test_data['predictions'], label='Predictions')
    plt.title(f'{company} Stock Close Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def main():
    """
    Main function to execute the data preprocessing, model building, training, prediction, 
    and saving results to an Excel file with plot.
    """
    company = 'AAPL'
    data_dir = Path(__file__).resolve().parent.parent / 'data' / 'all_stocks_5yr.csv'
    
    x_train, y_train, x_test, y_test, scaler, test_data, train_data = preprocess_data(data_dir, company)
    model = build_model(x_train)
    train_model(model, x_train, y_train)
    predictions, mse, rmse = predict_stock(model, x_test, scaler, y_test)

    results = pd.DataFrame({
        'date': pd.to_datetime(test_data['date']).values,
        'predictions': predictions.flatten()
    })
    
    output_path = Path(__file__).resolve().parent.parent / 'output' / 'stock_price.xlsx'
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Write data to the first sheet
        results.to_excel(writer, sheet_name='Predictions', index=False)
        
        # Create the plot and save it
        plot_path = Path(__file__).resolve().parent.parent / 'output' / 'stock_plot.png'
        plot_graph(company, test_data, train_data, plot_path, predictions)
        
        # Load the workbook and access the sheet
        workbook = writer.book
        plot_sheet = workbook.create_sheet(title='Plot')
        
        # Add the plot image to the sheet
        img = Image(plot_path)
        plot_sheet.add_image(img, 'A1')
    
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
