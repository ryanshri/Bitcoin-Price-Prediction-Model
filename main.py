import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import io

from tensorflow.keras.losses import MeanAbsoluteError

# Load the model and pass the custom_objects argument for 'mae'
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model_1.h5', custom_objects={'mae': tf.keras.metrics.MeanAbsoluteError})


model = load_model()

# Function to predict future prices using the univariate model
def predict_univariate_price(input_data, days_to_predict):
    input_data = np.array(input_data['Price'])  # Use only the 'Price' column
    input_data = input_data[-7:]  # Use the last 7 time steps (window size)

    # Initialize a list to store predictions
    predictions = []

    # Loop to predict for each day
    for _ in range(days_to_predict):
        # Reshape input to (1, 7) for the univariate model
        input_window = input_data.reshape((1, 7))

        # Predict using the model
        pred = model.predict(input_window)[0][0]  # Get the predicted price

        # Append the prediction
        predictions.append(pred)

        # Update the input data to include the new prediction
        input_data = np.append(input_data[1:], pred)  # Shift the window by 1 and append the new prediction

    return predictions

# Streamlit app interface
st.title('Bitcoin Price Prediction Model_1')
st.write('This app predicts the future price of Bitcoin using a univariate time series model.')

# Load historical Bitcoin price data
@st.cache_data
def load_bitcoin_data():
    # Load the CSV file containing bitcoin price data
    data = pd.read_csv('bitcoin_prices.csv', index_col='Date', parse_dates=True)
    return data

# Load historical data
data = load_bitcoin_data()

# Show historical data as a chart
st.subheader('Historical Bitcoin Prices')
st.line_chart(data['Price'])

# Input the number of future days to predict
days_to_predict = st.number_input('Enter number of days to predict:', min_value=1, max_value=30, value=7)

# Predict future prices based on the input features
if st.button('Predict Bitcoin Price'):
    future_prices = predict_univariate_price(data[['Price']], days_to_predict)

    # Create a date range for future predictions
    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=days_to_predict)

    # Create a DataFrame for the future predictions
    future_data = pd.DataFrame(future_prices, index=future_dates, columns=['Predicted Price'])

    # Display predicted prices
    st.subheader('Predicted Bitcoin Prices')
    st.write(future_data)

    # Plot historical and predicted prices
    st.subheader('Bitcoin Price Prediction Chart')
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data['Price'], label='Historical Price')
    plt.plot(future_data.index, future_data['Predicted Price'], label='Predicted Price', linestyle='--', color='orange')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Price')
    st.pyplot(plt.gcf())

    # Download predicted prices as CSV
    csv_buffer = io.StringIO()
    future_data.to_csv(csv_buffer)
    st.download_button(
        label="Download Predicted Prices as CSV",
        data=csv_buffer.getvalue(),
        file_name='predicted_bitcoin_prices.csv',
        mime='text/csv'
    )
