import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from keras.models import load_model 


st.title('Stock Trend Prediction')

# Create a text input for the user to enter a stock ticker
user_input = st.text_input('Enter Stock Ticker', 'AAPL')

# Define starting and ending dates for the data
start = '2009-12-31'
end = '2019-12-31'

# Fetch the historical stock data based on user input
df = yf.download(user_input, start=start, end=end)

# Display the subtitle and description of the fetched data
st.subheader('Data from 2010-2019')
st.write(df.describe())

# Display the first few rows of the fetched data
st.subheader('First Few Rows of the Data')
st.write(df.head())

# Visualisation of Closing Price vs Time
st.subheader('Closing Price vs Time chart')
fig = plt.figure (figsize =(12,6))
plt.plot(df.Close)
st.pyplot(fig)

# Visualisation with 100 day Moving Average
st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure (figsize =(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

# Visualisation with 100 day and 200 day Moving Average
st.subheader('Closing Price vs Time chart with 100MA and 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure (figsize =(12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)

# Splitting data into training and testing
data_training = pd.DataFrame (df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int (len(df)*0.70):int(len(df))])

# Scaling the training data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)



# Loading the pre-tained model
model = load_model('keras_model.h5')

# Prepare the testing data
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.transform(final_df[['Close']])

# Create test datasets
x_test =[]
y_test= []

# Create sequences of 100 days of data to predict the next day's price
for i in range (100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

# Convert the test datasets to numpy arrays
x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

# Adjust the predictions and test values back to the original scale
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#Final Graph
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)