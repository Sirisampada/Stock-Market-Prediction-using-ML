import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas_datareader as data
from tensorflow.keras.models import load_model
import streamlit as st
start = "2010-01-01"
end = "2022-03-01"
st.title("Stock Trend Prediction")
ui = st.text_input("Enter stock name ","TCS")
df = data.DataReader(ui,'yahoo',start,end)
#Describing data
st.subheader("Data from 2010 - 2021 ")
st.write(df.describe())
#Visualizations
st.subheader("Closing price vs time")
fig = plt.figure(figsize=(10,5))
plt.plot(df.Close)
st.pyplot(fig)
43
st.subheader("Closing price vs time with 100 ma")
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(10,5)
st.subheader("Closing price vs time with 100 ma")
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(10,5))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)
st.subheader("Closing price vs time with 200 ma")
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(10,5))
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)
st.subheader("Closing price vs time with 100 and 200 ma")
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(10,5))
plt.plot(ma100,"b")
plt.plot(ma200,"g")
plt.plot(df.Close,"r")
st.pyplot(fig)
#train test splitting
data_train = pd.DataFrame(df['Close'][0:int(len(df)*0.70)]) # first 70% values training
data_test = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
#splitting x_train , y_train
#scale down
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_train_array = scaler.fit_transform(data_train)
data_test_array = scaler.fit_transform(data_test)
#load model
model = load_model("stock.h5")
print(model)
#testing
past_100 = data_train.tail(100)
final_df = past_100.append(data_test,ignore_index=True)
#scaling down the data
input_data= scaler.fit_transform(final_df)
#input_data
x_test = []
y_test = []
for i in range(100,input_data.shape[0]):
x_test.append(input_data[i-100:i])
y_test.append(input_data[i,0])
x_test , y_test = np.array(x_test),np.array(y_test)
#predictions
y_pred = model.predict(x_test)
#scale back the values again
scaler = scaler.scale_ #This is the value by which all points in the data were divided to come into
range
scale_factor = 1/scaler[0]
y_pred = y_pred*scale_factor
y_test = y_test*scale_factor
#final graph
st.subheader("Predicted trend vs original")
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,"blue",label="Original Price")
plt.plot(y_pred,"red",label="Predicted Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig2)
