
# Installing packages
import pickle
from pickle import load
from pickle import dump
import pandas as pd
import numpy as np
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA


# Title and sidebar
st.title("Model Deployment: Forecasting Gold Price")
st.sidebar.header("User input parameters")


# Input function
def user_input_features():
    fc_days = st.sidebar.number_input("Insert number of days to predict.")
    return int(fc_days)

df = user_input_features()
st.subheader('User Input parameters')
st.write(df)

#load data
gold_new=pd.read_csv(r"C:\Users\Trupti Kendre\OneDrive\Desktop\Gold_data.csv",header=0, index_col=0,squeeze=True,parse_dates=True)

gold_new=gold_new.dropna()

x_train=gold_new[:2182]
x_test=gold_new[2182:]

Final_ARIMA_Model=ARIMA(x_train,order=(3,1,1)).fit()
model=Final_ARIMA_Model.forecast(30)

model=pd.DataFrame(model)
model.columns=[('forecast')]



# Output predictions
st.subheader('Predicted Result')


st.subheader('Prediction Probability')
st.write(model)