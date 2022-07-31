import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_errror
import warnings
import matpltlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
warnings.filterwarnings('ignore')

st.header("***Forecasting gold price for upcoming 30 days***")


st.write('This model is forecasts Gold Price for upcoming days')

gold_new=pd.read_csv(r"C:\Users\Trupti Kendre\Downloads\Gold_data.csv", header=0, index_col=0,squeeze=True,parse_dates=True)
gold_new=gold_new.set_index('date', drop =True)

##model
import statsmodels.api as sn
from statsmodels.tsa.holtwinters import ExponentialSmoothing

model=ARIMA(x_train,order=(3,1,1))

model_fore= Final_ARIMA_Model.forecast(30)
model_fore= pd.DataFrame(model_fore)
model_fore.columns=[('forecast')]

#plot
st.line_chart(data=SARIMA_fore, width=0,height=0, use_container_width=True

bt=st.button("Forecast")
if bt is True
   st.write(model_fore)
   plt.figure(figsize=(16,8))
   plt.plot(model_fore['forecast'])
   st.pyplot()

    
   
   
