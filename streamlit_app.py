import streamlit as st
import pandas as pd
import numpy as np
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
gold_validation = gold['price'][1746:]
model=ARIMA(x_train,order=(3,1,1))

model_fit=model.fit(disp=0)
print(model_fit.summary())
fc, se, conf=gold_new.forecast(436,alpha=0.05)

def forecast_accuracy(forecast,actual):
    mape=(np.mean(np.abs(forecast - actual)/np.abs(actual))*100).round(2)
    rmse=np.sqrt(((forecast-actual)**2).mean())
    return({'Mean Absolute Percentage Error(%) ':mape,
           'Root Mean Squared Error':rmse})
forecast_accuracy(fc, gold_validation.values)

forecast=model_fit.forecast(steps=30)[0]
model_fit.plot_predict(1,648)
plt.show()

pd.DataFrame(forecast)
