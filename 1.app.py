import streamlit as st
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')



st.header("***Forecasting gold price for upcoming 30 days***")


st.write('This model is forecasts Gold Price for upcoming days')


gold_new=pd.read_csv(r"C:\Users\Trupti Kendre\OneDrive\Desktop\Gold_data.csv"
, header=0, index_col=0,squeeze=True,parse_dates=True)

## Model
#import statsmodels.api as sm
#from statsmodels.tsa.hotwinters import ExponentialSmoothing

Final_ARIMA_Model= ARIMA(gold_new['price'],order=(3,1,1)).fit()

model_fore= Final_ARIMA_Model.forecast(30)

model_fore= pd.DataFrame(model_fore)
model_fore.columns=[('forecast')]

##plot
#st.line_chart(data=ARIMA_fore,width=0,height=0, use_container_width=True)
bt=st.button("Forecast")
if bt is True:
   st.write(model_fore)
   plt.figure(figsize=(16,8))
   plt.plot(model_fore['forecast'])
   st.pyplot()
