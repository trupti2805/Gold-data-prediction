import streamlit as st
import datetime

from pickle import dump
from pickle import load


import pandas as pd

from statsmodels.tsa.holtwinters import ExponentialSmoothing

import warnings
df=pd.read_csv(r"C:\Users\Trupti Kendre\Downloads\Gold_data.csv")


  
    
    st.sidebar.header('User Input Parameters')
    def user_input_features():
        CLPRICE=st.sidebar.selectbox('Price',('1','1','1'))
        data={'CLPRICE':CLPRICE}
  
     features = pd.DataFrame(data,index=[0])
     return features
    df = user_input_features()
st.subheader('User Input parameters')
st.write(df)
# load the model from disk
loaded_model = load(open('rmse_wf.pkl', 'rb'))

prediction = loaded_model.predict(df)
prediction_proba = loaded_model.predict_proba(df)

st.subheader('Predicted Result')
st.write('Yes' if prediction_proba[0][1] > 0.5 else 'No')

st.subheader('Prediction Probability')
st.write(prediction_proba)
