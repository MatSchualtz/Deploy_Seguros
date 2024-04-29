import pickle
import pandas as pd
import streamlit as st
import sklearn

#Page Config
st.set_page_config(page_title='Insurance Prediction')
st.sidebar.header('File Prediction')
st.title("Insurance prediction")

st.markdown("Predict medical insurance based on the using csv file:")

# -- Model -- #

with open('./models/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

data = st.file_uploader('Upload your file')

if data:
    df_input = pd.read_csv(data)
    insurance_prediction = model.predict(df_input)
    df_output = df_input.assign(prediction = insurance_prediction)

    st.markdown('Insurance Cost Prediction')
    st.write(df_output)
    st.download_button(
        label='Download CSV', 
        data=df_output.to_csv(index=False).encode('utf-8'), 
        mime='text/csv',
        file_name= 'Prediction_Insurance_Cost.csv')