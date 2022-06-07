import streamlit as st
import pandas as pd
import divalgo_class as div
import sys
import os
import pickle
import shutil
from PIL import Image

def main(df, model):
    colors = ["#99B898", "#42823C", "#FF847C", "#E84A5F", "#2A363B"]
    st.set_page_config(page_title="DIVALGO", layout="wide")

    st.sidebar.markdown("<br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> ", unsafe_allow_html=True)

    with st.sidebar.container():
        image = Image.open("/Users/au617011/Documents/Semester8/HCI//hci-divalgo/divalgo/logos/logo.png")
        st.image(image, use_column_width=True) 

    r1c1, r1c2, r1c3 = st.columns([1.5,5,1.5])

    with r1c1:
        st.write(' ')

    with r1c2:
        st.image("/Users/au617011/Documents/Semester8/HCI//hci-divalgo/divalgo/logos/logo.png", use_column_width=True)

    with r1c3:
        st.write(' ')

    st.markdown("<p style='text-align: center; '>Welcome to divalgo. On this page, you can explore visualizations of your models, its predictions and its errors. <br>  blablabla blabla bla bla bla bla bla bla bla bla bla blablablabla bla bla bla bla blablablabla blabla la bla <br> blablabla blabla bla bla bla bla bla bla bla bla bla bla <br> </p>", unsafe_allow_html=True)

    r2c1, r2c2, r2c3 , r2c4, r2c5 = st.columns([2,1,2,1,2])

    with r2c1:
        st.image("/Users/au617011/Documents/Semester8/HCI//hci-divalgo/divalgo/logos/circle.png", use_column_width=True)

    with r2c3:
        st.image("/Users/au617011/Documents/Semester8/HCI//hci-divalgo/divalgo/logos/circle.png", use_column_width=True)

    with r2c5:
        st.image("/Users/au617011/Documents/Semester8/HCI//hci-divalgo/divalgo/logos/circle.png", use_column_width=True)

        

    
if __name__ == "__main__":

    # Check if the data is already loaded
    if "data" not in st.session_state:
        df = pd.read_csv(os.path.join("tmp", "data.csv"))
        st.session_state["data"]= df
    
    # Use the loaded data if it is there
    else:
        df = st.session_state["data"]

    # Check if the model is already loaded
    if "model" not in st.session_state:
        model = pickle.load(open(os.path.join("tmp", "model.pkl"), "rb"))
        st.session_state["model"]= model
        # Remove the directory after loading model - little hacky solution but works. 
        shutil.rmtree("tmp")
    
    else:
        model = st.session_state["model"]

    main(df, model)

