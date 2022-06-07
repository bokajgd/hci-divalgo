import streamlit as st
import divalgo_class  as div
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.figure_factory as ff
import pickle
import os
import shutil
from PIL import Image

def main(df, model):
    st.sidebar.markdown("<br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> ", unsafe_allow_html=True)

    with st.sidebar.container():
        image = Image.open("/Users/au617011/Documents/Semester8/HCI//hci-divalgo/divalgo/logos/logo.png")
        st.image(image, use_column_width=True)

    print('Hi')

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
