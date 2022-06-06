import streamlit as st
import pandas as pd
import util as div
import sys
import os
import pickle
import shutil
from sklearn.metrics import accuracy_score, confusion_matrix

def main(df, model):
    colors = ["#99B898", "#42823C", "#FF847C", "#E84A5F", "#2A363B"]
    st.set_page_config(page_title="DIVALGO", layout="centered")

    st.title("DIVALGO - diagnose and evaluate your model")

    st.write("""Welcome to DIVALGO. 
    On this page, you can explroe visualizations of your models, its predictions and its errors. 
    This first visualization shows accuracy. 
    """)

    ############
    # ACCURACY #
    ############
    acc = accuracy_score(df["y_test"], df["y_pred"])
    if "acc_by_type" not in st.session_state:
        st.session_state["acc_by_type"]=None

    # col1, col2 = st.columns([1,4])
    # with col1:
    acc_by_type = st.checkbox("Split by type")
    st.session_state["acc_by_type"]=acc_by_type
    
    if not st.session_state["acc_by_type"]:
        accuracy_chart = div.accuracy_chart(acc, colors=[colors[i] for i in [0,2]])
    if st.session_state["acc_by_type"]:
        tn, fp, fn, tp = confusion_matrix(df["y_test"], df["y_pred"]).ravel()
        accuracy_chart = div.accuracy_chart_type((tp,tn,fp,fn), colors=colors)
    
    # with col2:
    st.plotly_chart(accuracy_chart)

    
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

