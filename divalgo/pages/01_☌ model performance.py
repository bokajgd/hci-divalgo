import streamlit as st
import divalgo_class as div
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.figure_factory as ff
import pickle
import os
import shutil
from PIL import Image

colors = ["#99B898", "#42823C", "#FF847C", "#E84A5F", "#2A363B"]
st.set_page_config(page_title="DIVALGO", layout="wide")


def main(df, model):
    st.sidebar.markdown("<br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> ", unsafe_allow_html=True)

    with st.sidebar.container():
        image = Image.open(os.path.join("logos", "trans_logo.png"))
        st.image(image, use_column_width=True)

    page_title = '<p style="font-family:Tahoma; text-align:center; color:#928374; font-size: 52px;"> Overall Model Performance</p>'
    st.markdown(page_title, unsafe_allow_html=True)    
    page_intro = '''<p style="font-family:Tahoma; font-size: 15px;" >This page enables you to explore general performance properties
    of your trained model. The page contains a tabular overview of common ML performance metrics, an AUC-ROC for the model,
    pie charts giving insight into the accuracies of the model, and a confusion matrix. Click the expanders below each element
    to read more about the information they convey.
    <br>
    _________________________________________________________________________________________________________________</p>'''
    st.markdown(page_intro, unsafe_allow_html=True)
    
    ############
    # PLOTS #
    ############

    roc = div.roc_curve_plot(df["y_test"], df["prob1"])
    roc.update_layout(
        margin=dict(
            l=10,
            r=60,
            b=80,
            t=2,
            # pad=4
        )
    )

    col1, col2 = st.columns(2)
    with col1:
        auc_title = '<p style="font-family:Tahoma; text-align:center;  color:#928374; font-size: 25px;"> AUC-ROC Curve</p>'
        st.markdown(auc_title, unsafe_allow_html=True)   
        st.plotly_chart(roc, use_container_width=True)

    acc = accuracy_score(df["y_test"], df["y_pred"])
    if "acc_by_type" not in st.session_state:
        st.session_state["acc_by_type"]=None
    
    col3, col4 = st.columns(2)
    with col3:
        pie_title = '<p style="font-family:Tahoma; text-align:center;  color:#928374; font-size: 25px;"> Pie Charts</p>'
        st.markdown(pie_title, unsafe_allow_html=True)   
        acc_by_type = st.checkbox("Split pie by type")
        st.session_state["acc_by_type"]=acc_by_type 
    
    if not st.session_state["acc_by_type"]:
        accuracy_chart = div.accuracy_chart(acc, colors=[colors[i] for i in [0,2]])
    if st.session_state["acc_by_type"]:
        tn, fp, fn, tp = confusion_matrix(df["y_test"], df["y_pred"]).ravel()
        accuracy_chart = div.accuracy_chart_type((tp,tn,fp,fn), colors=colors)
    accuracy_chart.update_layout(
        margin=dict(
            l=10,
            r=60,
            b=80,
            t=15,
            # pad=4
        )
    )

    with col4:
        cm_title = '<p style="font-family:Tahoma; text-align:center;  color:#928374; font-size: 25px;"> Confusion Matrix</p>'
        st.markdown(cm_title, unsafe_allow_html=True)   

    ####################
    # Confusion matrix #
    ####################
    cm_colors = [colors[0], colors[3], colors[2], colors[1]]
    cm = div.confusion_mat(df["y_test"], df["y_pred"], cm_colors)
    cm.update_layout(
        margin=dict(
            l=10,
            r=10,
            b=80,
            t=0,
            # pad=4
        )
    )

    col5, col6 = st.columns(2)
    with col5:
        st.plotly_chart(accuracy_chart, use_container_width=True)

    with col6:
        st.plotly_chart(cm, use_container_width=True)
    
    with st.expander("Help"):
        st.markdown("""Accuracy of model tells you the percentage of true predictions. An accuracy of 70% thus means that the model \
            predicts the correct label in 70% of the cases. 
A way of unpacking accuracy is to look at True Positives (TPs), True Negatives (TNs), False Positives (FPs) and False Negatives (FNs).
TP is the proportion of classifications that are correctly predicted to be class 1, while TN is correctly predicted class 0's. 
Conversely, FP is porportion of misclassifications of class 0's and class 1's, and FN class 1's as class 0's.
This is what is visualised in the pie chart when the 'Split by type' is checked, and in the confusion matrix. 
You can read more on these measures on https://en.wikipedia.org/wiki/Confusion_matrix""")

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
