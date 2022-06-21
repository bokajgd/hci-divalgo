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

def get_pie_text(acc_by_type):
    if acc_by_type:
        text = """
        <p style="font-family:Tahoma; font-size: 13px;" >This chart shows model accuracy. 
        Accuracy is percentage of true predictions out of total predictions. 
        Thus, an accuracy of 70% means the model predicts the true label for 70% of the test samples.</p>
        """
    else: 
        text = """
        <p style="font-family:Tahoma; font-size: 13px;" >This chart shows proportions of True Positives (TPs), 
        True Negatives (TNs), False Positives (FPs) and False Negatives (FNs).
        TP is the proportion of classifications that are correctly predicted to be class 1, 
        while TN is correctly predicted class 0's. 
        Conversely, FP is porportion of misclassifications of class 0's and class 1's, and FN class 1's as class 0's.
        Compare this chart with the confusion matrix to the right.</p>
        """
    return text

def get_cm_text():
    text = """ <p style="font-family:Tahoma; font-size: 13px;" >This matrix shows the distribution of True Positives (TPs), 
        True Negatives (TNs), False Positives (FPs) and False Negatives (FNs).
        TP is the proportion of classifications that are correctly predicted to be class 1, 
        while TN is correctly predicted class 0's. 
        Conversely, FP is porportion of misclassifications of class 0's and class 1's, and FN class 1's as class 0's.
        Compare this chart with the pie chart to the left.
        Read more <a href="https://en.wikipedia.org/wiki/Confusion_matrix">here</a></p>
    """
    return text

def get_aucroc_text():
    text = """ <p style="font-family:Tahoma; font-size: 13px;" >This plot shows the AUC-ROC curve, 
    that is, the Area Under the Curve - Receiver Operating Characteristics. 
    It shows the relationship between True Positive Rate (TPR), which is defined as TP /(TP+FN) (see figures below),
    and False Positive Rate (FPR), defined as FP / (TN+FP).
    AUC-ROC is a good measures is your dataset is unbalanced. 
    You can check this in the table to the right.
    Read more <a href="https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5">here</a></p>
    """
    return text

def get_table_text():
    text = """ <p style="font-family:Tahoma; font-size: 13px;" >This table shows basic model performance measures. 
    They all rely on relations between  True Positives (TPs), 
    True Negatives (TNs), False Positives (FPs) and False Negatives (FNs).
    TP is the proportion of classifications that are correctly predicted to be class 1, 
    while TN is correctly predicted class 0's. 
    Conversely, FP is porportion of misclassifications of class 0's and class 1's, and FN class 1's as class 0's.
    Support is the number of data points in each class. Thus, you can check the balance of your dataset here. 
    </p>
    """
    return text


def main(df, model):
    st.sidebar.markdown("", unsafe_allow_html=True)
    if "color_blind" not in st.session_state:
        st.session_state["color_blind"] = False
        st.session_state["value"] = False
    color_blind=st.sidebar.checkbox("Use colourblind friendly colors", value=st.session_state["color_blind"])
    
    st.session_state["color_blind"] = color_blind

    if st.session_state["color_blind"]:
        st.session_state["colors"] = ["#44AA99", "#117733", "#DDCC77", "#997700"]
    else:
        st.session_state["colors"] = ["#99B898", "#42823C", "#FF847C", "#E84A5F", "#2A363B"]
    st.sidebar.markdown("<br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br>", unsafe_allow_html=True)

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

    # metrics_table = div.metrics_table(df, model)
    col1, col2 = st.columns(2)
   
    with col1:
        table_title = '<p style="font-family:Tahoma; text-align:center;  color:#928374; font-size: 25px;"> <br> Metrics Table</p>'
        st.markdown(table_title, unsafe_allow_html=True)   
        eq_help = st.checkbox("See equations")

        st.session_state["eq_help"]=eq_help 

        if not st.session_state["eq_help"]:
            metrics_table = div.metrics_table(df, model)
        if st.session_state["eq_help"]:
            metrics_table = div.metrics_table(df, model, help=True)
        metrics_table.update_layout(height=400,
            margin=dict(
                l=15,
                r=35,
                b=0,
                t=10))
        
        st.plotly_chart(metrics_table, use_container_width=True)
        with st.expander("What does this show?"):
            table_text = get_table_text()
            st.markdown(table_text, unsafe_allow_html=True)

    with col2:
        auc_title = '<p style="font-family:Tahoma; text-align:center;  color:#928374; font-size: 25px;"> <br> AUC-ROC Curve</p>'
        st.markdown(auc_title, unsafe_allow_html=True)   
        st.plotly_chart(roc, use_container_width=True)
        aucroc_text = get_aucroc_text()
        with st.expander("What does this show?"):
            st.markdown(aucroc_text, unsafe_allow_html=True)



    acc = accuracy_score(df["y_test"], df["y_pred"])
    if "acc_by_type" not in st.session_state:
        st.session_state["acc_by_type"]=None

    col3, col4 = st.columns(2)
    with col3:
        pie_title = '<p style="font-family:Tahoma; text-align:center;  color:#928374; font-size: 25px;"><br><br> Pie Charts</p>'
        st.markdown(pie_title, unsafe_allow_html=True)   
        acc_by_type = st.checkbox("Split pie by type")
        st.session_state["acc_by_type"]=acc_by_type 
    
    if not st.session_state["acc_by_type"]:
        accuracy_chart = div.accuracy_chart(acc, colors=[st.session_state["colors"][i] for i in [0,2]])
    if st.session_state["acc_by_type"]:
        tn, fp, fn, tp = confusion_matrix(df["y_test"], df["y_pred"]).ravel()
        accuracy_chart = div.accuracy_chart_type((tp,tn,fp,fn), colors=st.session_state["colors"])
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
        cm_title = '<p style="font-family:Tahoma; text-align:center;  color:#928374; font-size: 25px;"><br><br> Confusion Matrix</p>'
        st.markdown(cm_title, unsafe_allow_html=True)   

    ####################
    # Confusion matrix #
    ####################
    cm_colors = [st.session_state["colors"][0], st.session_state["colors"][3], st.session_state["colors"][2], st.session_state["colors"][1]]
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
        chart_text = get_pie_text(st.session_state["acc_by_type"])
        with st.expander("What does this show?"):
            st.markdown(chart_text, unsafe_allow_html=True)

    with col6:
        st.plotly_chart(cm, use_container_width=True)
        cm_text = get_cm_text()
        with st.expander("What does this show?"):
            st.markdown(cm_text, unsafe_allow_html=True)


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
