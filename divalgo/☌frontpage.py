import streamlit as st
import pandas as pd
import divalgo_class as div
import sys
import os
import pickle
import shutil
from PIL import Image
import re

def main(df, model):
    st.sidebar.markdown("<br> <br> <br> <br> <br>", unsafe_allow_html=True)
    if "color_blind" not in st.session_state:
        st.session_state["color_blind"] = False
        st.session_state["value"] = False
    color_blind=st.sidebar.checkbox("Use colour blind friendly colors", value=st.session_state["color_blind"])
    
    st.session_state["color_blind"] = color_blind

    if st.session_state["color_blind"]:
        colors = ["#44AA99", "#117733", "#DDCC77", "#997700"]
    else:
        colors = ["#99B898", "#42823C", "#FF847C", "#E84A5F", "#2A363B"]
    st.sidebar.markdown("<br> <br> <br> <br> <br> <br> <br> <br> <br> <br>", unsafe_allow_html=True)

    with st.sidebar.container():
        image = Image.open(os.path.join("logos", "trans_logo.png"))
        st.image(image, use_column_width=True)
    st.set_page_config(page_title="DIVALGO", layout="wide")

    r1c1, r1c2, r1c3 = st.columns([1.5,5,1.5])

    with r1c1:
        st.write(' ')

    with r1c2:
        st.image(os.path.join("logos", "logo.png"), use_column_width=True)

    with r1c3:
        st.write(' ')

    # Get model params
    model_str = str(model)

    if model_str[0:8] == 'Logistic': 
        model_type = 'logistic regression'
        p_iter = re.compile(r'(?<=max_iter=)\d+')
        max_iter = int(re.findall(p_iter,model_str)[0])
    
    n_classes = len(model.classes_)
    pen = model.get_params()['penalty']
    tol = model.get_params()['tol']
    solver = model.get_params()['solver']

    st.markdown('''<p style="font-family:Tahoma; text-align:center; font-size: 16px;">Welcome to divalgo. This
    dashboard allows you to explore and diagnose your trained model through 
    interactive visuals and <br> performance  metrics. Navigate to the three site pages via the 
    menu bar in the left side of this pager.
    <br> 
    ___________________________________________________
    <br>
    ''', unsafe_allow_html=True)
    st.markdown(f'''<p style="font-family:Tahoma; text-align:center; font-size: 16px;" Your model specifications:</p>''', unsafe_allow_html=True)
    st.markdown(f'''<p style="font-family:Tahoma; text-align:center; font-size: 14px;">You have trained a <em>{model_type}</em> with <em>{str(n_classes)}</em> classes using <em>{pen}</em> regularisation. The model 
    was trained for a maximum of <em>{str(max_iter)} iterations</em> <br> using the <em>{solver}</em> solver with a
    tolerance set to <em>{tol}</em>. Enjoy exploring your model.
     <br> <br> <br> </p>''', unsafe_allow_html=True)

    r2c1, r2c2, r2c3 , r2c4, r2c5 = st.columns([2,1,2,1,2])

    with r2c1:
        st.image(os.path.join("logos", "circle_plot.png"), use_column_width=True)

    with r2c3:
        st.image(os.path.join("logos", "circle_pie.png"), use_column_width=True)

    with r2c5:
        st.image(os.path.join("logos", "circle_img.png"), use_column_width=True)

        

    
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

