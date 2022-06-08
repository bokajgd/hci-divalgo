from pydoc import pager
import streamlit as st
import divalgo_class as div
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.figure_factory as ff
import pickle
import os
import shutil
from PIL import Image
from bokeh.io import curdoc
from bokeh.themes import Theme
from bokeh.themes import built_in_themes

colors = ["#99B898", "#42823C", "#FF847C", "#E84A5F", "#2A363B"]
st.set_page_config(page_title="DIVALGO", layout="wide")


def main(df, model):
    st.sidebar.markdown("<br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> ", unsafe_allow_html=True)

    with st.sidebar.container():
        image = Image.open(os.path.join("logos", "trans_logo.png"))
        st.image(image, use_column_width=True)

    page_title = '<p style="font-family:Tahoma; text-align:center;  color:#928374; font-size: 52px;">Coefficients and Embeddings</p>'
    st.markdown(page_title, unsafe_allow_html=True)
    page_intro = '''<p style="font-family:Tahoma; font-size: 15px;" >The interactive plot below lets you explore the model
     performance on the test set by projecting the images in the test data set onto a 2D plane using uMAP embeddings. 
     Let your mouse hover over the data points to view the images.
     <br>
    _________________________________________________________________________________________________________________</p>'''
    st.markdown(page_intro, unsafe_allow_html=True)

    ##################
    # EMBEDDING PLOT #
    ##################
    col1, _, col2, col3 = st.columns((1.5,0.4,1.1,1))
    
    with col1:
        point_size = st.slider('Choose size of points on slider', 0, 50, 10, 5)

    with col2: 
        empty = '<p <br> </p>'
        st.markdown(empty, unsafe_allow_html=True)  
        color_emb_plt = st.checkbox("Colour by class")
        st.session_state["color_emb_plt"]=color_emb_plt 

    col4, col5 = st.columns((3,1))
    with col4:
        if not "embeddings" in st.session_state:
            if not st.session_state["color_emb_plt"]:
                embedding_plot, embeddings = div.embedding_plot(df,size=point_size)
                st.session_state["embeddings"] = embeddings
            else:
                embedding_plot, embeddings = div.embedding_plot(df, colour=True, size=point_size)
                st.session_state["embeddings"] = embeddings
        else:
            if not st.session_state["color_emb_plt"]:
                embedding_plot, embeddings = div.embedding_plot(df,size=point_size)
            else:
                embedding_plot, embeddings = div.embedding_plot(df, colour=True, size=point_size)

        doc = curdoc()
        doc.theme = Theme(filename='custom.yaml')
        doc.add_root(embedding_plot)
        st.bokeh_chart(embedding_plot, use_container_width=True)

    with col5:
        heatmap_title = '<p style="font-family:Tahoma; color:#928374; font-size: 20px;"> Coefficient Heatmaps</p>'
        st.markdown(heatmap_title, unsafe_allow_html=True)   
        coef_intro = '''<p style="font-family:Tahoma; font-size: 13px;" >The plot below shows a heatmap of the model coefficients rearranged into the 
        same shape as the training images were transformed into (50x50 pixels). Positive values indicate coefficients that push image prediction toward 
        the class 'dog'. Negative values co indicate coefficients that push image prediction toward the class 'wolf'. </p>'''

        with st.expander('What does this show?'):
            st.markdown(coef_intro, unsafe_allow_html=True)

        coef_heatmaps = div.coef_heatmaps(model)
        coef_heatmaps.update_layout(
            margin=dict(
                l=5,
                r=5,
                b=4,
                t=15,
                # pad=4
            )
        )
        st.plotly_chart(coef_heatmaps, use_container_width=True)

        abs_intro = '''<p style="font-family:Tahoma; font-size: 13px;" >The plot below shows a heatmap of the absolute 
        magnitude of the coefficient. Thus, higher values indicate pixels with higher influence on the predictions. </p>'''
        with st.expander('What does this show?'):
            st.markdown(abs_intro, unsafe_allow_html=True)

        abs_heatmaps = div.coef_heatmaps(model, absolute=True)
        abs_heatmaps.update_layout(
            margin=dict(
                l=5,
                r=5,
                b=4,
                t=15,
                # pad=4
            )
        )
        st.plotly_chart(abs_heatmaps, use_container_width=True)



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
