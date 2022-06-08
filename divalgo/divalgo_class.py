import os
import streamlit as st
from sklearn.base import is_classifier, is_regressor
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import pickle 
import numpy as np
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper, CustomJS, Circle

from utils import get_embeddings, prob_barplot, np_image_to_base64, get_embedding_df

def accuracy_chart_type(confusion:tuple, 
                        labels =["True positive", "True negative", "False positive", "False negative"], 
                        colors=None):
    """Function creating pie chart of accuracy based on prediction type (TP, TN, FP, FN)

    Args:
        confusion (tuple): tuple containing the confusion matrix info in the order TP, TN, FP, FN
        labels (list, optional): List of labels to give the input. Defaults to ["True positive", "True negative", "False positive", "False negative"].
        colors (list, optional): list of colors to use. Defaults to None in which case plotly default colors are used.

    Returns:
        fig: the plotly figure
    """                        
    df = pd.DataFrame({"Value":confusion, "Type":labels})
    fig = go.Figure(
        data=[go.Pie(labels=df["Type"], 
                     values=df["Value"], 
                     sort=False,
                     direction="clockwise")]
    )
    fig.update_traces(textposition='inside', textinfo='percent+label', marker = dict(colors = colors), textfont_size=18)
    fig.update_layout(showlegend=False, font_family="Times New Roman")
    
    return fig


def accuracy_chart(accuracy, 
                   labels=["True predictions", "False predictions"], 
                   colors=None):
    """Function creating pie chart of accuracy (True and False predictions)

    Args:
        accuracy (float): Accuracy of the model
        labels (list, optional): Labels to give the two parts of the chart. Defaults to ["True predictions", "False predictions"].
        colors (list, optional): List of colors to use. Defaults to None in which case plotly default colors are used.

    Returns:
        fig: the plotly figure
    """    
    
    sizes = [accuracy, 1-accuracy] # Calculate False predictions
    df = pd.DataFrame({"Value":sizes,"Type":labels})

    fig = go.FigureWidget()
    fig.add_pie(values=df["Value"], labels=df["Type"], marker = dict(colors=colors))
    fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=20)
    fig.update_layout(showlegend=False,
                      font_family="Times New Roman")
    
    return fig


def confusion_mat(y_test, y_pred, colors):
    matrix = confusion_matrix(y_test, y_pred)
    z_flat = matrix.flatten()
    d = pd.DataFrame({"value":z_flat, "color":colors[::-1]})

    # normalize array between 0 and 1 for the color scale
    def normalize(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))
    
    d["normalized_scale"] = normalize(d["value"])
    d = d.sort_values("normalized_scale").reset_index()
  
    color0 = d["color"][0]
    border1 = d["normalized_scale"][1]
    color1  = d["color"][1]
    border2 = d["normalized_scale"][2]
    color2  = d["color"][2]
    border3 = d["normalized_scale"][3]
    color3  = d["color"][3]

    # custom color scale
    mycolors=[[0, color0],        
            [border1, color0],
            [border1, color1], 
            [border2, color1],
            [border2, color2],
            [border3, color2], 
            [border3, color3],
            [1, color3]]


    labels=["Dogs", "Wolfs"] # How to know what to use here????
    z_text = z_text = [[str(y) for y in x] for x in matrix]
    fig = ff.create_annotated_heatmap(matrix, 
                                      x=labels, 
                                      y=labels, 
                                      annotation_text=z_text, 
                                      colorscale=mycolors
                                      )

    fig.update_layout(width=400, height=400)
    
    return fig


# Defining function for interactive embedding plot
def embedding_plot(df, size, new_df=None):

    if not isinstance(new_df, pd.DataFrame): 
        embeddings_2d, image_arrays = get_embeddings(df)
        new_df = get_embedding_df(df, embeddings_2d, image_arrays)
    
    s1 = ColumnDataSource(data=new_df)
    color_mapping = CategoricalColorMapper(factors=["True", "False"], palette=["#99B898", "#FF847C"])
    
    p1 = figure(plot_width=800, plot_height=800,
                tools=('pan, wheel_zoom, reset, box_zoom'), 
                title="UMAP projection of image embeddings")
    p1.circle('x', 'y', source=s1, alpha=0.6, size = size,
            color=dict(field='pred_is_true', transform=color_mapping))

    p1.add_tools(HoverTool(tooltips="""
    <div>
        <div class="column">
            <img src='@bar' style='float: left; margin: 5px 5px 5px 5px width:175px;height:125px;'/>
        <div>
        <div class="column">
            <img src='@image' style='float: left; margin: 5px 5px 5px 5px width:175px;height:125px;'/>
        <div>
            <span style='font-size: 12px'> <strong> Predicted class: </strong> @prediction</span>
        <div>
            <span style='font-size: 12px'> <strong> True class: </strong>  @category </span>
        </div>
    </div>
    """))

    return p1, new_df


######################
# DEFINING THE CLASS #
######################

class Evaluate:
    def __init__(self, feed, model):
        assert is_classifier(model) or is_regressor(model), "Please input an sklearn model"
        X_test, y_test, filenames = feed
        self.model = model
        self.X_test = X_test 
        self.y_test = y_test
        self.filenames = filenames
        self.y_pred = model.predict(X_test)
        self.y_pred_probs = self.model.predict_proba(self.X_test)
        self.colors = ["#99B898", "#42823C", "#FF847C", "#E84A5F", "#2A363B"]

        self.df = pd.DataFrame(self.X_test)
        self.df["y_test"] = self.y_test
        self.df["y_pred"] = self.y_pred
        self.df["filename"] = self.filenames

        df_2 = pd.DataFrame(self.y_pred_probs, columns=['prob0', 'prob1'])

        self.df = pd.concat([self.df, df_2], axis=1)

    def open_visualization(self):
        if not os.path.exists("tmp"):
            os.makedirs("tmp")
        self.df.to_csv(os.path.join("tmp", "data.csv"))
        pickle.dump(self.model, open(os.path.join("tmp", "model.pkl"), "wb"))

        os.system(f'streamlit run {os.path.join("☌frontpage.py")}')

    def confusion(self):
        fig = confusion_mat(self.y_test, self.y_pred)
        fig.show()
    
    def accuracy(self, labels=["True predictions", "False predictions"]):
        accuracy = accuracy_score(self.y_test, self.y_pred)
        fig = accuracy_chart(accuracy, labels, [self.colors[i] for i in [0,2]])
        fig.show()
    
    def accuracy_type(self, labels =["True positive", "True negative", "False positive", "False negative"]):
        tn, fp, fn, tp = confusion_matrix(self.y_test, self.y_pred).ravel()
        fig = accuracy_chart_type((tp, tn, fp, fn), labels, self.colors)
        fig.show()

    def explore_embeddings(self):
        p = embedding_plot(df=self.df, size = 10)
        show(p)
