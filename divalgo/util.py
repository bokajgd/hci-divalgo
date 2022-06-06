import os
import streamlit as st
from sklearn.base import is_classifier, is_regressor
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from ipywidgets import Output, VBox
import pandas as pd
import pickle 
from ipywidgets import HTML
from sklearn.metrics import confusion_matrix

def show_heatmap(palette=None):
    print("heatmap")

def show_confusion_matrix():
    print("Confusion")

def accuracy_chart_type(confusion, labels =["True positive", "True negative", "False positive", "False negative"], colors=None):
    df = pd.DataFrame({"Value":confusion, "Type":labels})
    fig = go.Figure(
        data=[go.Pie(labels=df["Type"], 
                     values=df["Value"], 
                     sort=False,
                     direction="clockwise")]
    )
  
    # fig = px.pie(df, 
    #              values="Value", 
    #              names="Type", 
    #              color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_traces(textposition='inside', textinfo='percent+label', marker = dict(colors = colors))
    fig.update_layout(showlegend=False, font_family="Times New Roman")
    
    return fig


def accuracy_chart(accuracy, labels=["True predictions", "False predictions"], colors=None):
    
    sizes = [accuracy, 1-accuracy]
    df = pd.DataFrame({"Value":sizes,"Type":labels})
    fig = go.FigureWidget()
    fig.add_pie(values=df["Value"], labels=df["Type"], marker = dict(colors=colors))
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=False,
                    #   font_family="Courier New",
                    #   font_color="blue",
                      font_family="Times New Roman")
                    #   title_font_color="red",
                    #   legend_title_font_color="green")

    return fig



class Evaluate:
    def __init__(self, feed, model):
        assert is_classifier(model) or is_regressor(model), "Please input an sklearn model"
        X_test, y_test, y_pred = feed
        self.model = model
        self.X_test = X_test 
        self.y_test = y_test 
        self.y_pred = y_pred
        self.colors = ["#99B898", "#42823C", "#FF847C", "#E84A5F", "#2A363B"]
    
    def open_visualization(self):
        df = pd.DataFrame({"X_test": self.X_test,
                           "y_test": self.y_test,
                           "y_pred": self.y_pred})

        os.makedirs("tmp")
        df.to_csv(os.path.join("tmp", "data.csv"))
        pickle.dump(self.model, open(os.path.join("tmp", "model.pkl"), "wb"))

        os.system(f'streamlit run {os.path.join("divalgo", "streamlit_app.py")}')

    def heatmap(self, palette=None):
        show_heatmap(palette)
    
    def confusion(self):
        show_confusion_matrix()
    
    def accuracy(self, accuracy, labels=["True predictions", "False predictions"]):
        fig = accuracy_chart(accuracy, labels, [self.colors[i] for i in [0,2]])
        fig.show()
    
    def accuracy_type(self, confusion,  labels =["True positive", "True negative", "False positive", "False negative"]):
        fig = accuracy_chart_type(confusion, labels, self.colors)
        fig.show()