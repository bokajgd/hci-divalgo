import os
import streamlit as st
from sklearn.base import is_classifier, is_regressor
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import pickle 
import numpy as np

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


def confusion_mat(y_test, y_pred):
    matrix = confusion_matrix(y_test, y_pred)
    # color_vals = matrix/np.amax(matrix)
    # print(list(color_vals.flatten()))
    # col_vals = [[]]
    min_color = "rgb(153, 184, 152)"
    max_color = "rgb(255, 132, 124)"

    labels=["Dogs", "Wolfs"] # How to know what to use here????
    z_text = z_text = [[str(y) for y in x] for x in matrix]
    fig = ff.create_annotated_heatmap(matrix, 
                                      x=labels, 
                                      y=labels, 
                                      annotation_text=z_text, 
                                      font_colors = [min_color, max_color]
                                      )

    # fig = ConfusionMatrixDisplay(matrix, display_labels=labels)
    # fig.plot()
    fig.update_layout(width=400, height=400)
    
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

        os.system(f'streamlit run {os.path.join("divalgo", "🚪frontpage.py")}')

    def confusion(self):
        fig = confusion_mat(self.y_test, self.y_pred)
        fig.show()
    
    def accuracy(self, accuracy, labels=["True predictions", "False predictions"]):
        fig = accuracy_chart(accuracy, labels, [self.colors[i] for i in [0,2]])
        fig.show()
    
    def accuracy_type(self, confusion,  labels =["True positive", "True negative", "False positive", "False negative"]):
        fig = accuracy_chart_type(confusion, labels, self.colors)
        fig.show()