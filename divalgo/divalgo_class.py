import os
import streamlit as st
from sklearn.base import is_classifier, is_regressor
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, classification_report
from sklearn.linear_model import LogisticRegression
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import pickle 
import numpy as np
import math
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
    fig.update_traces(textposition='inside', textinfo='percent+label', marker = dict(colors = colors), textfont_size=16)
    fig.update_layout(showlegend=False, font_family="Tahoma")
    
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
    fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=16)
    fig.update_layout(showlegend=False,
                      font_family="Tahoma")
    
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


    labelsx=["Actual Dogs", "Actual Wolfs"] 
    labelsy=["Predicted Dogs", "Predicted Wolfs"] 
    z_text = z_text = [[str(y) for y in x] for x in matrix]
    fig = ff.create_annotated_heatmap(matrix, 
                                      x=labelsx, 
                                      y=labelsy, 
                                      annotation_text=z_text, 
                                      colorscale=mycolors
                                      )
    fig.update_yaxes(tickangle=270)

    fig.layout.update(
        go.Layout(
            autosize=False,
            font=dict(
            family="Tahoma",
            size = 16
            ),
            xaxis=dict(
            domain=[0.05,1],
            position=0.99
            ),
            yaxis=dict(
            domain=[0,0.95],
            position=0.01
            )
        )
        )

    fig.update_layout(width=400, height=450)
    
    return fig




# Defining function for interactive embedding plot
def embedding_plot(df, size, colour=False, new_df=None):

    if not isinstance(new_df, pd.DataFrame): 
        embeddings_2d, image_arrays = get_embeddings(df)
        new_df = get_embedding_df(df, embeddings_2d, image_arrays)
    
    s1 = ColumnDataSource(data=new_df)
    
    if colour:
        color_mapping_dw = CategoricalColorMapper(factors=["Dog", "Wolf"], palette=["#8B959A", "#FECEA8"])
    else:
        color_mapping = CategoricalColorMapper(factors=["True", "False"], palette=["#99B898", "#FF847C"])
    
    p1 = figure(plot_width=800, plot_height=700,
                tools=('pan, wheel_zoom, reset, box_zoom'), 
                title="UMAP Projection of Image Embeddings")
    p1.title.text_font_size = '14pt'
    p1.xaxis.major_label_text_font_size = "10pt"
    p1.xaxis.major_label_text_color = '#928374'
    p1.yaxis.major_label_text_font_size = "10pt"
    p1.yaxis.major_label_text_color = '#928374'

    if colour:
        p1.circle('x', 'y', source=s1, alpha=0.7, size = size,
        color=dict(field='category_cap', transform=color_mapping_dw),  legend='category_cap')

    else:
        p1.circle('x', 'y', source=s1, alpha=0.7, size = size,
        color=dict(field='pred_is_true', transform=color_mapping), legend='pred_is_true')

    p1.legend.location = "bottom_left"
    p1.legend.label_text_font = "tahoma"
    p1.legend.orientation = "horizontal"
    p1.legend.label_text_color = "#8B959A"
    p1.legend.background_fill_color = "#1D2427"
    p1.legend.background_fill_alpha = 0.7


    p1.add_tools(HoverTool(tooltips="""
    <div>
        <div class="column">
            <img src='@bar' style='float: left; margin: 5px 5px 5px 5px width:250px;height:200px;'/>
        <div>
        <div class="column">
            <img src='@image' style='float: left; margin: 5px 5px 5px 5px width:250px;height:200px;'/>
        <div>
            <span> <strong> Predicted class: </strong> @prediction</span>
        <div>
            <span> <strong> True class: </strong>  @category </span>
        </div>
    </div>
    """))

    return p1, new_df



def roc_curve_plot(y_test, y_pred_probs):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs, pos_label=np.unique(y_test)[1])

    fig = px.area(
        x=fpr, y=tpr,
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=540, height=440,
        color_discrete_sequence=['#FECEA8']
        
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'), line_color='#1D2427',
        x0=0, x1=1, y0=0, y1=1
    )

    fig.add_annotation(
            text = f'AUC={auc(fpr, tpr):.4f}',
            x=0.85, y=0.1, showarrow=False
        )
    fig.update_annotations(bgcolor='#1D2427', font_color='#8B959A')

    fig.update_layout(
        font_color="#8B959A",
        font=dict(
            family="Tahoma",
            size=14,
            color="#8B959A"
        )
    )

    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 
                        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                        'font_color': '#8B959A',
                        #'title_text': f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
                        'title_font_color': '#8B959A'})

    fig.update_yaxes(scaleanchor="x", scaleratio=1, showgrid=False)
    fig.update_xaxes(constrain='domain', showgrid=False)

    return fig

# Function for plotting coefficients
def coef_heatmaps(model, absolute=False, height=240):
    
    if absolute:
        coefs = abs(model.coef_)
    else:
        coefs = model.coef_
    dim = int(math.sqrt(coefs.shape[1]))
    n_classes = coefs.shape[0]
    rows = n_classes

    fig = make_subplots(rows, 1, vertical_spacing=0.08)
    for i in range(n_classes):
        fig.add_trace(go.Heatmap(z=coefs[i].reshape(dim, dim),coloraxis='coloraxis1'), i+1, 1)
        #fig.update_layout(coloraxis_showscale=False, )

        fig.update_xaxes(showticklabels=False) # Hide x axis ticks 
        fig.update_yaxes(showticklabels=False) # Hide y axis ticks
        fig.update_layout(width=400, height=height,coloraxis=dict(colorscale=[(0.00, "#8B959A"),   (1, "#FECEA8")]), coloraxis_colorbar_thickness=5, showlegend=False)

    return fig

def metrics_table(df, model, help=False):

    n_classes = df['y_test'].nunique()
    cr = pd.DataFrame.from_dict(classification_report(np.array(df['y_test']), np.array(df['y_pred']), output_dict=True))
    cm = confusion_matrix(np.array(df['y_test']), np.array(df['y_pred']))
    accs = cm.diagonal()/cm.sum(axis=1)

    for i in range(n_classes+1):
        if i == 0:
            f1 = ['F1-score']
            rec = ['Recall']
            pre = ['Precision']
            sup = ['Support']
            acc = ['Accuracy']
            headers = ['Metric']
        else:
            f1.append(np.round(cr._get_value(0, i-1, takeable = True),3))
            rec.append(np.round(cr._get_value(1, i-1, takeable = True),3))
            pre.append(np.round(cr._get_value(2, i-1, takeable = True),3))
            sup.append(int(cr._get_value(3, i-1, takeable = True)))
            acc.append(f'{np.round(accs[i-1]*100)}%')
            headers.append(f"Class '{df['y_test'].unique()[i-1]}'")
    
    if help:
        headers.append('Equation')
        rec.append("<i>TP &#8725; (TP + FN) <i>")
        pre.append('<i>TP &#8725; (TP + FP) <i>')
        sup.append('<i>Num. samples in class<i>')
        f1.append('<i>2 &#8727; TP &#8725; (2 &#8727; TP + FP + FN)')
        acc.append('<i> (TP + TN) &#8725; (TP + TN + FP + FN)<i>')

    matrix = np.column_stack((f1, rec, pre, acc, sup))
    
    if not help:
        data=[go.Table(
            columnwidth = [3.6,3,3],
            header=dict(values=[f"{col}" for col in headers],
                        fill_color='#928374',
                        line_color='#fecea8',
                        line_width=1.5,
                        align='center',
                        font=dict(color='#2A363B', family="tahoma", size=18),
                        height=40
                        ),
            cells=dict(values=matrix,
                    fill_color='#2A363B',
                        line_color='#928374',
                    line_width=1.5,
                    align='left',
                    font=dict(color='#8B959A', family="tahoma", size=18),
                    height=55
                    ))
        ]
    else:
        data=[go.Table(
        columnwidth = [1.5,1, 1,2.7],
        header=dict(values=[f"{col}" for col in headers],
                    fill_color='#928374',
                    line_color='#fecea8',
                    line_width=1.5,
                    align='center',
                    font=dict(color='#2A363B', family="tahoma", size=18),
                    height=40
                    ),
        cells=dict(values=matrix,
                fill_color='#2A363B',
                    line_color='#928374',
                line_width=1.5,
                align='left',
                font=dict(color='#8B959A', family="tahoma", size=14),
                height=50
                ))
    ]

    fig = go.Figure(data=data)
    
    return fig
            

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
        colors = self.colors
        cm_colors = [colors[0], colors[3], colors[2], colors[1]]
        fig = confusion_mat(self.y_test, self.y_pred, cm_colors)
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
    
    def plot_roc_curve(self):
        fig = roc_curve_plot(self.y_test, self.y_pred_probs[:, 1])
        fig.show()

    def plot_coefs(self, absolute=False):
        fig = coef_heatmaps(self.model, absolute=absolute)
        fig.show()

    def get_metrics(self, equations=False):
        fig = metrics_table(self.df, self.model, help=equations)
        fig.show()