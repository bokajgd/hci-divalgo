import os
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as implt
import seaborn as sns
import cv2 as cv
from tqdm import tqdm
import random
from PIL import Image
import warnings
warnings.filterwarnings('ignore') # filter warnings

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import metrics

import umap.umap_ as umap #pip install umap-learn
from img2vec_pytorch import Img2Vec #also pip install Pillow, scikit-learn

from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper, CustomJS, Circle
from bokeh.palettes import Spectral10
import bokeh.io
import io
import base64



# Make function for making and saving bar plots
def prob_barplot(probabilities: np.array):

    s = io.BytesIO()

    plt.figure()
    ax = sns.barplot(x=['Wolf', 'Dog'], y=probabilities*100, 
    palette=['#6593B1', '#FECEA8'] )
    sns.despine()
    sns.set_style("whitegrid")

    for bar, label in zip(ax.patches, probabilities):
        full_lab = f"{np.round(label*100, 3)}%"
        x = bar.get_x()
        width = bar.get_width()
        height = bar.get_height()
        ax.text(x+width/2., height + 2, full_lab, ha="center") 

    plt.savefig(s, format='png', bbox_inches="tight")
    plt.close()
    s = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
    return 'data:image/png;base64,%s' % s

# Prepare data for interactive plot
def np_image_to_base64(im_matrix):
    im = Image.fromarray(im_matrix)
    buffer = io.BytesIO()
    im.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image
    return im_url



