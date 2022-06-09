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

# Function for generating embeddings
def get_embeddings(df):
    # Create embeddings with pytorch
    img2vec = Img2Vec()

    image_arrays = []
    vectors = []

    for i, img in tqdm(enumerate(df['filename'])):
        an_image = Image.open(img).resize((200, 150), Image.BICUBIC)
        image_arrays.append(np.asarray(an_image).astype(np.uint8))
        vectors.append(img2vec.get_vec(an_image, tensor=False))


    # Project embeddings to 2D space with UMAP
    embeddings = np.vstack(vectors)

    reducer = umap.UMAP()
    embeddings_2d = reducer.fit_transform(embeddings).tolist()

    return embeddings_2d, image_arrays

# Make function for making and saving bar plots
def prob_barplot(probabilities: np.array):

    s = io.BytesIO()

    plt.figure()
    ax = sns.barplot(x=['Dog', 'Wolf'], y=probabilities*100, 
    palette=['#8B959A', '#FECEA8'] )
    ax.tick_params(axis="y",which="major",labelsize=12,color="#2A363B")
    ax.tick_params(axis="x", which="major", labelsize=18, color="#2A363B")
    sns.despine()
    sns.set_style("whitegrid")

    for bar, label in zip(ax.patches, probabilities):
        full_lab = f"{np.round(label*100, 3)}%"
        x = bar.get_x()
        width = bar.get_width()
        height = bar.get_height()
        ax.text(x+width/2., height + 2, full_lab, fontsize=18, color="#2A363B", family="tahoma", ha="center") 

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


# Function for generating df for plot
def get_embedding_df(base_df, embeddings_2d, image_arrays):
    
    pred_probs = base_df[['prob0', 'prob1']].to_numpy()

    df = pd.DataFrame({
            'x': [embeddings_2d[x][0] for x in range(len(embeddings_2d))],
            'y': [embeddings_2d[y][1] for y in range(len(embeddings_2d))],
            'file': base_df['filename'],
            'image': list(map(np_image_to_base64, image_arrays)),
            'bar': list(map(prob_barplot, pred_probs)),
            'prediction': base_df['y_pred'],
            "category": base_df['y_test'],
            "pred_is_true": [str(base_df['y_pred'][i] == base_df['y_test'][i]) for i in range(len(base_df['y_pred']))]
            })

    df['category_cap'] = df['category'].copy().str.capitalize()

    return df

