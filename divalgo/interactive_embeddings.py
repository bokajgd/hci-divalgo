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
from bokeh.resources import INLINE
import bokeh.io
from io import BytesIO
import io
import base64

pwd = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
dogs = sorted(os.listdir(os.path.join(pwd, "data/dogs")))
wolves =  sorted(os.listdir(os.path.join(pwd, "data/wolves")))

# Load data
# dogs = sorted(os.listdir(os.path.join("data", "dogs")))
# wolves =  sorted(os.listdir(os.path.join("data", "wolves")))

# Preprocessing
img_size = 50
dogs_images = []
wolves_images = []

for i in dogs:
    if os.path.isfile(os.path.join(pwd, "data/dogs", f"{i}")):
        img = Image.open(os.path.join(pwd, "data/dogs", f"{i}")).convert('L')
        img = img.resize((img_size,img_size), Image.ANTIALIAS)
        img = np.asarray(img)/255.0
        dogs_images.append(img)

for i in wolves:
    if os.path.isfile(os.path.join(pwd, "data/wolves", f"{i}")):
        img = Image.open(os.path.join(pwd, "data/wolves", f"{i}")).convert('L')
        img = img.resize((img_size,img_size), Image.ANTIALIAS)
        img = np.asarray(img)/255.0
        wolves_images.append(img)

# for i in dogs:
#     if os.path.isfile(os.path.join("data", "dogs", f"{i}")):
#         img = Image.open(os.path.join("data", "dogs", f"{i}")).convert('L')
#         img = img.resize((img_size,img_size), Image.ANTIALIAS)
#         img = np.asarray(img)/255.0
#         dogs_images.append(img)

# for i in wolves:
#     if os.path.isfile(os.path.join("data", "wolves", f"{i}")):
#         img = Image.open(os.path.join("data", "wolves", f"{i}")).convert('L')
#         img = img.resize((img_size,img_size), Image.ANTIALIAS)
#         img = np.asarray(img)/255.0
#         wolves_images.append(img)


#x = np.concatenate((dogs_images,wolves_images),axis=0) # data
#x = x.reshape(x.shape[0],x.shape[1]*x.shape[2]) #3D to 2D
#y = np.asarray(label) # corresponding labels
#y = y.reshape(y.shape[0], 1)

#X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Manual train-test split (to track filenames)
X_train = np.asarray(dogs_images[0:800] + wolves_images[0:800])
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])
X_test = np.asarray(dogs_images[800:1000] + wolves_images[800:1000])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2])
y_train = np.asarray(["dog" for y in range(800)] + ["wolf" for y in range(800)])
y_train = y_train.reshape(y_train.shape[0],1)
y_test_ar = np.asarray(["dog" for y in range(200)] + ["wolf" for y in range(200)])
y_test = y_test_ar.reshape(y_test_ar.shape[0],1)

y_train, y_test = [k.T for k in [y_train, y_test]]

filenames_test = [os.path.join(pwd, "data/dogs", d) for d in dogs[800:1000]] + [os.path.join(pwd, "data/wolves", w) for w in wolves[800:1000]]

#filenames_test = [os.path.join("data", "dogs", d) for d in dogs[800:1000]] + [os.path.join("data", "wolves", w) for w in wolves[800:1000]]

# Train model
model = LogisticRegression(penalty='none', tol=0.1).fit(X_train, y_train[0])

# Take the trained model and use to predict test class
y_pred = model.predict(X_test)
y_pred_brobs = model.predict_proba(X_test)

# Calculate evaluation metrics
# cm = metrics.classification_report(y_test[0], y_pred)



# Create embeddings with pytorch
img2vec = Img2Vec()

image_arrays = []
vectors = []
category = []
preds = []
for i, img in tqdm(enumerate(filenames_test)):
    an_image = Image.open(img).resize((200, 150), Image.BICUBIC)
    image_arrays.append(np.asarray(an_image).astype(np.uint8))
    vectors.append(img2vec.get_vec(an_image, tensor=False))
    preds.append(y_pred[i])
    category.append(y_test_ar[i])


# Project embeddings to 2D space with UMAP
embeddings = np.vstack(vectors)

reducer = umap.UMAP()
embeddings_2d = reducer.fit_transform(embeddings).tolist()

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

 

df = pd.DataFrame({
        'x': [embeddings_2d[x][0] for x in range(len(embeddings_2d))],
        'y': [embeddings_2d[y][1] for y in range(len(embeddings_2d))],
        'file': filenames_test,
        'image': list(map(np_image_to_base64, image_arrays)),
        'bar': list(map(prob_barplot, y_pred_brobs)),
        'pred_probs': [probs for probs in y_pred_brobs],
        'prediction': preds,
        "category": category,
        "pred_is_true": [str(preds[i] == category[i]) for i in range(len(preds))]
        })








# Interactive plot

datasource = ColumnDataSource(df)

color_mapping = CategoricalColorMapper(factors=["True", "False"], palette=["#99B898", "#FF847C"])

plot_figure = figure(
    title='UMAP projection of image embeddings',
    plot_width=800,
    plot_height=800,
    tools=('pan, wheel_zoom, reset, lasso_select')
)

plot_figure.add_tools(HoverTool(tooltips="""
<div>
    <div>
        <img src='@image' style='float: left; margin: 5px 5px 5px 5px'/>
    </div>
    <div>
        <span style='font-size: 12px'> <strong> Predicted class: </strong> @prediction</span>
    </div>
    <div>
        <span style='font-size: 12px'> <strong> True class: </strong>  @category </span>
    </div>
</div>
"""))

#plot_figure.circle('x', 'y', source=datasource, alpha=0.6, color=dict(field="pred_is_true", transform=color_mapping))

plot_figure.scatter(
    'x',
    'y',
    source=datasource,
    color=dict(field="pred_is_true", transform=color_mapping),
    line_alpha=0.6,
    fill_alpha=0.6,
    size=8
)

#source.selected.indices #this is what the lasso-selected points should be saved under

show(plot_figure)





from bokeh.layouts import row
from bokeh.models import CustomJS, ColumnDataSource
from bokeh.plotting import figure
from bokeh.models.widgets import Button
from bokeh.io import curdoc

x = df['x'].to_list()
y = df['y'].to_list()
tf = df['pred_is_true'].to_list()

s1 = ColumnDataSource(data=df)
color_mapping = CategoricalColorMapper(factors=["True", "False"], palette=["#99B898", "#FF847C"])

p1 = figure(plot_width=400, plot_height=400, x_range=(min(x), max(x)), y_range=(min(y), max(y)),
            tools=('pan, wheel_zoom, reset', 'lasso_select'), 
            title="UMAP projection of image embeddings")
p1.circle('x', 'y', source=s1, alpha=0.6,
          color=dict(field='pred_is_true', transform=color_mapping))

s2 = ColumnDataSource(data=dict(x=[], y=[]))
p2 = figure(plot_width=400, plot_height=400, x_range=(min(x), max(x)), y_range=(min(y), max(y)),
            tools="", title="Watch Here")
p2.circle('x', 'y', source=s2, alpha=0.6)

s1.selected.js_on_change('indices', CustomJS(args=dict(s1=s1, s2=s2), code="""
        var inds = cb_obj.indices;
        var d1 = s1.data;
        var d2 = s2.data;
        d2['x'] = []
        d2['y'] = []
        for (var i = 0; i < inds.length; i++) {
            d2['x'].push(d1['x'][inds[i]])
            d2['y'].push(d1['y'][inds[i]])
        }
        s2.change.emit();
    """)
)

p1.add_tools(HoverTool(tooltips="""
<div>
    <div>
        <img src='@image' style='float: left; margin: 5px 5px 5px 5px'/>
    </div>
    <div>
        <span style='font-size: 12px'> <strong> Predicted class: </strong> @prediction</span>
    </div>
    <div>
        <span style='font-size: 12px'> <strong> True class: </strong>  @category </span>
    </div>
</div>
"""))

layout = row(p1, p2)

show(layout)





x = df['x'].to_list()
y = df['y'].to_list()
tf = df['pred_is_true'].to_list()
img_array = np.array(X_test).reshape(400, 50, 50)

s1 = ColumnDataSource(data=dict(x=x, y=y, pred_is_true=tf, img_array=img_array))
color_mapping = CategoricalColorMapper(factors=["True", "False"], palette=["#99B898", "#FF847C"])

p1 = figure(plot_width=400, plot_height=400, x_range=(min(x), max(x)), y_range=(min(y), max(y)),
            tools=('pan, wheel_zoom, reset', 'lasso_select'), 
            title="UMAP projection of image embeddings")
p1.circle('x', 'y', source=s1, alpha=0.6,
          color=dict(field='pred_is_true', transform=color_mapping))

s2 = ColumnDataSource(data=dict(image=[]))
# p2 = figure(plot_width=400, plot_height=400, x_range=(min(x), max(x)), y_range=(min(y), max(y)),
#             tools="", title="Watch Here")
# p2.circle('x', 'y', source=s2, alpha=0.6)


p2 = figure(x_range=(0, 35), y_range=(0, 35))
p2.image(source=s2, image='image', x=0, y=0, dw=10, dh=10, palette="Spectral11", name="Image Glyph")

s1.selected.js_on_change('indices', CustomJS(args=dict(s1=s1, s2=s2), code="""
        var inds = cb_obj.indices;
        var d1 = s1.data;
        var data = s2.data;
        data['image'] = [...Array(50)].map(e => Array(50));
        for (var i = 0; i < inds.length; i++) {
            data['image'].push(d1['img_array'][inds[i]])
        }
        data.change.emit();
    """)
)


# p1.add_tools(HoverTool(tooltips="""
# <div>
#     <div>
#         <img src='@image' style='float: left; margin: 5px 5px 5px 5px'/>
#     </div>
#     <div>
#         <span style='font-size: 12px'> <strong> Predicted class: </strong> @prediction</span>
#     </div>
#     <div>
#         <span style='font-size: 12px'> <strong> True class: </strong>  @category </span>
#     </div>
# </div>
# """))

layout = row(p1, p2)

show(layout)











s1 = ColumnDataSource(data=df)
color_mapping = CategoricalColorMapper(factors=["True", "False"], palette=["#99B898", "#FF847C"])

p1 = figure(plot_width=800, plot_height=800,
            tools=('pan, wheel_zoom, reset, box_zoom'), 
            title="UMAP projection of image embeddings")
p1.circle('x', 'y', source=s1, alpha=0.6, size = 10,
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

show(p1)

