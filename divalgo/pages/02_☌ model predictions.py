import streamlit as st
import os
from PIL import Image 

def info_pred_type(prediction_type:str):
    if prediction_type == "True predictions":
        return 1, "correctly classified"
    if prediction_type == "False predictions":
        return 0, "misclassified"

def sample_image(df, error_class, prediction_type, class_list, n_images):
    class_list = [i.lower() for i in class_list]
    error_class = error_class.lower()
    # Subset dataframe to match requirements by user
    subset = df[df["y_test"]==error_class]
    bool_pred, classification = info_pred_type(prediction_type)
    subset = subset[subset["correct_prediction"]==bool_pred]
    # Sample
    sam = subset.sample(n_images)
    images = list(sam["filename"])

    # Strings to return
    if bool_pred == 1:
        other = error_class 
    elif bool_pred == 0:
        other = [i for i in class_list if i != error_class][0]
    
    if error_class == "dog":
        own = "dogs"
    elif error_class == "wolf":
        own = "wolves"
    return images, classification, own, other

def main(df):
    st.sidebar.markdown("", unsafe_allow_html=True)
    if "color_blind" not in st.session_state:
        st.session_state["color_blind"] = False
        st.session_state["value"] = False    
    color_blind=st.sidebar.checkbox("Use colourblind friendly colors", value=st.session_state["color_blind"])
    
    st.session_state["color_blind"] = color_blind

    if st.session_state["color_blind"]:
        colors = ["#44AA99", "#117733", "#DDCC77", "#997700"]
    else:
        colors = ["#99B898", "#42823C", "#FF847C", "#E84A5F", "#2A363B"]
    st.sidebar.markdown("<br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br>", unsafe_allow_html=True)

    with st.sidebar.container():
        image = Image.open(os.path.join("logos", "trans_logo.png"))
        st.image(image, use_column_width=True)

    page_title = '<p style="font-family:Tahoma; text-align:center;  color:#928374; font-size: 52px;"> Explore Model Predictions</p>'
    st.markdown(page_title, unsafe_allow_html=True)    
    page_intro = '''<p style="font-family:Tahoma; font-size: 15px;" >This page lets you explore the test data 
    in relation to the decisions made by the model. Choose whether you want to see true predictions or false predictions
    from either one of the categories using the instruments below. <br>
    ____________________________________________________________________________________________________________________</p>'''
    st.markdown(page_intro, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1: # User chooses class to see predictions for
        class_list = df["y_test"].unique()
        class_list = [i.capitalize() for i in class_list]
        error_class = st.radio("Choose class", class_list)
    with col2: # User chooses between true and false predictions 
        prediction_type = st.radio("Choose prediction type", ["True predictions", "False predictions"])

    col3, _ = st.columns((2.3,1))
    with col3: # User chooses number of images to see
        n_images = st.slider("Choose how many images you want to see", 3, 12, 6, 3)

    if "images" not in st.session_state or  "classification" not in st.session_state or  "other" not in st.session_state or "own" not in st.session_state:
        images, classification, own, other = sample_image(df, error_class, prediction_type, class_list, n_images)
        st.session_state["images"] = images

    images, classification, own, other = sample_image(df, error_class, prediction_type, class_list, n_images)
    st.session_state["images"] = images
    
    click = st.button("Get new examples")    
    if click:
        images, classification, own, other = sample_image(df, error_class, prediction_type, class_list, n_images)
        st.session_state["images"] = images

    class_str =f'<p style="font-family:Tahoma;  color:#928374; font-size: 25px;">These {own} were {classification} as <em>{other}</em></p>'
    st.markdown(class_str, unsafe_allow_html=True)
    st.image(st.session_state["images"], width=312)


if __name__ == "__main__":
    df = st.session_state["data"]
    True_false = df["y_test"] == df["y_pred"]
    df["correct_prediction"] =  [1 if T else 0 for T in True_false]
    main(df)