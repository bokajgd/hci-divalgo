import os
import streamlit as st

class Evaluate:
    def __init__(self, model):
        self.model = model
    
    def open_visualization(self):
        print(os.getcwd())
        os.system(f'streamlit run {os.path.join("divalgo", "streamlit_app.py")}')

    def heatmap(self):
        print("Now showing heatmap")
    
    def confusion(self):
        print("Now showing confusion matrix")