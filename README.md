
<p align="center">
    <img src="divalgo/logos/logo.png" alt="Logo" width="500" height="180">
  </a>


<br />
  <h1 align="center">Divalgo - An interactive tool for diagnosing and evaluating machine learning algorithms
 </h1>
 <h1 align="center">Human Computer Interaction Exam 2022</h1>

  <p align="center">
    Frida Hæstrup, Stine Nyhus Larsen and Jakob Grøhn Damgaard
    <br />
</p>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About the project</a></li>
    <li><a href="#getting-started">Getting started</a></li>
    <li><a href="#repository-structure">Repository structure</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About the project

This project contains the exam for the course Human Computer Interaction. 
It consists of a streamlit dashboard for visualizating trained machine learning models. 

## Abstract
An interactive tool for diagnoising and evaluating machine learning algorithms.

<!-- GETTING STARTED -->
## Getting started

For running the scripts, we recommend following the below steps in your bash-terminal. 

### Cloning repository and creating virtual environment

Clone repository and make virtual environment

```bash
git clone https://github.com/bokajgd/hci-divalgo.git
cd hci-divalgo
```

### Virtual environment

Create and activate a new virtual environment your preferred way, and install the required packages in it.
Using pip, it is done by running

```bash
python3 -m venv divalgo
source divalgo/bin/activate
pip install -r requirements.txt
```

### Run the demo
This repository comes with a Jupyter notebook (divalgo/demo.ipynb) demonstrating the visualization tool, using the dataset 'dogs vs wolfs' from Kaggle.
The dataset can be found here: https://www.kaggle.com/datasets/harishvutukuri/dogs-vs-wolves.
To run the demonstration, download the data and place the 'data' folder in the top level of this project (see folder structure below). 
Then navigate to the directory of the demonstration notebook by running

```bash
cd divalgo
```
Open and run the notebook (demo.ipynb) to follow the demonstration and reproduce the figures presented in the synopsis. 

<!-- REPOSITORY STRUCTURE -->
## Repository structure

This repository has the following structure:

```
├── .streamlit             <- folder with app setup configuration file
├── divalgo                <- main folder with class and functions                      
│   ├── .streamlit         <- folder with app setup configuration file
│   ├── logos              <- logo and symbols for pages
|   |   └── ...
│   ├── pages              <- folder containing subpages for the streamlit app
|   |   └── ...
│   ├── demo.ipynb         <- jupyter notebook demonstrating the use of the class
│   ├── divalgo_class.py   <- script with class and main functions 
│   ├── utils.py           <- script with helper-functions for the class and app 
│   └──🚪frontpage.py      <- main streamlit file and frontpage
├── data                   <- folder containing the data - dogs vs wolf from Kaggle for the demonstration     
|   ├── dogs               <- folder containing images of dogs
|   └── wolfs              <- folder containing images of wolfs
├── .gitignore                 
├── synopsis.pdf           <- the synopsis for the project
├── README.md              <- the top-level README
└── requirements.txt       <- required packages
```
