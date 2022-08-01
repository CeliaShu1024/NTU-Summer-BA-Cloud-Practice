import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import ensemble
from sklearn import neural_network
from sklearn import svm
from sklearn import tree
from flask import Flask, request, render_template
from sklearn.metrics import mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("DBS_SingDollar.csv", encoding='utf-8')
x = np.array([1.39]).reshape(-1, 1)

model = joblib.load("regression")
print(model.predict(x))

model = joblib.load("tree")
print(model.predict(x))