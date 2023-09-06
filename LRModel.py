from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import pickle

app = Flask(__name__)

# linear_regression_model = linear_model.LinearRegression()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/linear_regression/predict_score')
def linear_regression_predict_score(methods = ['GET']):
    student_attributes = []
    for input_name in request.args:
        student_attributes.append(request.args.get(input_name))
    for i in range(len(student_attributes)):
        student_attributes[i] = float(student_attributes[i])
    linear_regression_prediction = linear_regression_model.predict(np.array([student_attributes]))
    return render_template('index.html', prediction=linear_regression_prediction)

if __name__ == "__main__":
    global linear_regression_model
    pickle_in = open("final_linear_regression_model.pickle", "rb")
    linear_regression_model = pickle.load(pickle_in) #load the model
    app.run(host= "0.0.0.0")



