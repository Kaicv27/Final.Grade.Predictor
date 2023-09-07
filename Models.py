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

@app.route('/k_nearest/predict_score')
def k_nearest_predict_score(methods = ['GET']):
    student_attributes = []
    for input_name in request.args:
        student_attributes.append(request.args.get(input_name))
    for i in range(len(student_attributes)):
        student_attributes[i] = float(student_attributes[i])
    k_nearest_prediction = k_nearest_model.predict(np.array([student_attributes]))
    return render_template('index.html', prediction=k_nearest_prediction)

@app.route('/random_forest/predict_score')
def random_forest_predict_score(methods = ['GET']):
    student_attributes = []
    for input_name in request.args:
        student_attributes.append(request.args.get(input_name))
    for i in range(len(student_attributes)):
        student_attributes[i] = float(student_attributes[i])
    random_forest_prediction = random_forest_model.predict(np.array([student_attributes]))
    return render_template('index.html', prediction=random_forest_prediction)


if __name__ == "__main__":
    global linear_regression_model
    pickle_in = open("final_linear_regression_model.pickle", "rb")
    linear_regression_model = pickle.load(pickle_in) #load the linear_regression model
    app.run(host= "0.0.0.0")

if __name__ == "__main__":
    global k_nearest_model
    pickle_in = open("final_knn_model.pickle", "rb")
    k_nearest_model = pickle.load(pickle_in) #load the k_nearest model
    app.run(host= "0.0.0.0")

if __name__ == "__main__":
    global random_forest_model
    pickle_in = open("final_rf_model.pickle", "rb")
    random_forest_model = pickle.load(pickle_in) #load the random_forest model
    app.run(host= "0.0.0.0")


