import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import joblib, inspect

k_nearest_model = joblib.load("final_knn_model.pkl")

a = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
a = np.array(a)
print(a)
df = pd.DataFrame(np.array(a), columns = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences","G1", "G2", "G3"])
print(df)
y_pred = k_nearest_model.predict(df)
print(y_pred)
