import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore  
from sklearn.preprocessing import StandardScaler

fp = "employee_performance_preprocessed_data.csv"
df = pd.read_csv(fp)

df = df.drop(columns=["Unnamed: 0"], errors='ignore')
X = df.drop(columns=["PerformanceRating"], errors='ignore')
y = df["PerformanceRating"].values

sc = StandardScaler()
X_s = sc.fit_transform(X)

mdl = load_model("dnn_employee_performance_model.h5")
pred = np.argmax(mdl.predict(X_s), axis=1)

print("Original vs Predicted Performance Ratings:")
for i, (t, p) in enumerate(zip(y, pred)):
    print(f"Sample {i+1}: True Class {t}, Predicted Class {p + 2}")

