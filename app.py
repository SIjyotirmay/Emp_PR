from flask import Flask, render_template, request, redirect
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import os
from preprocess import preprocess_data

app = Flask(__name__)
model = load_model('dnn_employee_performance_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded."

    file = request.files['file']

    try:
      
        # Preprocess the uploaded file
        processed_data, emp_numbers, original_data = preprocess_data(file)

       
        # Run prediction
        predictions = model.predict(processed_data)
        predicted_labels = np.argmax(predictions, axis=1)
        original_data['Predicted_Performance'] = predicted_labels + 2
        result_html = original_data[['EmpNumber', 'Predicted_Performance']].to_html(classes='table table-bordered', index=False)

        return f"<h2>Prediction Results:</h2>{result_html}<br><a href='/'>Back</a>"

    except Exception as e:
        return f"Error processing file: {e}"

# Your routes and logic above...

if __name__ == '__main__':
    app.run(debug=True)
