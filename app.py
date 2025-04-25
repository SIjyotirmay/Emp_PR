from flask import Flask, render_template, request, redirect
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import os

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
      

        filename = file.filename.lower()

        if filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file, engine='openpyxl')  # specify engine just in case
        else:
            return "Unsupported file format. Please upload .csv or .xlsx files."


        # Drop the label and index columns if present
        df = df.drop(columns=['Unnamed: 0', 'PerformanceRating'], errors='ignore')

        # Select only the model input columns
        expected_columns = [
            'pca1', 'pca2', 'pca3', 'pca4', 'pca5',
            'pca6', 'pca7', 'pca8', 'pca9', 'pca10',
            'pca11', 'pca12', 'pca13', 'pca14', 'pca15',
            'pca16', 'pca17', 'pca18', 'pca19', 'pca20',
            'pca21', 'pca22', 'pca23', 'pca24', 'pca25'
        ]

        df = df[expected_columns]

        # Run prediction
        predictions = model.predict(df)
        predicted_labels = np.argmax(predictions, axis=1)
        df['Predicted_Performance'] = predicted_labels
        df['Predicted_Performance']=  df['Predicted_Performance']+2
            

        # Convert to HTML table
        result_html = df.to_html(classes='table table-bordered', index=False)

        return f"<h2>Prediction Results:</h2>{result_html}<br><a href='/'>Back</a>"

    except Exception as e:
        return f"Error processing file: {e}"

# Your routes and logic above...

if __name__ == '__main__':
    app.run(debug=True)
