
from flask import Flask, render_template, request, send_file
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model  # type: ignore
from io import BytesIO
from preprocess import preprocess_data

app = Flask(__name__)
model = load_model('dnn_employee_performance_model.h5')
latest_result = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global latest_result

    if 'file' not in request.files:
        return "No file uploaded."

    file = request.files['file']

    try:
        features, emp_numbers, data_imputed = preprocess_data(file)

        predictions = model.predict(features)
        predicted_labels = np.argmax(predictions, axis=1)
        predicted_ratings = predicted_labels + 2

        output_df = pd.DataFrame({
            'EmpNumber': emp_numbers,
            'PredictedPerformanceRating': predicted_ratings
        })

        latest_result = output_df

        return """
        <h3>Prediction completed!</h3>
        <form action="/download" method="post">
            <button type="submit" class="btn btn-success">Download Results as Excel</button>
        </form>
        <br><a href='/' class='btn btn-secondary'>Back</a>
        """

    except Exception as e:
        return f"<h4 style='color:red;'>Error processing file: {e}</h4><br><a href='/'>Back</a>"

@app.route('/download', methods=['POST'])
def download():
    global latest_result

    if latest_result is None:
        return "No results available for download."

    output = BytesIO()
    latest_result.to_excel(output, index=False)
    output.seek(0)
    return send_file(output, download_name="predicted_performance.xlsx", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
