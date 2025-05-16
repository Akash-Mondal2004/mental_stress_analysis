from flask import Flask, Response, request, render_template, send_from_directory
import numpy as np
import pandas as pd
import joblib
import traceback
import os
import json

app = Flask(__name__)

# Load model and scaler
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
model = joblib.load(os.path.join(MODEL_PATH, 'random_forest_modelmodel2.pkl'))
scaler = joblib.load(os.path.join(MODEL_PATH, 'scaler.pkl'))

# Define a handler for Vercel serverless function
def handler(request):
    if request.method == 'POST':
        try:
            # Parse form data
            form_data = request.form.to_dict()
            data = [float(form_data[key]) for key in ['acc_x', 'acc_y', 'acc_z', 'bvp', 'eda', 'hr', 'temp', 'ibi']]
            columns = ['acc_x', 'acc_y', 'acc_z', 'bvp', 'eda', 'hr', 'temp', 'ibi']
            df = pd.DataFrame([data], columns=columns)
            
            # Scale and predict
            scaled_data = scaler.transform(df)
            prediction = model.predict(scaled_data)
            
            result = "Stressed" if prediction[0] == 1 else "Not Stressed"
            return {"prediction": result}
        except Exception as e:
            return {"error": str(e)}
    else:
        # For GET requests, return a simple message
        return {"message": "Send a POST request to this endpoint for predictions"}

# Handle API requests
def api_handler(request):
    response = handler(request)
    return Response(
        json.dumps(response),
        mimetype='application/json'
    )

# Define routes
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join('public', path)):
        return send_from_directory('public', path)
    else:
        return send_from_directory('public', 'index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    response = handler(request)
    if request.headers.get('Content-Type') == 'application/json':
        return response
    else:
        # Render the HTML template with the prediction
        prediction_text = f"Prediction: {response.get('prediction', 'Error')}"
        return render_template('index.html', prediction_text=prediction_text)

# For local development
if __name__ == '__main__':
    app.run(debug=True)