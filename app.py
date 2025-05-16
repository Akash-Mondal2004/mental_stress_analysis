from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import joblib
import traceback

app = Flask(__name__)

# Load model and scaler
model = joblib.load('./models/random_forest_modelmodel2.pkl')
scaler = joblib.load('./models/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from form
        data = [float(request.form[key]) for key in request.form.keys()]
        columns = ['acc_x', 'acc_y', 'acc_z', 'bvp', 'eda', 'hr', 'temp', 'ibi']
        df = pd.DataFrame([data], columns=columns)
        
        # Scale and predict
        scaled_data = scaler.transform(df)
        prediction = model.predict(scaled_data)
        
        result = "Stressed" if prediction[0] == 1 else "Not Stressed"
        return render_template('index.html', prediction_text=f'Prediction: {result}')
    except Exception as e:
        print(traceback.format_exc())
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
