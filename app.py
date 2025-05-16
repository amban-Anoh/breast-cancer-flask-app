from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(request.form[f'feature{i}']) for i in range(1, 10)]
    prediction = model.predict([features])[0]
    diagnosis = 'Malignant' if prediction == 1 else 'Benign'
    return render_template('index.html', prediction=diagnosis)
