from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
import joblib

# Load model and preprocessing data
model = tf.keras.models.load_model('model/house_price_model.h5', compile=False)
X_mean, X_std, y_mean, y_std = joblib.load('model/preprocessing.pkl')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            marlas = float(request.form['marlas'])
            portions = float(request.form['portions'])
            age = float(request.form['age'])

            # Normalize input
            input_data = np.array([[marlas, portions, age]])
            input_norm = (input_data - X_mean) / X_std

            # Predict
            prediction_norm = model.predict(input_norm)
            prediction = prediction_norm[0][0] * y_std + y_mean

            return render_template('index.html',
                                   prediction=f"Predicted House Price: Rs {prediction:.2f}M")
        except:
            return render_template('index.html',
                                   prediction="Invalid input. Please enter numbers.")
    return render_template('index.html', prediction='')

if __name__ == '__main__':
    app.run(debug=True)
