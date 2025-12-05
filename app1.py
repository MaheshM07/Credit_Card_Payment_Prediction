from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("CC_prediction_model.pkl")
scaler = joblib.load("CC_prediction_scaler.pkl")

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = ""
    if request.method == 'POST':
        try:
            # Get form data
            data = [
                float(request.form['limit_bal']),
                float(request.form['gender']),     
                float(request.form['marriage']),
                float(request.form['pay_0']),
                float(request.form['pay_2']),
                float(request.form['pay_3']),
                float(request.form['bill_amt1']),
                float(request.form['bill_amt2']),
                float(request.form['bill_amt3']),
                float(request.form['pay_amt1']),
                float(request.form['pay_amt2']),
                float(request.form['pay_amt3'])
            ]

            scaled_data = scaler.transform([data])

            pred = model.predict(scaled_data)
            prediction = " The Amount will be Paid on Time" if pred[0] == 1 else " The Amount will not be Paid on Time"
        except Exception as e:
            prediction = f"⚠️ Error: {e}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
