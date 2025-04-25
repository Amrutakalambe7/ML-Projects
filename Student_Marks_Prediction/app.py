from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('student_marks_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            study_hours = float(request.form['hours'])

            # Validate input range
            if study_hours < 1 or study_hours > 24:
                prediction = 'Please enter valid hours between 1 to 24.'
            else:
                pred = model.predict([[study_hours]])[0][0]
                prediction = round(pred, 2)
        except ValueError:
            prediction = "Invalid input! Please enter a numeric value."

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
