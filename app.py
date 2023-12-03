from flask import Flask, render_template, request
from joblib import load
import pandas as pd

app = Flask(__name__)

# Load the ETL pipeline and the trained RandomForestRegressor model
rf_model = load('/MODELS/lr_model.joblib')


@app.route('/index.html')
def index():
    return render_template('index.html')


@app.route('/result.html', methods=['POST'])
def predictions():
    if request.method == 'POST':

        # Collect user input from the form
        County = request.form['County']
        Neighborhood = request.form['Neighborhood']
        Bedrooms = request.form['Bedrooms']
        Bathrooms = request.form['Bathrooms']
        Size = request.form['Size']
        Quality = request.form['Quality']
        Type = request.form['Type']
        Furnished = request.form['Furnished']

        # Create a DataFrame from user input
        user_data = pd.DataFrame({
            'County': [County],
            'Neighborhood': [Neighborhood],
            'Bedrooms': [Bedrooms],
            'Bathrooms': [Bathrooms],
            'Size': [Size],
            'Quality': [Quality],
            'Type': [Type],
            'Furnished': [Furnished],
        })

        # Make prediction using the RandomForestRegressor model
        prediction = rf_model.predict(user_data)[0]

        # Display the prediction to the user
        return render_template('/templates/result.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
