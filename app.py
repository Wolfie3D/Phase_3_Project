from flask import Flask, render_template, request
import joblib 
import pandas as pd

app = Flask(__name__)

# Load the ETL pipeline and the trained RandomForestRegressor model
rf_model = joblib.load('MODELS\\lr_model.joblib')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/Result', methods=['POST'])
def predictions():
    if request.method == 'POST':
        # Collect user input from the form
        County = request.form['County']
        Neighborhood = request.form['Neighborhood']
        bedrooms = request.form['bedrooms']
        bathrooms = request.form['bathrooms']
        Size = request.form['Size']
        Quality = request.form['Quality']
        Type = request.form['Type']
        Furnished = request.form['Furnished']

        # Create a DataFrame from user input
        user_data = pd.DataFrame({
            'County': [County],
            'Neighborhood': [Neighborhood],
            'bedrooms': [bedrooms],
            'bathrooms': [bathrooms],
            'Size': [Size],
            'Quality': [Quality],
            'Type': [Type],
            'Furnished': [Furnished],
        })

        prediction = rf_model.predict(user_data)

        # Display the prediction to the user
        return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
