from flask import Flask, render_template, request
from sklearn.externals import joblib
import pandas as pd

app = Flask(__name__)

# Load the ETL pipeline and the trained RandomForestRegressor model
etl_pipeline = joblib.load('etl_pipeline.joblib')
rf_model = joblib.load('best_model_rf.joblib')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Collect user input from the form
        cost = request.form['cost']
        title = request.form['title']
        description = request.form['description']
        location = request.form['location']
        bedrooms = request.form['bedrooms']
        bathrooms = request.form['bathrooms']
        furnished = request.form['furnished']
        space = request.form['space']

        # Create a DataFrame from user input
        user_data = pd.DataFrame({
            'Cost': [cost],
            'Title': [title],
            'Description': [description],
            'Location': [location],
            'Bedrooms': [bedrooms],
            'Bathrooms': [bathrooms],
            'Furnished': [furnished],
            'Space': [space],
        })

        # Apply ETL transformations
        transformed_data = etl_pipeline.transform(user_data)

        # Make prediction using the RandomForestRegressor model
        prediction = rf_model.predict(transformed_data)[0]

        # Display the prediction to the user
        return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
