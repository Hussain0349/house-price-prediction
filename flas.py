# app.py
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Model training function (based on your code)
def train_model():
    df1 = pd.read_csv('dha.csv')
    df1 = df1.drop(columns=['Initial Amount', 'Monthly Installment', 'Remaining Installments', 'Added', 'Location', 'Purpose'])
    df1['Bath(s)'] = df1['Bath(s)'].str.replace('-', '0').astype(int)
    df1['Bedroom(s)'] = df1['Bedroom(s)'].str.replace('-', '0').astype(int)
    df1['Total Room'] = df1['Bedroom(s)'] + df1['Bath(s)']
    df1 = df1.drop(columns=['Bath(s)', 'Bedroom(s)'])
    df1['Price'] = df1['Price'].str.split('R').str[1].str.split('C').str[0].str.split('L').str[0].str.strip().astype(float)
    df1['Area'] = df1['Area'].str.split('Kanal').str[0].str.split('Marla').str[0].str.strip().astype(float)
    df1['Area'] = df1['Area'].apply(lambda x: x * 20 if x == 1.0 else x)
    df1['Type'] = df1['Type'].replace({'House': 2, 'Flat': 1})
    df1['Prices'] = np.log(df1['Price'])
    df1 = df1.drop(columns='Price')

    x = df1[['Type', 'Area', 'Total Room']]
    y = df1['Prices']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, train_size=0.9, random_state=20)

    model = LinearRegression()
    model.fit(x_train, y_train)

    return model

# Train the model when the application starts
model = train_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the form data
        property_type = int(request.form['type'])
        area = float(request.form['area'])
        total_rooms = int(request.form['total_rooms'])

        # Create a DataFrame from the input
        input_data = pd.DataFrame([[property_type, area, total_rooms]], columns=['Type', 'Area', 'Total Room'])

        # Make the prediction (log-transformed)
        prediction_log = model.predict(input_data)[0]

        # Convert the prediction back to original scale
        predicted_price = np.exp(prediction_log)

        # Return the result to the template
        return render_template('index.html', prediction=round(predicted_price, 2))

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)