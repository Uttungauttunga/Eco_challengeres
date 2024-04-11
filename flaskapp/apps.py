from flask import Flask, request, jsonify

from flask_cors import CORS
import pandas as pd
import math
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)
CORS(app)

# Load the trained model and preprocessing transformers
model = RandomForestRegressor(n_estimators=100, random_state=0)

training = pd.read_csv('cf2.csv')
testing = pd.read_csv('td(1).csv')

training['Weight_Price_Ratio'] = training['Weight'] / training['Price']
testing['Weight_Price_Ratio'] = testing['Weight'] / testing['Price']

features = ['Weight', 'Price', 'Screen_Size', 'Weight_Price_Ratio', 'Category']

X_train = training[features]
y_train = training['Carbon_Emissions']

ct = ColumnTransformer(
    [('onehot', OneHotEncoder(drop='first'), ['Category'])],
    remainder='passthrough'
)

X_train_encoded = ct.fit_transform(X_train)
model.fit(X_train_encoded, y_train)


@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from request
    data = request.json
    weight = data['weight']
    screenSize = data['screenSize']
    price = data['price']
    category = data['category']

    # Define some random input values
    random_input = {'Weight': weight, 'Price': price , 'Screen_Size': screenSize, 'Category':category}

# Calculate Weight_Price_Ratio
    

    random_input['Weight_Price_Ratio'] = float(random_input['Weight'] )/ float(random_input['Price'])

  # Prepare the input for prediction
    input_df = pd.DataFrame([random_input], columns=features)
    input_encoded = ct.transform(input_df)

    # Predict the carbon emissions
    predicted_carbon_emissions = model.predict(input_encoded)[0]

    rounded_carbon_footprint = round(predicted_carbon_emissions, 2)
   
    # Return prediction as JSON response
    return jsonify({'carbonFootprint':rounded_carbon_footprint })


if __name__ == '__main__':
    app.run(debug=True)
