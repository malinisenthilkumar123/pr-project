from flask import Flask, request, render_template
import pickle
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and polynomial transformer
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

poly = joblib.load("poly_transformer.pkl")  # Load transformer for polynomial features

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None
    error_message = None

    if request.method == "POST":
        try:
            # Get form inputs and convert them to float
            square_footage = float(request.form["square_footage"])
            bedrooms = float(request.form["bedrooms"])
            age_of_house = float(request.form["age_of_house"])
            distance_from_city = float(request.form["distance_from_city"])

            # Transform input using PolynomialFeatures
            input_features = np.array([[square_footage, bedrooms, age_of_house, distance_from_city]])
            input_transformed = poly.transform(input_features)

            # Predict house price
            prediction = round(model.predict(input_transformed)[0], 2)

        except ValueError:
            error_message = "❌ Invalid input! Please enter valid numbers."

    return render_template("index.html", prediction=prediction, error_message=error_message)

if __name__ == "__main__":
    app.run(debug=True)
