import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("house_price_prediction.csv")

# Define features and target
X = df[['Square Footage', 'Bedrooms', 'Age of House', 'Distance from City']]
y = df['House Price']

# Transform features for Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Calculate accuracy score (R²)
accuracy = r2_score(y_test, y_pred)
print(f"✅ Model trained and saved successfully!\n🎯 Accuracy (R² Score): {accuracy:.4f}")

# Save the model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

# Save PolynomialFeatures transformer
joblib.dump(poly, "poly_transformer.pkl")
