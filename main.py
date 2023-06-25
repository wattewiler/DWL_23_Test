# Setup
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Generate sample data
np.random.seed(42)
random.seed(42)

n_samples = 1000
width = np.random.uniform(1, 50, n_samples)
length = np.random.uniform(1, 50, n_samples)
height = np.random.uniform(1, 50, n_samples)
weight = np.random.uniform(0.1, 10, n_samples)
categories = ["GPU", "CPU", "RAM", "mainboard", "cooler", "monitor", "soundcard", "SSD", "SATA_Drives"]
category = [random.choice(categories) for _ in range(n_samples)]

CO2_emissions = np.random.uniform(1, 1000, n_samples)

data = pd.DataFrame({
    "width": width,
    "length": length,
    "height": height,
    "weight": weight,
    "category": category,
    "CO2_emissions": CO2_emissions
})

data = pd.get_dummies(data, columns=["category"], prefix="cat")

# Prepare the data
X = data.drop("CO2_emissions", axis=1)
y = data["CO2_emissions"]

# Create the models
linear_model = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# Evaluate each model using 5-fold cross-validation
def evaluate_model_with_cross_val(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    avg_mse = -np.mean(scores)
    return avg_mse

linear_mse = evaluate_model_with_cross_val(linear_model, X, y, cv=5)
rf_mse = evaluate_model_with_cross_val(rf_model, X, y, cv=5)
xgb_mse = evaluate_model_with_cross_val(xgb_model, X, y, cv=5)

# Print the average mean squared error for each model
print("Linear Regression: Mean Squared Error =", linear_mse)
print("Random Forest: Mean Squared Error =", rf_mse)
print("XGBoost: Mean Squared Error =", xgb_mse)

# Determine the best model based on the lowest MSE
models = {"Linear Regression": linear_mse, "Random Forest": rf_mse, "XGBoost": xgb_mse}
best_model = min(models, key=models.get)
print(f"\n The {best_model} model performs the best")