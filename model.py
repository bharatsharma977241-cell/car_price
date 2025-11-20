import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load dataset
df = pd.read_csv(r'C:\Users\bhara\OneDrive\Desktop\car project 1\CarPrice_Assignment.csv')

# Features and target
x = df[['symboling','enginetype','enginesize','horsepower','peakrpm',
        'highwaympg','citympg','cylindernumber','CarName','fueltype','carbody']]
y = df['price']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2304)

# Separate numeric and categorical features
num_features = ['symboling','enginesize','horsepower','peakrpm','highwaympg','citympg']
cat_features = ['cylindernumber','enginetype','CarName','fueltype','carbody']

# Preprocessing
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
scaler = StandardScaler()

ct = ColumnTransformer(transformers=[
    ('ohe_transform', ohe, cat_features),
    ('scaler_transform', scaler, num_features)
], remainder="drop")

# Pipeline with Linear Regression
model_pipeline = Pipeline(steps=[
    ('Preprocess', ct),
    ('model', LinearRegression())
])

# Train
model_pipeline.fit(x_train, y_train)

# Evaluate
y_pred = model_pipeline.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Model trained ✅")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Save model
joblib.dump(model_pipeline, r"C:\Users\bhara\OneDrive\Desktop\car project 1\car_price_model.h5")
print("Model saved as car_price_model.h5")
