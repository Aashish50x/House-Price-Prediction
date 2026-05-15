import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ---------------------------------------------------------
# Step 1: Load the Data
# ---------------------------------------------------------
print("1. Loading Data...")
# Load the dataset (Make sure you have your dataset saved as 'house_data.csv')
# For this example, we'll use a sample CSV generated earlier.
df = pd.DataFrame()
try:
    df = pd.read_csv('house_data.csv')
    print(f"Data loaded successfully! Number of records: {len(df)}")
except FileNotFoundError:
    print("Error: 'house_data.csv' not found. Please provide the dataset.")
    exit()

# ---------------------------------------------------------
# Step 2: Separate Features (X) and Target (y)
# ---------------------------------------------------------
print("\n2. Separating Features and Target...")
# 'Price' is what we want to predict
X = df.drop('Price', axis=1)
y = df['Price']

# Identify numerical and categorical columns
numeric_features = ['BHK', 'Size_sqft', 'Floors', 'Bathrooms', 'Age_years']
categorical_features = ['City', 'Furnishing', 'Property_Type', 'Parking']

# ---------------------------------------------------------
# Step 3: Clean and Preprocess the Data
# ---------------------------------------------------------
print("3. Preprocessing Data (Handling Missing Values and Encoding)...")

# Setup pipelines for numerical and categorical data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # Fill missing numbers with the median
    ('scaler', StandardScaler())                   # Scale numbers so they are on a similar range
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # Fill missing text with the most frequent value
    ('onehot', OneHotEncoder(handle_unknown='ignore'))    # Convert text to numerical flags (0 or 1)
])

# Combine them into a single preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# ---------------------------------------------------------
# Step 4: Split the Data into Training and Testing Sets
# ---------------------------------------------------------
print("\n4. Splitting Data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training data size: {len(X_train)} records")
print(f"Testing data size: {len(X_test)} records")

# ---------------------------------------------------------
# Step 5: Train Simple Machine Learning Models
# ---------------------------------------------------------
print("\n5. Training and Evaluating Models...")

# Dictionary of models to train
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=25, max_depth=12, random_state=42)
}

best_model_name = ""
best_model_score = -float('inf')
best_pipeline = None

for name, model in models.items():
    # Create a pipeline that combines preprocessing and the model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions on the test set
    predictions = pipeline.predict(X_test)
    
    # Calculate Evaluation Metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print(f"--- {name} ---")
    print(f"Mean Absolute Error (MAE): ₹{mae:,.2f}")
    print(f"Root Mean Squared Error (RMSE): ₹{rmse:,.2f}")
    print(f"R-squared: {r2:.4f}\n")
    
    # Keep track of the best model based on R-squared
    if r2 > best_model_score:
        best_model_score = r2
        best_model_name = name
        best_pipeline = pipeline

print(f"-> The best model is **{best_model_name}** with an R-squared of {best_model_score:.4f}.")

# ---------------------------------------------------------
# Step 6: Make a Sample Prediction
# ---------------------------------------------------------
print(f"\n6. Making a Prediction using the Best Model ({best_model_name})...")

# Let's create a sample house that a user might input
sample_house = pd.DataFrame([{
    'BHK': 3,
    'City': 'Bangalore',
    'Size_sqft': 1500,
    'Floors': 5,
    'Bathrooms': 2,
    'Furnishing': 'Semi-furnished',
    'Property_Type': 'Apartment',
    'Age_years': 5,
    'Parking': 'Yes'
}])

print("Sample Input House:")
print(sample_house.to_string(index=False))

# Predict the price
predicted_price = best_pipeline.predict(sample_house)
print(f"\nPredicted House Price: ₹{predicted_price[0]:,.2f}")

# Save the trained model for the web application
print("\nSaving the best model to 'house_price_model.pkl'...")
joblib.dump(best_pipeline, 'house_price_model.pkl')
print("Model saved successfully!")
