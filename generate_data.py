import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 15000

cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Pune', 'Hyderabad', 'Kolkata', 'Ahmedabad', 'Jaipur', 'Surat']
furnishing_types = ['Unfurnished', 'Semi-furnished', 'Fully-furnished']
property_types = ['Apartment', 'Independent House', 'Villa']

data = {
    'BHK': np.random.randint(1, 6, n_samples),
    'City': np.random.choice(cities, n_samples),
    'Size_sqft': np.random.randint(500, 5000, n_samples),
    'Floors': np.random.randint(1, 15, n_samples),
    'Bathrooms': np.random.randint(1, 5, n_samples),
    'Furnishing': np.random.choice(furnishing_types, n_samples),
    'Property_Type': np.random.choice(property_types, n_samples),
    'Age_years': np.random.randint(0, 30, n_samples),
    'Parking': np.random.choice(['Yes', 'No'], n_samples),
}

df = pd.DataFrame(data)

# Introduce some missing values
df.loc[np.random.choice(n_samples, 20), 'Age_years'] = np.nan
df.loc[np.random.choice(n_samples, 10), 'Size_sqft'] = np.nan

# Calculate price based on features to make it somewhat realistic
base_price = 2000000
price = base_price + (df['Size_sqft'].fillna(1000) * 5000) + (df['BHK'] * 500000) - (df['Age_years'].fillna(0) * 50000)

city_multiplier = {'Mumbai': 2.0, 'Delhi': 1.5, 'Bangalore': 1.2, 'Chennai': 1.0, 'Pune': 0.9, 'Hyderabad': 1.1, 'Kolkata': 0.8, 'Ahmedabad': 0.85, 'Jaipur': 0.7, 'Surat': 0.75}
price = price * df['City'].map(city_multiplier)

furnishing_multiplier = {'Unfurnished': 1.0, 'Semi-furnished': 1.1, 'Fully-furnished': 1.25}
price = price * df['Furnishing'].map(furnishing_multiplier)

# Add some noise
price = price + np.random.normal(0, 500000, n_samples)

df['Price'] = price.abs() # Ensure positive

df.to_csv('house_data.csv', index=False)
print("Data generated successfully.")
