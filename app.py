from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

def indian_format(number):
    num_str = str(int(number))
    if len(num_str) <= 3:
        return num_str
    last_three = num_str[-3:]
    other_digits = num_str[:-3]
    chunks = []
    while len(other_digits) > 2:
        chunks.insert(0, other_digits[-2:])
        other_digits = other_digits[:-2]
    if other_digits:
        chunks.insert(0, other_digits)
    return ",".join(chunks) + "," + last_three

# Load the trained model
try:
    model = joblib.load('house_price_model.pkl')
except FileNotFoundError:
    print("Error: Model file not found. Please run 'predict_house_price.py' first.")
    exit()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from the form
    try:
        data = {
            'BHK': [int(request.form.get('bhk', 0))],
            'City': [request.form.get('city')],
            'Size_sqft': [int(request.form.get('size_sqft', 0))],
            'Floors': [int(request.form.get('floors', 0))],
            'Bathrooms': [int(request.form.get('bathrooms', 0))],
            'Furnishing': [request.form.get('furnishing')],
            'Property_Type': [request.form.get('property_type')],
            'Age_years': [int(request.form.get('age_years', 0))],
            'Parking': [request.form.get('parking')]
        }
        
        # Convert into a pandas DataFrame
        df = pd.DataFrame(data)
        
        # Make prediction
        prediction = model.predict(df)[0]
        
        # Format the price nicely in Indian style
        formatted_price = f"₹ {indian_format(prediction)}"
        
        return jsonify({'success': True, 'price': formatted_price})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == "__main__":
    # Run the app on port 8080 to avoid macOS AirPlay conflicts
    app.run(host='0.0.0.0', port=8080, debug=True)
