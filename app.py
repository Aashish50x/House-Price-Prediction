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

import os

# Get absolute path for Vercel
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'house_price_model.pkl')
csv_path = os.path.join(base_dir, 'house_data.csv')

# Load the trained model
try:
    model = joblib.load(model_path)
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

@app.route('/dataset')
def dataset():
    # Load the CSV safely for Vercel
    df = pd.read_csv(csv_path).head(100)
    # Simple, clean HTML table rendering with some inline styling
    html_table = df.to_html(classes='data-table', index=False)
    
    html_page = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Cleaned Dataset Sample</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600&display=swap" rel="stylesheet">
        <style>
            body {{ font-family: 'Outfit', sans-serif; background: #0f172a; color: #f8fafc; padding: 2rem; }}
            h1 {{ color: #c9985b; text-align: center; }}
            .table-container {{ overflow-x: auto; max-width: 1200px; margin: 0 auto; background: rgba(15, 23, 42, 0.85); border-radius: 12px; padding: 1rem; border: 1px solid rgba(201, 152, 91, 0.3); }}
            table.data-table {{ width: 100%; border-collapse: collapse; text-align: left; }}
            table.data-table th {{ background-color: rgba(201, 152, 91, 0.2); color: #c9985b; padding: 12px; }}
            table.data-table td {{ border-bottom: 1px solid rgba(255, 255, 255, 0.1); padding: 10px; }}
            a.back-btn {{ display: inline-block; margin-bottom: 20px; color: #c9985b; text-decoration: none; border: 1px solid #c9985b; padding: 8px 16px; border-radius: 6px; }}
            a.back-btn:hover {{ background: #c9985b; color: #0f172a; }}
        </style>
    </head>
    <body>
        <a href="/" class="back-btn">&larr; Back to App</a>
        <h1>Cleaned Dataset (Sample 100 Rows)</h1>
        <div class="table-container">
            {html_table}
        </div>
    </body>
    </html>
    """
    return html_page

if __name__ == "__main__":
    # Run the app on port 8080 to avoid macOS AirPlay conflicts
    app.run(host='0.0.0.0', port=8080, debug=True)
