from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load models and transformers
model = joblib.load("xgb_classifier.pkl")
scaler = joblib.load("scaler.pkl")

# Map your model's numerical cluster outputs to human-readable names
# Ensure these align with your K-Means cluster profiles!
segment_map = {
    0: "Budget Customer",
    1: "Regular Customer",
    2: "Premium Customer"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Extract inputs from the form submission
    age = float(request.form['age'])
    total_purchases = float(request.form['total_purchases'])
    income = float(request.form['income'])
    spending = float(request.form['spending'])
    children = float(request.form['children'])
    recency = float(request.form['recency'])
    web = float(request.form['web'])
    catalog = float(request.form['catalog'])
    store = float(request.form['store'])

    # 2. Feature Engineering: Match your training preprocessing (Log transforms)
    income_log = np.log1p(income)
    spend_log = np.log1p(spending)

    # 3. Structure features array exactly as the scaler expects
    features = np.array([[
        age, total_purchases, income_log, spend_log, 
        children, recency, web, catalog, store
    ]])

    # 4. Scale features and make predictions
    features_scaled = scaler.transform(features)
    cluster = int(model.predict(features_scaled)[0])
    probs = model.predict_proba(features_scaled)[0]

    # 5. Extract classification probabilities to fuel the Chart.js Pie Chart
    pie_data = {
        "Budget Customer": round(probs[0] * 100, 2),
        "Regular Customer": round(probs[1] * 100, 2),
        "Premium Customer": round(probs[2] * 100, 2)
    }

    # 6. Map the numerical cluster index to its string name
    prediction = segment_map.get(cluster, f"Cluster {cluster}")

    # 7. Render index.html with the model results
    return render_template(
        'index.html',
        prediction=prediction,
        cluster=cluster,
        pie_data=pie_data
    )

if __name__ == "__main__":
    app.run(debug=True)
