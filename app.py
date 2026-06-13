from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("xgb_classifier.pkl")
scaler = joblib.load("scaler.pkl")

segment_map = {
    0: "Cluster 0",
    1: "Cluster 1",
    2: "Cluster 2"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = float(request.form['age'])
    total_purchases = float(request.form['total_purchases'])
    income = float(request.form['income'])
    spending = float(request.form['spending'])
    children = float(request.form['children'])
    recency = float(request.form['recency'])
    web = float(request.form['web'])
    catalog = float(request.form['catalog'])
    store = float(request.form['store'])

    income_log = np.log1p(income)
    spend_log = np.log1p(spending)

    features = np.array([[
        age, total_purchases, income_log, spend_log, 
        children, recency, web, catalog, store
    ]])

    features_scaled = scaler.transform(features)
    cluster = int(model.predict(features_scaled)[0])
    probs = model.predict_proba(features_scaled)[0]

    # This creates percentages out of the classification probabilities
    pie_data = {
        "Budget Customer": round(probs[0] * 100, 2),
        "Regular Customer": round(probs[1] * 100, 2),
        "Premium Customer": round(probs[2] * 100, 2)
    }

    prediction = segment_map.get(cluster, f"Cluster {cluster}")

    return render_template(
        'index.html',
        prediction=prediction,
        cluster=cluster,
        pie_data=pie_data
    )

if __name__ == "__main__":
    app.run(debug=True)