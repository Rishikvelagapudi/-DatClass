import sys
import joblib
import numpy as np

def predict():
    if len(sys.argv) != 5:
        print("Usage: python predict.py <SepalLength> <SepalWidth> <PetalLength> <PetalWidth>")
        print("Example: python predict.py 5.1 3.5 1.4 0.2")
        return

    try:
        features = [float(x) for x in sys.argv[1:5]]
    except ValueError:
        print("Error: All features must be numeric values.")
        return

    try:
        # Load the trained model and scaler
        model = joblib.load('saved_model/classifier.pkl')
        scaler = joblib.load('saved_model/scaler.pkl')
    except FileNotFoundError:
        print("Error: Model or scaler not found. Please run train.py first.")
        return

    # Scale the input
    features_scaled = scaler.transform([features])

    # Predict
    prediction = model.predict(features_scaled)
    
    print(f"Input: {features}")
    print(f"Predicted Species: {prediction[0]}")

if __name__ == "__main__":
    predict()
