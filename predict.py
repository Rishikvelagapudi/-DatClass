import sys
import joblib
import numpy as np

def predict():
    try:
        # Load the trained model and scaler
        model = joblib.load('saved_model/classifier.pkl')
        scaler = joblib.load('saved_model/scaler.pkl')
    except FileNotFoundError:
        print("Error: Model or scaler not found. Please run train.py first.")
        return

    print("\n" + "="*40)
    print("🌺  Iris Species Predictor (Interactive) 🌺")
    print("="*40)
    print("Please enter the following dimensions:")

    try:
        sepal_length = float(input("Sepal Length (cm): "))
        sepal_width = float(input("Sepal Width (cm): "))
        petal_length = float(input("Petal Length (cm): "))
        petal_width = float(input("Petal Width (cm): "))
    except ValueError:
        print("Error: All features must be numeric values.")
        return
    except KeyboardInterrupt:
        print("\nExiting...")
        return

    features = [sepal_length, sepal_width, petal_length, petal_width]
    # Scale the input
    features_scaled = scaler.transform([features])

    # Predict
    prediction = model.predict(features_scaled)
    
    print("-" * 40)
    print(f"✅ Predicted Species: {prediction[0].upper()}")
    print("-" * 40 + "\n")

if __name__ == "__main__":
    predict()
