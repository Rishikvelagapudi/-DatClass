import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from model import get_models

def main():
    print("Step 3: Loading Dataset...")
    # Load from sklearn and save to csv to maintain folder structure requirement
    iris = load_iris()
    df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                      columns=iris['feature_names'] + ['target'])
    
    # Map target numbers to species names for better EDA
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    df.drop('target', axis=1, inplace=True)

    # Save to dataset/iris.csv
    os.makedirs('dataset', exist_ok=True)
    df.to_csv('dataset/iris.csv', index=False)
    
    # Load dataset into pandas DataFrame
    df = pd.read_csv('dataset/iris.csv')
    
    print("\nStep 4: Explore Dataset")
    print("--- Head ---")
    print(df.head())
    print("\n--- Shape ---")
    print(df.shape)
    print("\n--- Info ---")
    df.info()
    print("\n--- Statistical Summary ---")
    print(df.describe())
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    print("\n--- Class Distribution ---")
    print(df['species'].value_counts())
    
    print("\nStep 5: Data Preprocessing")
    # Missing Values
    if df.isnull().sum().any():
        print("Handling missing values...")
        df.fillna(df.mean(numeric_only=True), inplace=True)
    
    # Duplicate Values
    if df.duplicated().sum() > 0:
        print(f"Removing {df.duplicated().sum()} duplicates...")
        df.drop_duplicates(inplace=True)
        
    # Feature Selection
    X = df.drop('species', axis=1)
    y = df['species']
    
    print("\nStep 6: Split Dataset")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training Data: {X_train.shape[0]} samples")
    print(f"Testing Data: {X_test.shape[0]} samples")
    
    # Optional Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for prediction later
    os.makedirs('saved_model', exist_ok=True)
    joblib.dump(scaler, 'saved_model/scaler.pkl')

    print("\nSteps 7 & 8: Train Models")
    models = get_models()
    
    print("\nSteps 9 & 10: Evaluate Models")
    results = {}
    best_model_name = None
    best_accuracy = 0
    best_model = None
    
    os.makedirs('images', exist_ok=True)
    
    for name, model in models.items():
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Evaluate
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"\n--- {name} ---")
        print(f"Accuracy: {acc*100:.2f}%")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f'images/cm_{name.replace(" ", "_")}.png')
        plt.close()
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            best_model = model

    print("\nStep 11 & 12: Compare Algorithms & Visualization")
    print("\nModel Accuracies:")
    for name, acc in results.items():
        print(f"{name}: {acc*100:.2f}%")
        
    print(f"\nBest Model: {best_model_name} with Accuracy {best_accuracy*100:.2f}%")
    
    # Plot Accuracy Comparison Bar Chart
    plt.figure(figsize=(10,6))
    bars = plt.bar(results.keys(), [acc * 100 for acc in results.values()], color=['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD'])
    plt.title('Accuracy Comparison of Classification Algorithms')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Algorithms')
    plt.ylim(0, 110)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('images/accuracy_comparison.png')
    # Can also save at root level as requested in folder structure
    plt.savefig('accuracy_comparison.png')
    plt.close()
    
    print("\nStep 13: Save Best Model")
    joblib.dump(best_model, 'saved_model/classifier.pkl')
    print("Model saved to 'saved_model/classifier.pkl'")

if __name__ == "__main__":
    main()
