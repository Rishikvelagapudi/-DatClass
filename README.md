# Data Classification Using AI

This project demonstrates a complete Machine Learning workflow for classifying data, using the well-known Iris dataset. It covers data loading, Exploratory Data Analysis (EDA), data preprocessing, training multiple classifiers, evaluating them, and saving the best model for future predictions. 

## Features
- **Exploratory Data Analysis:** Includes statistical summaries, missing values checks, and class distribution.
- **Model Training:** Trains six different algorithms:
  - Decision Tree
  - K-Nearest Neighbors (KNN)
  - Logistic Regression
  - Random Forest
  - Naive Bayes
  - Support Vector Machine (SVM)
- **Model Evaluation:** Evaluates models based on Accuracy, Precision, Recall, and F1-Score, and generates Confusion Matrices.
- **Automatic Model Selection:** Compares the algorithms and automatically saves the best-performing model using `joblib`.
- **Web Interface:** Includes a Streamlit web application for interactive predictions and CSV batch processing.

## Folder Structure
```text
DataClassification/
├── dataset/
│   └── iris.csv                 # The dataset used for training
├── images/                      # Contains generated confusion matrices and accuracy comparison
├── saved_model/
│   ├── classifier.pkl           # The trained best model
│   └── scaler.pkl               # The fitted StandardScaler
├── model.py                     # Contains classifier definitions
├── train.py                     # Main script for EDA, training, and evaluation
├── predict.py                   # Script to predict on new data via CLI
├── notebook.ipynb               # Jupyter notebook with detailed walkthrough
├── app.py                       # Streamlit web application
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## Installation

1. Make sure you have Python installed (3.8+ recommended).
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Train the Models
To execute the data pipeline, train all models, generate visualizations, and save the best model:
```bash
python train.py
```
*Note: This will output EDA details in the console, save plots in `images/`, and save the best model in `saved_model/`.*

### 2. Predict using CLI
To predict the species of a new Iris flower, provide its features (Sepal Length, Sepal Width, Petal Length, Petal Width):
```bash
python predict.py 5.1 3.5 1.4 0.2
```

### 3. Run the Streamlit Web App
To start the interactive web application:
```bash
streamlit run app.py
```
This will open a local web server (usually at `http://localhost:8501`) where you can:
- Manually enter features to get a prediction.
- Upload a CSV for batch predictions.
- View the model comparison charts.
