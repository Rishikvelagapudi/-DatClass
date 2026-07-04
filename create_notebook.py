import nbformat as nbf

nb = nbf.v4.new_notebook()

text_1 = """\
# Data Classification Using AI
This notebook demonstrates a complete machine learning workflow to classify Iris species based on sepal and petal dimensions.
"""

code_1 = """\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
"""

text_2 = """\
## Step 3 & 4: Load and Explore Dataset
"""

code_2 = """\
# Load dataset
iris = load_iris()
df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
df.drop('target', axis=1, inplace=True)

# Explore dataset
print("Shape:", df.shape)
print("\\nMissing Values:\\n", df.isnull().sum())
display(df.head())
display(df.describe())
"""

text_3 = """\
## Step 5: Data Preprocessing & Step 6: Split Dataset
"""

code_3 = """\
# Feature Selection
X = df.drop('species', axis=1)
y = df['species']

# Split Data (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
"""

text_4 = """\
## Step 7 & 8: Define & Train Models
"""

code_4 = """\
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=200),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(random_state=42, probability=True)
}
"""

text_5 = """\
## Step 9 & 10: Evaluate Models
"""

code_5 = """\
results = {}
best_model_name = None
best_accuracy = 0
best_model = None

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"--- {name} ---")
    print(f"Accuracy: {acc*100:.2f}%")
    if acc > best_accuracy:
        best_accuracy = acc
        best_model_name = name
        best_model = model
"""

text_6 = """\
## Step 11 & 12: Compare & Visualize
"""

code_6 = """\
plt.figure(figsize=(10,6))
plt.bar(results.keys(), [acc * 100 for acc in results.values()], color=['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD'])
plt.title('Accuracy Comparison of Classification Algorithms')
plt.ylabel('Accuracy (%)')
plt.show()
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_1),
    nbf.v4.new_code_cell(code_1),
    nbf.v4.new_markdown_cell(text_2),
    nbf.v4.new_code_cell(code_2),
    nbf.v4.new_markdown_cell(text_3),
    nbf.v4.new_code_cell(code_3),
    nbf.v4.new_markdown_cell(text_4),
    nbf.v4.new_code_cell(code_4),
    nbf.v4.new_markdown_cell(text_5),
    nbf.v4.new_code_cell(code_5),
    nbf.v4.new_markdown_cell(text_6),
    nbf.v4.new_code_cell(code_6)
]

with open('notebook.ipynb', 'w') as f:
    nbf.write(nb, f)
