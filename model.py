from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def get_models():
    """
    Returns a dictionary of the six classifiers to be trained and evaluated.
    """
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=200),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(random_state=42, probability=True)
    }
    return models
