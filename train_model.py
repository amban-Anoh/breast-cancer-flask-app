# Step 1: Install and import packages
# pip install ucimlrepo scikit-learn pandas

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
import pandas as pd
import pickle

# Step 2: Load dataset
breast_cancer = fetch_ucirepo(id=15)
X = breast_cancer.data.features
y = breast_cancer.data.targets

# Step 3: Impute missing values
X = pd.DataFrame(SimpleImputer(strategy='median').fit_transform(X), columns=X.columns)

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 5: Train model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Step 6: Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 7: Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
