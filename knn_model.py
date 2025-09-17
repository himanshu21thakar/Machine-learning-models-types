# Install required libraries before running (if not already installed)
# pip install kagglehub pandas scikit-learn

import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ---- Step 1: Download the Kaggle dataset ----
path = kagglehub.dataset_download("uciml/iris")
print("Path to dataset files:", path)

# The dataset includes iris.data or iris.csv depending on version.
# Use pandas to load it:
csv_file = f"{path}/Iris.csv"  # KaggleHub usually stores it as Iris.csv
df = pd.read_csv(csv_file)

# ---- Step 2: Prepare features (X) and target (y) ----
# Inspect columns: typically ['Id','SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']
df = df.drop(columns=["Id"])  # Drop the Id column if present
X = df.drop(columns=["Species"])
y = df["Species"]

# ---- Step 3: Split the data ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---- Step 4: Hyperparameter tuning for KNN ----
param_grid = {'n_neighbors': range(3, 15), 'weights': ['uniform', 'distance']}
knn = KNeighborsClassifier()
grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

# ---- Step 5: Best model and evaluation ----
best_knn = grid.best_estimator_
print(f"Best Parameters: {grid.best_params_}")

y_pred = best_knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc*100:.2f}%")

if acc >= 0.80:
    print("✅ Achieved accuracy above 80%.")
else:
    print("❌ Accuracy below 80%, consider tuning further.")
