import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



df = pd.read_csv("winequality-red.csv")

print("Dataset successfully loaded!")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

print("\nALL COLUMNS:")
print(df.columns.tolist())

print("\nNUMBER OF COLUMNS:", len(df.columns))
print("DATASET SHAPE:", df.shape)


features = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol"
]

target = "quality"



required_columns = features + [target]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    print("\nERROR!")
    print("Missing columns:")
    print(missing_columns)
    print("\nYour CSV file is incorrect.")
    exit()




X = df[features]
y = df[target]




X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)




y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel trained successfully!")
print("Accuracy:", accuracy)



model_data = {
    "model": model,
    "columns": features
}

joblib.dump(
    model_data,
    "wine_quality_model.joblib"
)

print("\nModel saved successfully!")
print("File: wine_quality_model.joblib")