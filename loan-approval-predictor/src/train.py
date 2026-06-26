import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv("data/loan_approval_dataset.csv")

df.columns = df.columns.str.strip()

df.drop("loan_id", axis=1, inplace=True)

df["education"] = df["education"].map({
    " Graduate": 1,
    " Not Graduate": 0,
    "Graduate": 1,
    "Not Graduate": 0
})

df["self_employed"] = df["self_employed"].map({
    " Yes": 1,
    " No": 0,
    "Yes": 1,
    "No": 0
})

df["loan_status"] = df["loan_status"].map({
    " Approved": 1,
    " Rejected": 0,
    "Approved": 1,
    "Rejected": 0
})

X = df.drop("loan_status", axis=1)
y = df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy*100:.2f}%")

joblib.dump(model, "models/loan_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Model Saved Successfully!")