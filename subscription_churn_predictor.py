import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

train_df = pd.read_csv("customer_churn_dataset-training-master.csv")
test_df  = pd.read_csv("customer_churn_dataset-testing-master.csv")

train_df = train_df.dropna(subset=["Churn"])

X = train_df.drop("Churn", axis=1)
y = train_df["Churn"]

X = pd.get_dummies(X, drop_first=True)
X_test = pd.get_dummies(test_df.drop("Churn", axis=1), drop_first=True)

X_test = X_test.reindex(columns=X.columns, fill_value=0)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_val_poly   = poly.transform(X_val_scaled)
X_test_poly  = poly.transform(X_test_scaled)

logreg = LogisticRegression(max_iter=1000, solver="lbfgs")
logreg.fit(X_train_poly, y_train)

y_pred = logreg.predict(X_val_poly)
print(confusion_matrix(y_val, y_pred))
print(classification_report(y_val, y_pred))
print("ROC-AUC:", roc_auc_score(y_val, logreg.predict_proba(X_val_poly)[:, 1]))

test_preds = logreg.predict(X_test_poly)
test_probs = logreg.predict_proba(X_test_poly)[:, 1]
out = test_df.copy()
out["Churn_Prediction"]  = test_preds
out["Churn_Probability"] = test_probs
out.to_csv("churn_predictions.csv", index=False)
print("✅ Klar – resultat sparat i churn_predictions.csv")
