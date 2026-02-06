#	0.112829664
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

# --- 1. Load data ---
train = pd.read_csv("Train.csv")
test = pd.read_csv("Test.csv")

print("Train shape:", train.shape, "Test shape:", test.shape)

# --- 2. Keep test IDs and country ---
test_ids = test["uniqueid"].copy()
test_country = test["country"].copy()

# Drop IDs from features
train = train.drop(columns=["uniqueid"])
test_features = test.drop(columns=["uniqueid"])

# --- 3. Encode target ---
y = train["bank_account"].map({"Yes": 1, "No": 0})
X = train.drop(columns=["bank_account"])

# --- 4. Separate numeric and categorical columns ---
numeric_cols = ["year", "household_size", "age_of_respondent"]
categorical_cols = [
    "country", "location_type", "cellphone_access", "gender_of_respondent",
    "relationship_with_head", "marital_status", "education_level", "job_type"
]

# Convert numeric columns
for col in numeric_cols:
    X[col] = pd.to_numeric(X[col], errors="coerce")
    test_features[col] = pd.to_numeric(test_features[col], errors="coerce")

# Convert categorical columns to 'category' dtype
for col in categorical_cols:
    X[col] = X[col].astype("category")
    test_features[col] = test_features[col].astype("category")

# --- 5. Train-validation split for local validation ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)

# --- 6. Train LightGBM ---
model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    random_state=1
)

model.fit(X_train, y_train, categorical_feature=categorical_cols)

# --- 7. Validate ---
val_preds = model.predict(X_val)
print("Validation accuracy:", accuracy_score(y_val, val_preds))

# --- 8. Retrain on full train and predict test ---
model.fit(X, y, categorical_feature=categorical_cols)
test_preds = model.predict(test_features)

# --- 9. Build submission ---
submission_ids = test_ids + " x " + test_country
submission = pd.DataFrame({
    "unique_id": submission_ids,
    "bank_account": test_preds
})

# --- 10. Save CSV ---
submission.to_csv("submission.csv", index=False)
print("submission.csv saved")
print(submission.head())
