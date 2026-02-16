import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

TRAIN_PATH = "Train.csv"
TEST_PATH = "Test.csv"
ID_COL = "ID"
TARGET = "Yield"
N_SPLITS = 5
RANDOM_STATE = 42

# Load
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
test_ids = test[ID_COL]

# Clip target to 99th percentile & log1p
train[TARGET] = train[TARGET].clip(upper=train[TARGET].quantile(0.99))
y = np.log1p(train[TARGET])
train = train.drop([TARGET, ID_COL], axis=1)
test = test.drop([ID_COL], axis=1)

# Date features
date_cols = ['CropTillageDate','RcNursEstDate','SeedingSowingTransplanting','Harv_date','Threshing_date']
for df in [train, test]:
    for col in date_cols: df[col] = pd.to_datetime(df[col], errors='coerce')
    df['Duration_Growth'] = (df['Harv_date'] - df['SeedingSowingTransplanting']).dt.days
    df['Duration_Prep'] = (df['SeedingSowingTransplanting'] - df['CropTillageDate']).dt.days
    df['Duration_PostHarvest'] = (df['Threshing_date'] - df['Harv_date']).dt.days
    for col in date_cols:
        df[col+'_month'] = df[col].dt.month
        df[col+'_day'] = df[col].dt.dayofyear
        df.drop(col, axis=1, inplace=True)

# Numeric fill
for col in train.select_dtypes(include=[np.number]).columns:
    median_val = train[col].median()
    train[col] = train[col].fillna(median_val)
    if col in test.columns: test[col] = test[col].fillna(median_val)

# Categorical label encoding
for col in train.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    full = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(full)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

lgb_params = {
    'objective': 'regression', 'metric':'rmse', 'learning_rate':0.005,
    'num_leaves':80, 'feature_fraction':0.75, 'bagging_fraction':0.75,
    'bagging_freq':5, 'min_data_in_leaf':30, 'n_jobs':-1, 'random_state':RANDOM_STATE,
    'verbose':-1
}

# KFold training
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

for train_idx, val_idx in kf.split(train, y):
    X_tr, X_val = train.iloc[train_idx], train.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    model = lgb.train(lgb_params, dtrain, num_boost_round=5000,
                      valid_sets=[dval],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)])
    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(test) / N_SPLITS

# CV RMSE
rmse = np.sqrt(mean_squared_error(np.expm1(y), np.expm1(oof_preds)))
print(f"Local CV RMSE: {rmse}")

# Final predictions
final_preds = np.expm1(test_preds)
final_preds = np.where(final_preds<0,0,final_preds)

submission = pd.DataFrame({ID_COL: test_ids, TARGET: final_preds})
submission.to_csv("submission.csv", index=False)
print("Saved submission_final.csv")
print(submission.head())