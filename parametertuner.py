import os

import numpy as np
from scipy.io import arff
import pandas as pd
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_curve
import matplotlib.pyplot as plt
from itertools import combinations
from tqdm import tqdm

file_path = os.path.join("vpndata/ScenarioA1-ARFF", "TimeBasedFeatures-Dataset-15s-VPN.arff")
dataset_name = os.path.splitext("TimeBasedFeatures-Dataset-15s-VPN.arff")[0]
print(f"\nProcessing file: {file_path}")

# Load ARFF file
data, meta = arff.loadarff(file_path)
df = pd.DataFrame(data)

# Decode byte strings
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

df = df.loc[:, (df != -1.0).any(axis=0)]  # drop columns if -1 is present

# Ensure class label is numeric
if 'class1' not in df.columns:
    print("No 'class1' column found. Skipping this dataset.")

if df['class1'].dtype == 'object':
    df['class1'] = df['class1'].map({'Non-VPN': 0, 'VPN': 1})  # classify vpns to 1 and non-vpns to 0

print(df)
y = df['class1']

# if we increase and the accuracy rate increases then we keep the value
def findbest(upperbound, lowerbound, parameter):
    # parameter gives the index for the parameter we are tuning
    # binary search for the best value once we know the range
    while lowerbound < upperbound:
        mid = (lowerbound + upperbound) // 2
        match(parameter):
            case 1:
                mid = (lowerbound + upperbound) // 2
                xgb1 = XGBClassifier(
                    learning_rate=mid,
                    n_estimators=1000,
                    max_depth=5,
                    min_child_weight=1,
                    gamma=0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='binary:logistic',
                    nthread=4,
                    scale_pos_weight=1,
                    seed=27)
            case 2:
                mid = (lowerbound + upperbound) // 2
                xgb1 = XGBClassifier(
                    learning_rate=0.1,
                    n_estimators=mid,
                    max_depth=5,
                    min_child_weight=1,
                    gamma=0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='binary:logistic',
                    nthread=4,
                    scale_pos_weight=1,
                    seed=27)
            case 3:
                mid = (lowerbound + upperbound) // 2
                xgb1 = XGBClassifier(
                    learning_rate=0.1,
                    n_estimators=1000,
                    max_depth=mid,
                    min_child_weight=1,
                    gamma=0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='binary:logistic',
                    nthread=4,
                    scale_pos_weight=1,
                    seed=27)
            case 4:
                mid = (lowerbound + upperbound) // 2
                xgb1 = XGBClassifier(
                    learning_rate=0.1,
                    n_estimators=1000,
                    max_depth=5,
                    min_child_weight=mid,
                    gamma=0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='binary:logistic',
                    nthread=4,
                    scale_pos_weight=1,
                    seed=27)
            case 5:
                mid = (lowerbound + upperbound) // 2
                xgb1 = XGBClassifier(
                    learning_rate=0.1,
                    n_estimators=1000,
                    max_depth=5,
                    min_child_weight=1,
                    gamma=mid,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='binary:logistic',
                    nthread=4,
                    scale_pos_weight=1,
                    seed=27)
            case 6:
                mid = (lowerbound + upperbound) // 2
                xgb1 = XGBClassifier(
                    learning_rate=0.1,
                    n_estimators=1000,
                    max_depth=5,
                    min_child_weight=1,
                    gamma=0,
                    subsample=mid,
                    colsample_bytree=0.8,
                    objective='binary:logistic',
                    nthread=4,
                    scale_pos_weight=1,
                    seed=27)
            case 7:
                mid = (lowerbound + upperbound) // 2
                xgb1 = XGBClassifier(
                    learning_rate=0.1,
                    n_estimators=1000,
                    max_depth=5,
                    min_child_weight=1,
                    gamma=0,
                    subsample=0.8,
                    colsample_bytree=mid,
                    objective='binary:logistic',
                    nthread=4,
                    scale_pos_weight=1,
                    seed=27)
            case 8:
                mid = (lowerbound + upperbound) // 2
                xgb1 = XGBClassifier(
                    learning_rate=0.1,
                    n_estimators=1000,
                    max_depth=5,
                    min_child_weight=1,
                    gamma=0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='binary:logistic',
                    nthread=mid,
                    scale_pos_weight=1,
                    seed=27)
            case 9:
                mid = (lowerbound + upperbound) // 2
                xgb1 = XGBClassifier(
                    learning_rate=0.1,
                    n_estimators=1000,
                    max_depth=5,
                    min_child_weight=1,
                    gamma=0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='binary:logistic',
                    nthread=4,
                    scale_pos_weight=mid,
                    seed=27)
            case 10:
                mid = (lowerbound + upperbound) // 2
                xgb1 = XGBClassifier(
                    learning_rate=0.1,
                    n_estimators=1000,
                    max_depth=5,
                    min_child_weight=1,
                    gamma=0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='binary:logistic',
                    nthread=4,
                    scale_pos_weight=1,
                    seed=mid)
        kfold = KFold(n_splits=10, shuffle=True, random_state=7)
        results = []
        for train, test in kfold.split(df):
            x_train, x_test = df.iloc[train], df.iloc[test]
            y_train, y_test = y.iloc[train], y.iloc[test]
            xgb1.fit(x_train, y_train) # fit with the current parameters and train the model
            predictions = xgb1.predict(x_test)

        # if the current is better we go in same direction
        if(results[len(results)] > results[len(results) - 1]):

        print(f"n_estimators: {mid}, Accuracy: {np.mean(results)}")
        if np.mean(results) > 0.95:
            upperbound = mid
        else:
            lowerbound = mid + 1

#xgboost model we are going to fine tune
for i in range(11):
    best = False



