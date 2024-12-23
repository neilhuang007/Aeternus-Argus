import os
import numpy as np
from scipy.io import arff
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_curve
import matplotlib.pyplot as plt
from itertools import combinations
from tqdm import tqdm

path = 'vpndata/Scenario A1-ARFF'

# Features of interest
packet_iat_features = [
    'min_fiat', 'max_fiat', 'mean_fiat',
    'min_biat', 'max_biat', 'mean_biat',
    'min_flowiat', 'max_flowiat', 'mean_flowiat', 'std_flowiat'
]

active_idle_features = [
    'min_active', 'mean_active', 'max_active', 'std_active',
    'min_idle', 'mean_idle', 'max_idle', 'std_idle'
]

speed_features = [
    'flowPktsPerSecond', 'flowBytesPerSecond'
]

additional_features = [
    'duration', 'class1'
]

all_features = packet_iat_features + active_idle_features + speed_features + additional_features


frames = []

def preprocess_arff(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
    with open(filepath, 'w') as file:
        for line in lines:
            if line.strip().startswith('@attribute'):
                file.write(line.replace('\n', ' ').replace('\r', ''))
            else:
                file.write(line)

for root, dirs, files in tqdm(os.walk(path)):
    for file in files:
        if file.endswith('.arff'):
            file_path = os.path.join(root, file)
            dataset_name = os.path.splitext(file)[0]
            print(f"\nProcessing file: {file_path}")

            # Preprocess ARFF file to handle multi-line attributes
            preprocess_arff(file_path)

            # Load ARFF file
            data, meta = arff.loadarff(file_path)
            frames.append(pd.DataFrame(data))

df = pd.concat(frames) # now it is one big frame
print(len(df))
# Decode byte strings
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
#print out the onese with n/a values

df.drop(columns=['total_fiat', 'total_biat', 'std_fiat','std_biat'], inplace=True) # drop columns with n/a values
print(df.isnull().sum()) # check for null fields

if df['class1'].dtype == 'object':
    df['class1'] = df['class1'].map({'Non-VPN': 0, 'VPN': 1})  # classify vpns to 1 and non-vpns to 0

y = df['class1']

# Check available features
available_features = [f for f in all_features if f in df.columns]

feature_pairs = list(combinations(available_features, 2))

best_accuracy = 0.0
best_pair = None
best_pair_accuracies = []

# we use optuma to fine tune our model


# Wrap the pair iteration with tqdm for progress
for (f1, f2) in tqdm(feature_pairs, desc="Evaluating feature pairs"):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    accuracies = []
    roccurves = []

    for train_index, test_index in kf.split(df,y=y):
        X = df[[f1, f2]]
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method='hist',
            eval_metric='mae'
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        roccurve = roc_curve(y_test, y_pred_prob)
        roccurves.append(roccurve)
        # print(roccurve)

        accuracies.append(acc)

    avg_acc = sum(accuracies) / len(accuracies)

    plt.figure(figsize=(8, 5))
    for fpr, tpr, _ in roccurves:
        plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--', color='red')
    plt.title(f'{dataset_name}: ROC Curve for Features ({f1}, {f2})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.show()

    if avg_acc > best_accuracy:
        best_accuracy = avg_acc
        best_pair = (f1, f2)
        best_pair_accuracies = accuracies

if best_pair is not None:
    print(f"Best feature pair for {dataset_name}: {best_pair} with accuracy: {best_accuracy:.4f}")
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(best_pair_accuracies) + 1), best_pair_accuracies, marker='o', color='red')
    plt.title(f'{dataset_name}: Accuracy per Fold for Best Pair {best_pair}')
    plt.xlabel('Fold Number')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.grid(True)
    plt.show()
else:
    print("No valid pairs found for this dataset.")