import os
from scipy.io import arff
import pandas as pd
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from itertools import combinations
from tqdm import tqdm

path = 'vpndata/'

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

all_features = packet_iat_features + active_idle_features + speed_features

for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith('.arff'):
            file_path = os.path.join(root, file)
            dataset_name = os.path.splitext(file)[0]
            print(f"\nProcessing file: {file_path}")

            # Load ARFF file
            data, meta = arff.loadarff(file_path)
            df = pd.DataFrame(data)

            # Decode byte strings
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

            # Ensure class label is numeric
            if 'class1' not in df.columns:
                print("No 'class1' column found. Skipping this dataset.")
                continue

            if df['class1'].dtype == 'object':
                df['class1'] = df['class1'].map({'Non-VPN': 0, 'VPN': 1})

            # Handle missing values by imputing column means
            df = df.fillna(df.mean())

            y = df['class1']

            # Check available features
            available_features = [f for f in all_features if f in df.columns]

            if len(available_features) < 2:
                print("Not enough features of interest available in this dataset. Skipping.")
                continue

            feature_pairs = list(combinations(available_features, 2))

            best_accuracy = 0.0
            best_pair = None
            best_pair_accuracies = []

            # Wrap the pair iteration with tqdm for progress
            for (f1, f2) in tqdm(feature_pairs, desc="Evaluating feature pairs"):
                X = df[[f1, f2]]

                # Ensure numeric
                for col in X.columns:
                    if X[col].dtype == 'object':
                        X[col] = X[col].astype('category').cat.codes

                kf = KFold(n_splits=10, shuffle=True, random_state=42)
                accuracies = []

                # Wrap the fold iteration with tqdm for additional visibility if desired
                for train_index, test_index in kf.split(X):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                    # Use GPU with new parameters
                    model = XGBClassifier(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=6,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42,
                        tree_method='hist',  # CPU hist method
                        eval_metric='logloss'
                    )

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    acc = accuracy_score(y_test, y_pred)
                    accuracies.append(acc)

                avg_acc = sum(accuracies) / len(accuracies)

                # Plot accuracies for this pair
                plt.figure(figsize=(8,5))
                plt.plot(range(1, len(accuracies)+1), accuracies, marker='o')
                plt.title(f'{dataset_name}: Accuracy per Fold for Features ({f1}, {f2})')
                plt.xlabel('Fold Number')
                plt.ylabel('Accuracy')
                plt.ylim([0,1])
                plt.grid(True)
                plt.show()

                # Update best pair if needed
                if avg_acc > best_accuracy:
                    best_accuracy = avg_acc
                    best_pair = (f1, f2)
                    best_pair_accuracies = accuracies

            if best_pair is not None:
                print(f"Best feature pair for {dataset_name}: {best_pair} with accuracy: {best_accuracy:.4f}")
                # Plot the best pair's accuracy too if desired
                plt.figure(figsize=(8,5))
                plt.plot(range(1, len(best_pair_accuracies)+1), best_pair_accuracies, marker='o', color='red')
                plt.title(f'{dataset_name}: Accuracy per Fold for Best Pair {best_pair}')
                plt.xlabel('Fold Number')
                plt.ylabel('Accuracy')
                plt.ylim([0,1])
                plt.grid(True)
                plt.show()
            else:
                print("No valid pairs found for this dataset.")
