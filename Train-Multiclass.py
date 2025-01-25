import os
import numpy as np
from scipy.io import arff
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from itertools import combinations
from tqdm import tqdm
from optuna import create_study
from optuna.pruners import MedianPruner

from Utils import load_scenario_data

# Paths to datasets
sinarioA2path = 'vpndata/Scenario A2-ARFF'
sinarioBpath  = 'vpndata/Scenario B-ARFF'

# Feature lists
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
    'duration'
]

all_features = packet_iat_features + active_idle_features + speed_features + additional_features

# Load and preprocess data
totalframe = []
totalframe.append(load_scenario_data(sinarioA2path, scenario_label='A2'))
totalframe.append(load_scenario_data(sinarioBpath, scenario_label='B'))

# Combine all frames
df = pd.concat(totalframe, ignore_index=True)
print(f"Total combined rows after all scenarios: {len(df)}")

# Drop columns known to cause issues (if they exist)
to_drop = ['total_fiat', 'total_biat', 'std_fiat', 'std_biat']
for col in to_drop:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

# Check for missing values
print("Missing values count:\n", df.isnull().sum())

# Convert relevant columns to numeric if necessary
for col in df.columns:
    if col != 'class1' and df[col].dtype == 'object':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Encode class labels
df['class1'] = df['class1'].astype('category').cat.codes

# Define features and target
available_features = [f for f in all_features if f in df.columns]
X = df[available_features]
y = df['class1']

# Stratified K-Fold cross-validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

def objective(trial):
    # Parameter search space for XGBoost
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'random_state': trial.suggest_int('random_state', 0, 100),
        'gamma': trial.suggest_float('gamma', 0, 10),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1e-3, 10.0),
        'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.5, 1.0),
        'tree_method': 'hist',  # or "gpu_hist" if XGBoost <2.0.0 + GPU usage
        'eval_metric': 'mlogloss'
    }

    model = XGBClassifier(**params)

    fold_accuracies = []
    for train_index, test_index in kf.split(X, y):
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]

        model.fit(X_train, y_train)

        # Predict class labels
        y_pred = model.predict(X_test)

        # Compute accuracy
        acc = accuracy_score(y_test, y_pred)
        fold_accuracies.append(acc)

    return np.mean(fold_accuracies)

# Optuna study
pruner = MedianPruner(n_warmup_steps=5)
study = create_study(direction='maximize', pruner=pruner)
study.optimize(objective, n_trials=140)

print(f"Best trial accuracy: {study.best_trial.value}")
print(f"Best parameters: {study.best_trial.params}")

# Train final model with best parameters
best_params = study.best_trial.params
final_model = XGBClassifier(
    **best_params,
    tree_method='hist',
    eval_metric='mlogloss',
    device="cuda"  # cuda makes it faster
)

final_model.fit(X, y)

# Save the trained model to a file
model_filename = 'final_model.model'
final_model.save_model(model_filename)
print(f"Model saved to {model_filename}")

# Feature Importances (if available)
if hasattr(final_model, 'feature_importances_'):
    importances = final_model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': available_features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance Plot')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()

    print(importance_df)
else:
    print("Final model does not expose 'feature_importances_'.")