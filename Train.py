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


# we have to load different datasets different
sinarioA1path = 'vpndata/Scenario A1-ARFF'
sinarioA2path = 'vpndata/Scenario A2-ARFF'
sinarioBpath  = 'vpndata/Scenario B-ARFF'

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

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

all_features = packet_iat_features + active_idle_features + speed_features + additional_features


totalframe = []

# ok so we added loading based on sinario, forgot to record, this func tells us the legitamancy of the data
totalframe.append(load_scenario_data(sinarioA1path, scenario_label='A1'))
# A1 is the only one that has a class1 column, the rest we have to determine if it is a vpn or not
totalframe.append(load_scenario_data(sinarioA2path, scenario_label='A2'))
# A2 is based on the file name, if it has NO-VPN in the name then it is not a vpn
totalframe.append(load_scenario_data(sinarioBpath,  scenario_label='B'))
#B is based on the class1 as well, if its somethign like Stream-VPN then it is a vpn, if its -NOVPN then its not

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

# Now we have a consistent 'class1' column: 0 => Non-VPN, 1 => VPN
y = df['class1']

# Filter the DataFrame to only keep features we care about (+ 'class1')
available_features = [f for f in all_features if f in df.columns]
print(f"Available features: {available_features}")
df = df[available_features + ['class1']]
print(df)
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
print(kf)

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
        'eval_metric': 'mae'
    }

    # Suggest a threshold between 0.0 and 1.0
    threshold = trial.suggest_float('threshold', 0.0, 1.0)

    model = XGBClassifier(**params)

    fold_accuracies = []
    for train_index, test_index in kf.split(df[available_features], y):
        X_train = df.iloc[train_index][available_features]
        y_train = y.iloc[train_index]
        X_test = df.iloc[test_index][available_features]
        y_test = y.iloc[test_index]

        model.fit(X_train, y_train)

        # Predict probabilities for class=1
        y_proba = model.predict_proba(X_test)[:, 1]

        # Apply the suggested threshold
        y_pred = (y_proba >= threshold).astype(int)

        # Compute accuracy (or your preferred metric)
        acc = accuracy_score(y_test, y_pred)
        fold_accuracies.append(acc)

    return np.mean(fold_accuracies)


pruner = MedianPruner(n_warmup_steps=5)
#pruner that prunes trials that are not in the top 50% of the trials, so stop when results are getting worse
study = create_study(direction='maximize', pruner=pruner)
study.optimize(objective, n_trials=50)

print(f"Best trial accuracy: {study.best_trial.value}")
print(f"Best parameters: {study.best_trial.params}")



best_params = study.best_trial.params
final_model = XGBClassifier(
    **best_params,
    tree_method='hist',
    eval_metric='mae',
    device="cuda"  # cuda makes it faster
)

final_model.fit(df[available_features], y)

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y, final_model.predict_proba(df[available_features])[:, 1])
roc_auc = auc(fpr, tpr)
#AUC is the area under the curve, the higher the better
#FPR is the false positive rate
#TPR is the true positive rate

# Plot ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()



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

# now we have to train the threshold since the output is ultimately a list of probabilities
# we can use the roc curve to determine the best threshold
# https://www.iguazio.com/glossary/classification-threshold/
# www.kaggle.com/code/para24/xgboost-stepwise-tuning-using-optuna#6.-What-is-Optuna?
# https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_float
