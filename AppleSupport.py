import xgboost as xgb
import coremltools as ct

# Load the trained XGBoost model
model = xgb.Booster()
model.load_model('final_model.model')

# Define feature names and class labels
feature_names = [
    'min_fiat', 'max_fiat', 'mean_fiat', 'min_biat', 'max_biat', 'mean_biat',
    'min_flowiat', 'max_flowiat', 'mean_flowiat', 'std_flowiat', 'min_active',
    'mean_active', 'max_active', 'std_active', 'min_idle', 'mean_idle',
    'max_idle', 'std_idle', 'flowPktsPerSecond', 'flowBytesPerSecond', 'duration'
]
class_labels = ['class1']  # Replace with your actual class labels

# Convert the model with specified feature names and class labels
coreml_model = ct.converters.xgboost.convert(model, feature_names=feature_names, class_labels=class_labels, mode='classifier')

# Save the Core ML model to a file
coreml_model.save('my_model.mlmodel')