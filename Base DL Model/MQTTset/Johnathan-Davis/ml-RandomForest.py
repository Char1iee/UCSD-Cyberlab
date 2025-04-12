import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the dataset
df = pd.read_csv('mqttdataset_reduced.csv')

# Separate features and the target variable
X = df.drop('target', axis=1)
y = df['target']

# Apply label encoding to categorical columns
label_encoder = LabelEncoder()
for col in X.select_dtypes(include=['object', 'category']).columns:
    X[col] = label_encoder.fit_transform(X[col])

# Encode the target variable
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Define and train the Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
auc = roc_auc_score(y_test, rf_model.predict_proba(X_test), multi_class='ovr')

# Function to calculate TPR, FPR, TNR, FNR for each class
def calculate_class_specific_rates(conf_matrix, class_index):
    tp = conf_matrix[class_index, class_index]
    fn = sum(conf_matrix[class_index, :]) - tp
    fp = sum(conf_matrix[:, class_index]) - tp
    tn = conf_matrix.sum() - (tp + fp + fn)

    tpr = tp / (tp + fn) if (tp + fn) != 0 else 0  # True Positive Rate
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0  # False Positive Rate
    tnr = tn / (tn + fp) if (tn + fp) != 0 else 0  # True Negative Rate
    fnr = fn / (fn + tp) if (fn + tp) != 0 else 0  # False Negative Rate

    return tpr, fpr, tnr, fnr

# Calculate the confusion matrix for multi-class
conf_matrix = confusion_matrix(y_test, y_pred)

# Calculate TPR, FPR, TNR, FNR for each class and average them
class_rates = np.array([calculate_class_specific_rates(conf_matrix, i) for i in range(len(np.unique(y_test)))])
average_tpr = np.mean(class_rates[:, 0])
average_fpr = np.mean(class_rates[:, 1])
average_tnr = np.mean(class_rates[:, 2])
average_fnr = np.mean(class_rates[:, 3])

# Print the metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"AUC: {auc}")
print(f"Average TPR: {average_tpr}")
print(f"Average FPR: {average_fpr}")
print(f"Average TNR: {average_tnr}")
print(f"Average FNR: {average_fnr}")
