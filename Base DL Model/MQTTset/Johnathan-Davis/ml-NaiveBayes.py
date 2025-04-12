import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix)
from sklearn.preprocessing import LabelEncoder

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

# Define the model and parameters for GridSearch
model = GaussianNB()
param_grid = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
}
cv = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')

# Train the model
cv.fit(X_train, y_train)

# Predictions
y_pred = cv.predict(X_test)
y_prob = cv.predict_proba(X_test)  # Modified to include probabilities for all classes

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
auc = roc_auc_score(y_test, y_prob, multi_class='ovr')  # Assuming a multi-class scenario

# Confusion matrix for TPR, FPR, TNR, FNR
# Multi-class confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

class_metrics = {}
for class_index in range(len(conf_matrix)):
    tn = conf_matrix[class_index, class_index]
    fp = conf_matrix[:, class_index].sum() - tn
    fn = conf_matrix[class_index, :].sum() - tn
    tp = conf_matrix.sum() - (fp + fn + tn)
    class_metrics[class_index] = {'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp}

tpr = tp / (tp + fn)
fpr = fp / (fp + tn)
tnr = tn / (tn + fp)
fnr = fn / (fn + tp)

print("Best Parameters:", cv.best_params_)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Area Under Curve:", auc)
print("True Positive Rate:", tpr)
print("False Positive Rate:", fpr)
print("True Negative Rate:", tnr)
print("False Negative Rate:", fnr)
