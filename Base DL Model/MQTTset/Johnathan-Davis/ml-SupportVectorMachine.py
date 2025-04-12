import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearnex import patch_sklearn
# Load the dataset
file_path = 'mqttdataset_reduced.csv'
df = pd.read_csv(file_path)

# Encode the categorical columns
label_encoders = {}
for column in df.columns:
    if df[column].dtype == type(object):
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

# Split the data into features and target
X = df.drop('target', axis=1)
y = df['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initialize the Support Vector Machine classifier
svm_model = LinearSVC(dual=False, max_iter=50)

# Train the model
svm_model.fit(X_train, y_train)

# Make predictions
y_pred = svm_model.predict(X_test)
y_pred_proba = svm_model._predict_proba_lr(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

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
    
# Add logic here to use class_metrics as needed
tpr = tp / (tp + fn)
fpr = fp / (fp + tn)
tnr = tn / (tn + fp)
fnr = fn / (fn + tp)

# Output metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("AUC:", auc)
print("TPR:", tpr)
print("FPR:", fpr)
print("TNR:", tnr)
print("FNR:", fnr)

patch_sklearn()