
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('test30.csv')

# One-hot encoding for categorical columns
categorical_columns = [col for col in df.columns if df[col].dtype == 'object' or len(df[col].unique()) < 10]
categorical_columns = [col for col in categorical_columns if col != 'target' and col != 'mqtt.msg'] # Excluding 'target' and 'mqtt.msg'
df_encoded = pd.get_dummies(df, columns=categorical_columns)

# Excluding the 'mqtt.msg' column from the feature set
df_encoded = df_encoded.drop('mqtt.msg', axis=1)

# Label encoding the target variable
label_encoder = LabelEncoder()
df_encoded['target'] = label_encoder.fit_transform(df_encoded['target'])

# Splitting the dataset into features and target variable
X = df_encoded.drop('target', axis=1)
y = df_encoded['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter optimization
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

dt = DecisionTreeClassifier()
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best model
best_dt = grid_search.best_estimator_

# Predictions and predicted probabilities
y_pred = best_dt.predict(X_test)
y_pred_proba = best_dt.predict_proba(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
auc = roc_auc_score(y_test, y_pred_proba, average='weighted', multi_class='ovo')

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

# Print metrics
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'AUC: {auc}')
print(f'TPR: {tpr}')
print(f'FPR: {fpr}')
print(f'TNR: {tnr}')
print(f'FNR: {fnr}')
