import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the dataset
df = pd.read_csv('mqttdataset_reduced.csv')

# Separate features and the target variable
labels = df['target']
features = df.drop('target', axis=1)

# Encode the target variable
label_encoder = LabelEncoder()
df['target'] = label_encoder.fit_transform(df['target'])

# Identifying categorical columns for encoding
categorical_columns = features.select_dtypes(include=['object']).columns

# Encoding categorical columns
ct = ColumnTransformer(
    [("ohe", OneHotEncoder(), categorical_columns)], 
    remainder='passthrough'
)
features_encoded = ct.fit_transform(features)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_encoded, labels, test_size=0.2, random_state=42)

# Simplified parameter grid for RandomizedSearchCV
param_grid = {
    'n_neighbors': [3, 5, 7, 10],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# Using RandomizedSearchCV for hyperparameter tuning
random_search = RandomizedSearchCV(
    KNeighborsClassifier(), 
    param_grid, 
    n_iter=10, 
    n_jobs=-1, 
    cv=3, 
    random_state=42
)

# Using a subset of the training data for RandomizedSearchCV
subset_size = 5000
X_train_subset = X_train[:subset_size]
y_train_subset = y_train.iloc[:subset_size]

# Performing RandomizedSearchCV
random_search.fit(X_train_subset, y_train_subset)

# Best model from RandomizedSearchCV
best_model = random_search.best_estimator_

# Evaluating the model on the test set
y_pred = best_model.predict(X_test)

# Predict probability to use in AUC 
y_pred_prob = best_model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
roc_auc = roc_auc_score(y_test, y_pred_prob, average='macro', multi_class='ovr')
confusion = confusion_matrix(y_test, y_pred)

# Printing the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC:", roc_auc)
print("Confusion Matrix:\n", confusion)


TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

# Calculating the rates
TPR = TP / (TP + FN)
FPR = FP / (FP + TN)
TNR = TN / (TN + FP)
FNR = FN / (TP + FN)

# Printing the rates
print("True Positive Rate:", TPR)
print("False Positive Rate:", FPR)
print("True Negative Rate:", TNR)
print("False Negative Rate:", FNR)