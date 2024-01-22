import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import scipy.sparse as sp

# Load dataset
dataset = pd.read_csv('test30_reduced.csv')

# Handling missing values
dataset.fillna(0, inplace=True)

# Extracting and encoding the target variable
target = dataset['target']
y = pd.get_dummies(target).values  # One-hot encoding the target variable

# Removing the target column from the dataset
X = dataset.drop('target', axis=1)

# Identify categorical columns (non-numeric)
categorical_columns = X.select_dtypes(include=['object']).columns

# Apply one-hot encoding to categorical columns
encoder = OneHotEncoder(sparse=True)
X_categorical = encoder.fit_transform(X[categorical_columns])

# Drop original categorical columns
X = X.drop(categorical_columns, axis=1)

# Combine numeric and encoded categorical columns
X_combined = sp.hstack((sp.csr_matrix(X), X_categorical), format='csr')

# Standardizing the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Building the RNN model
model = Sequential()
model.add(SimpleRNN(50, input_shape=(1, X_train.shape[2]), activation='relu'))
model.add(Dense(y_train.shape[1], activation='softmax'))

# Compiling the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training the model
model.fit(X_train, y_train, epochs=2, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Making predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Calculating the metrics with 'macro' average for multiclass classification
precision = precision_score(y_true, y_pred_classes, average='macro')
recall = recall_score(y_true, y_pred_classes, average='macro')
f1 = f1_score(y_true, y_pred_classes, average='macro')
auc = roc_auc_score(to_categorical(y_true), y_pred, multi_class='ovr')

# Summary of the model
model.summary()

# Generate predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

# Calculating metrics for each class in a multi-class setting
num_classes = y_pred.shape[1]
fpr = [0] * num_classes
tpr = [0] * num_classes
fnr = [0] * num_classes
tnr = [0] * num_classes
for i in range(num_classes):
    tn = conf_matrix[i, i]
    fp = conf_matrix[i, :].sum() - tn
    fn = conf_matrix[:, i].sum() - tn
    tp = conf_matrix.sum() - (fp + fn + tn)
    fpr[i] = fp / (fp + tn)  # False Positive Rate
    tpr[i] = tp / (tp + fn)  # True Positive Rate
    fnr[i] = fn / (fn + tp)  # False Negative Rate
    tnr[i] = tn / (tn + fp)  # True Negative Rate

# Print the rates
print("True Positive Rate (TPR):", tpr)
print("False Negative Rate (FNR):", fnr)
print("True Negative Rate (TNR):", tnr)
print("False Positive Rate (FPR):", fpr)

# Printing the metrics
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'AUC: {auc}')
print(f'Confusion matrix: {conf_matrix}')
print('False Positive Rate by class:', fpr)
print('True Positive Rate by class:', tpr)
print('False Negative Rate by class:', fnr)
print('True Negative Rate by class:', tnr)
