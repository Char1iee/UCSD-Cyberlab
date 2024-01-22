
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
import scipy.sparse as sp

# Load the dataset
dataset = pd.read_csv('test30_reduced.csv')

# Handling missing values - filling with 0
dataset.fillna(0, inplace=True)

# Extracting and encoding the target variable
target = dataset['target']
y = pd.get_dummies(target).values  # One-hot encode the target variable

# Remove the target column from the dataset
dataset = dataset.drop('target', axis=1)

# Identify categorical columns (non-numeric)
categorical_columns = dataset.select_dtypes(include=['object']).columns

# Apply one-hot encoding to categorical columns
from sklearn.preprocessing import LabelEncoder
# Custom function to apply label encoding to high cardinality columns
def label_encode_columns(df, columns):
    for col in columns:
        if df[col].nunique() > 10:  # Threshold for high cardinality
            df[col] = LabelEncoder().fit_transform(df[col])
    return df

# Apply label encoding to high cardinality columns
dataset = label_encode_columns(dataset, categorical_columns)

one_hot_encoder = OneHotEncoder(sparse=True)  # Keep the encoded data in dense format
encoded_categorical = one_hot_encoder.fit_transform(dataset[categorical_columns])

# Drop original categorical columns
dataset = dataset.drop(categorical_columns, axis=1)

# Combine dense encoded categorical data with the rest of the dataset
numeric_features = dataset  # Assuming the remaining features are numeric
X = sp.hstack((numeric_features, encoded_categorical), format="csr")

# Normalizing the feature values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler(with_mean=False)  

# Building the Neural Network model
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=6, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Calculate additional performance metrics after model training
y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=-1)
y_test_classes = y_test.argmax(axis=-1)
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)

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

# Train the model
model.fit(X_train, y_train, epochs=1, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Predictions
y_pred = model.predict(X_test)

# Converting predictions from one-hot encoded back to a single column
y_pred = y_pred.argmax(axis=1)
y_test = y_test.argmax(axis=1)

# Calculating additional metrics
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, model.predict(X_test), multi_class='ovr')

# Calculating confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)   

# Print the metrics
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'ROC AUC: {roc_auc}')
print(f'Confusion matrix: {conf_matrix}')
print('False Positive Rate by class:', fpr)
print('True Positive Rate by class:', tpr)
print('False Negative Rate by class:', fnr)
print('True Negative Rate by class:', tnr)

