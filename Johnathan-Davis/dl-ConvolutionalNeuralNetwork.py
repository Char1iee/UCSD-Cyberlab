
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.compose import ColumnTransformer

# Load the dataset
dataset = pd.read_csv('test30_reduced.csv')

# Separate features and target
features = dataset.drop('target', axis=1)
target = dataset['target']

# Identifying categorical columns (excluding the target column)
categorical_cols = features.select_dtypes(include=['object']).columns

# Apply one-hot encoding to categorical columns using sparse matrices
ct = ColumnTransformer([("onehot", OneHotEncoder(sparse_output=True), categorical_cols)], remainder='passthrough')
transformed_features = ct.fit_transform(features)

# Encode target labels
label_encoder = LabelEncoder()
encoded_target = label_encoder.fit_transform(target)

# Normalize features
scaler = StandardScaler(with_mean=False) 
normalized_features = scaler.fit_transform(transformed_features)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(normalized_features, encoded_target, test_size=0.2, random_state=42)

# Define the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=1, batch_size=32)

# Evaluate the model
y_pred = model.predict(X_test)
loss, accuracy = model.evaluate(X_test, y_test)
cm = confusion_matrix(y_test, y_pred.argmax(axis=1))
print(classification_report(y_test, y_pred.argmax(axis=1)))

# Calculate TPR (Recall), FPR, FNR, and TNR
TP = cm[1, 1]
FP = cm[0, 1]
FN = cm[1, 0]
TN = cm[0, 0]

TPR = TP / (TP + FN)  # True Positive Rate (Recall)
FPR = FP / (FP + TN)  # False Positive Rate
FNR = FN / (TP + FN)  # False Negative Rate
TNR = TN / (TN + FP)  # True Negative Rate

print("Accuracy: {:.2f}%".format(accuracy * 100))
print(f"True Positive Rate (TPR): {TPR}")
print(f"False Positive Rate (FPR): {FPR}")
print(f"False Negative Rate (FNR): {FNR}")
print(f"True Negative Rate (TNR): {TNR}")
