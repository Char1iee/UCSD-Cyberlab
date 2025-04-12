import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import scipy.sparse as sp

# Load datasets
dftrain = pd.read_csv("train70_reduced.csv")
dftest = pd.read_csv("test30_reduced.csv")

# Train datasets
class_names = dftrain.target.unique()
dftrain = dftrain.astype('category')
cat_columns = dftrain.select_dtypes(['category']).columns
dftrain[cat_columns] = dftrain[cat_columns].apply(lambda x: x.cat.codes)
x_columns = dftrain.columns.drop('target')
x_train = dftrain[x_columns].values
y_train = dftrain['target']

# Test datasets
class_names = dftest.target.unique()
dftest = dftest.astype('category')
cat_columns = dftest.select_dtypes(['category']).columns
dftest[cat_columns] = dftest[cat_columns].apply(lambda x: x.cat.codes)
x_columns = dftest.columns.drop('target')
x_test = dftest[x_columns].values
y_test = dftest['target']

# If num_classes is 2, classification is binary
num_classes = len(class_names)
print("Number of unique classes:", num_classes)

# Reshape x_train and x_test to add a timesteps dimension
# Assuming each sample is a sequence of 1 timestep
x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

# Building the RNN model
rnn = Sequential()

# Fully connected layer
rnn.add(Dense(64, activation='relu'))

# Dropout for regularization
rnn.add(Dropout(0.5))

# Output layer
rnn.add(Dense(6, activation='softmax'))

# Compiling the rnn
rnn.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training the rnn
rnn.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Predict metrics
y_pred_rnn = rnn.predict(x_test)
y_pred_rnn = np.squeeze(y_pred_rnn)  # Remove the extra dimension
y_pred_rnn = np.argmax(y_pred_rnn, axis=1)
y_pred_rnn_proba = rnn.predict(x_test)
y_pred_rnn_proba = np.squeeze(y_pred_rnn_proba) # Remove extra dimension to ensure a 2D array

# Now you can use y_test_int_labels for evaluation metrics
print("Recurrent Neural Network \n Accuracy: " + str(accuracy_score(y_test, y_pred_rnn)))
print("F1 score: " + str(f1_score(y_test, y_pred_rnn, average='weighted')))
print("Precision: " + str(precision_score(y_test, y_pred_rnn, average='weighted')))
print("AUC Score: " + str(roc_auc_score(y_test, y_pred_rnn_proba, multi_class='ovo', average='weighted')))
matrixrnn = confusion_matrix(y_test,y_pred_rnn)
print(matrixrnn)

def calculate_classification_metrics(conf_matrix):
    # Initialize arrays to hold the metrics for each class
    TPRs, FPRs, FNRs, TNRs = [], [], [], []

    # Calculate metrics for each class
    for i in range(len(conf_matrix)):
        TP = conf_matrix[i, i]
        FN = sum(conf_matrix[i, :]) - TP
        FP = sum(conf_matrix[:, i]) - TP
        TN = conf_matrix.sum() - (TP + FP + FN)

        TPR = TP / (TP + FN) if (TP + FN) != 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) != 0 else 0
        FNR = FN / (TP + FN) if (TP + FN) != 0 else 0
        TNR = TN / (TN + FP) if (TN + FP) != 0 else 0

        TPRs.append(TPR)
        FPRs.append(FPR)
        FNRs.append(FNR)
        TNRs.append(TNR)

    # Calculate average of each metric
    avg_TPR = np.mean(TPRs)
    avg_FPR = np.mean(FPRs)
    avg_FNR = np.mean(FNRs)
    avg_TNR = np.mean(TNRs)

    return avg_TPR, avg_FPR, avg_FNR, avg_TNR

avg_TPR, avg_FPR, avg_FNR, avg_TNR = calculate_classification_metrics(matrixrnn)
print(f"Average TPR: {avg_TPR}, Average FPR: {avg_FPR}, Average FNR: {avg_FNR}, Average TNR: {avg_TNR}")