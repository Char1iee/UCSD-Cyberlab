import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
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

# Building the Neural Network model
dnn = Sequential()
dnn.add(Dense(50, input_dim=x_train.shape[1], kernel_initializer='normal', activation='relu'))
dnn.add(Dense(30, input_dim=x_train.shape[1], kernel_initializer='normal', activation='relu'))
dnn.add(Dense(20, kernel_initializer='normal'))
dnn.add(Dense(6,activation='softmax'))
dnn.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
dnn.fit(x_train, y_train, epochs=10, batch_size=256) # Larger batch size for fewer batches, may result in less accuracy

# Predict metrics
y_pred_dnn = dnn.predict(x_test)
y_pred_dnn = np.argmax(y_pred_dnn,axis=1)
y_pred_dnn_proba = dnn.predict(x_test)

print("Deep Neural Network: \n Accuracy: " + str(accuracy_score(y_test, y_pred_dnn)) + " F1 score:" + str(f1_score(y_test, y_pred_dnn,average='weighted')) + " Precision: ", str(precision_score(y_test, y_pred_dnn, average='weighted')))
print("AUC Score: " + str(roc_auc_score(y_test, y_pred_dnn_proba, multi_class='ovo', average='weighted')))
matrixdnn = confusion_matrix(y_test,y_pred_dnn)
print(matrixdnn)

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

avg_TPR, avg_FPR, avg_FNR, avg_TNR = calculate_classification_metrics(matrixdnn)
print(f"Average TPR: {avg_TPR}, Average FPR: {avg_FPR}, Average FNR: {avg_FNR}, Average TNR: {avg_TNR}")
