import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
import numpy as np

# Load datasets
dftrain = pd.read_csv("UNSW_NB15_training-set.csv")
dftest = pd.read_csv("UNSW_NB15_testing-set.csv")

# Train datasets
class_names = dftrain.label.unique()
dftrain = dftrain.astype('category')
cat_columns = dftrain.select_dtypes(['category']).columns
dftrain[cat_columns] = dftrain[cat_columns].apply(lambda x: x.cat.codes)
x_columns = dftrain.columns.drop('label')
x_train = dftrain[x_columns].values
y_train = dftrain['label']
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[1], 1))

# Test datasets
class_names = dftest.label.unique()
dftest = dftest.astype('category')
cat_columns = dftest.select_dtypes(['category']).columns
dftest[cat_columns] = dftest[cat_columns].apply(lambda x: x.cat.codes)
x_columns = dftest.columns.drop('label')
x_test = dftest[x_columns].values
y_test = dftest['label']
x_test = x_test.reshape((x_test.shape[0],x_test.shape[1],x_test.shape[1],1))

# Building the Neural Network model
cnn = Sequential()
cnn.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
cnn.add(MaxPool2D(2,2))
cnn.add(Flatten())
cnn.add(Dense(units=10, activation='softmax'))

# Compile the cnn
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
cnn.fit(x_train, y_train, epochs=1, batch_size=256) # Larger batch size for fewer batches, may result in less accuracy

# Predict metrics
y_pred_cnn = cnn.predict(x_test)
y_pred_cnn = np.argmax(y_pred_cnn,axis=1)
y_pred_cnn_proba = cnn.predict(x_test)

print("Deep Neural Network: \n Accuracy: " + str(accuracy_score(y_test, y_pred_cnn)) + " F1 score:" + str(f1_score(y_test, y_pred_cnn,average='weighted')) + " Precision: ", str(precision_score(y_test, y_pred_cnn, average='weighted')))
print("AUC Score: " + str(roc_auc_score(y_test, y_pred_cnn_proba, multi_class='ovo', average='weighted')))
matrixcnn = confusion_matrix(y_test,y_pred_cnn)
print(matrixcnn)

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

avg_TPR, avg_FPR, avg_FNR, avg_TNR = calculate_classification_metrics(matrixcnn)
print(f"Average TPR: {avg_TPR}, Average FPR: {avg_FPR}, Average FNR: {avg_FNR}, Average TNR: {avg_TNR}")
