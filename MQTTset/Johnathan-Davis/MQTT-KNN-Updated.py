import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

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

# Define the model and parameters for hypertuning
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred_knn = knn.predict(x_test) 
y_pred_knn_proba = knn.predict_proba(x_test) # predict_proba finds probabilities of the classes, which is required for the ROC AUC score

print("K Nearest Neighbor: \n Accuracy: " + str(accuracy_score(y_test, y_pred_knn)) + " F1 score: " + str(f1_score(y_test, y_pred_knn,average='weighted')) + " Precision: ", str(precision_score(y_test, y_pred_knn, average='weighted')))
print("AUC Score: " + str(roc_auc_score(y_test, y_pred_knn_proba, multi_class='ovo', average='weighted')))
matrixknn = confusion_matrix(y_test,y_pred_knn)
print(matrixknn)

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

avg_TPR, avg_FPR, avg_FNR, avg_TNR = calculate_classification_metrics(matrixknn)
print(f"Average TPR: {avg_TPR}, Average FPR: {avg_FPR}, Average FNR: {avg_FNR}, Average TNR: {avg_TNR}")
