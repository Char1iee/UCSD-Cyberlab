import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier

# Load datasets
dftest = pd.read_csv("UNSW_NB15_testing-set.csv")
dftrain = pd.read_csv("UNSW_NB15_training-set.csv")

# Train datasets
class_names = dftrain.label.unique()
dftrain = dftrain.astype('category')
cat_columns = dftrain.select_dtypes(['category']).columns
dftrain[cat_columns] = dftrain[cat_columns].apply(lambda x: x.cat.codes)
x_columns = dftrain.columns.drop('label')
x_train = dftrain[x_columns].values
y_train = dftrain['label']

# Test datasets
class_names = dftest.label.unique()
dftest = dftest.astype('category')
cat_columns = dftest.select_dtypes(['category']).columns
dftest[cat_columns] = dftest[cat_columns].apply(lambda x: x.cat.codes)
x_columns = dftest.columns.drop('label')
x_test = dftest[x_columns].values
y_test = dftest['label']

# Define the model and parameters for hypertuning
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
y_pred_dt = dt.predict(x_test) 
y_pred_dt_proba = dt.predict_proba(x_test) # predict_proba finds probabilities of the classes, which is required for the ROC AUC score

print("Decision Tree: \n Accuracy: " + str(accuracy_score(y_test, y_pred_dt)) + " F1 score: " + str(f1_score(y_test, y_pred_dt,average='weighted')) + " Precision: ", str(precision_score(y_test, y_pred_dt, average='weighted')))
print("AUC Score: " + str(roc_auc_score(y_test, y_pred_dt_proba, multi_class='ovo', average='weighted')))
matrixdt = confusion_matrix(y_test,y_pred_dt)
print(matrixdt)

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

avg_TPR, avg_FPR, avg_FNR, avg_TNR = calculate_classification_metrics(matrixdt)
print(f"Average TPR: {avg_TPR}, Average FPR: {avg_FPR}, Average FNR: {avg_FNR}, Average TNR: {avg_TNR}")
