from sklearn.metrics import roc_curve, auc, roc_auc_score, average_precision_score, accuracy_score, \
    precision_recall_curve, confusion_matrix, precision_score, recall_score
import numpy as np
import pandas as pd


def calculate_metrics(y, y_pred_score, y_pred_value):
    """
    Compute the following metric for the given data:
    Accuracy, TPR, FPR, Precision, AUC, PR-Curve
    @param y: True labels or binary label indicators
    @param y_pred_score: Target scores
    @param y_pred_value: Target predicted label
    @return: list with the calculated metrics
    """
    # calculate confusion matrix
    cm = confusion_matrix(y, y_pred_value)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    FP = np.mean(FP.astype(float))
    FN = np.mean(FN.astype(float))
    TP = np.mean(TP.astype(float))
    TN = np.mean(TN.astype(float))

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    TPR_sk = recall_score(y, y_pred_value, average='macro')

    # Specificity or true negative rate
    TNR = TN / (TN + FP)

    # Precision or positive predictive value
    Precision = TP / (TP + FP)
    prec_score = precision_score(y, y_pred_value, average='macro')

    # Negative predictive value
    NPV = TN / (TN + FN)

    # Fall out or false positive rate
    FPR_man = FP / (FP + TN)

    # False negative rate
    FNR = FN / (TP + FN)

    # False discovery rate
    FDR = FP / (TP + FP)

    # Overall accuracy for each class
    ACC_man = (TP + TN) / (TP + FP + FN + TN)
    accuracy_sk = accuracy_score(y, y_pred_value)


    y_test_hot = pd.get_dummies(y)
    y_pred_score_df = pd.DataFrame(list(map(np.ravel, y_pred_score)))
    # Compute ROC curve and ROC area for each class - extract tpr and fpr
    fpr = []
    tpr = []
    for i in range(len(set(y))):
        fpr_i, tpr_i, _ = roc_curve(y_test_hot.iloc[:, i], y_pred_score_df.iloc[:, i])
        fpr.append(fpr_i)
        tpr.append(tpr_i)
    tpr_roc_curve = np.mean([i.mean() for i in tpr])
    fpr_roc_curve = np.mean([i.mean() for i in fpr])

    # Compute average precision a summarizes a precision-recall curve
    pr_curve = average_precision_score(y_test_hot, np.array(y).reshape(-1, 1))

    # Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
    auc = roc_auc_score(y, y_pred_score, multi_class='ovr')

    all_metric = [accuracy_sk, tpr_roc_curve, fpr_roc_curve, prec_score, auc, pr_curve]
    return accuracy_sk, all_metric

