import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score


def fun_class_accuracy_calcaulator(y_test, y_predicted):
    """Calculates balanced accuracy, f1 and roc_auc clasification metrics with y_test and y_predicted.  Produces both a DataFrame and a dictionary.

    Args:
        y_test (binary): y_test from a train test split
        y_predicted (int): predicted probability from a model

    Balanced accuracy: (Sensitivity [true positives] + Specificity [true negatives]) / 2
    F1: It's often used when the classes are imbalanced and there is a serious downside to predicting false negatives

    Returns:
        DataFrame: See summary
    """
    results_balanced_accuracy = balanced_accuracy_score(y_test, y_predicted)
    results_roc_auc = roc_auc_score(y_test, y_predicted)
    results_f1_score = f1_score(y_test, y_predicted)
    frames = [
        results_balanced_accuracy,
        results_roc_auc,
        results_f1_score,
    ]
    results_dic = ({'balanced_accuracy': results_balanced_accuracy, 'roc_auc_score': results_roc_auc, 'f1_score': results_f1_score})
    results_combine = pd.DataFrame(frames, columns=["metric_score"])
    results_metrics = {
        "metric_type": ["balanced_accuracy_score", "roc_auc_score", "f1_score"]
    }
    results_metrics_pd = pd.DataFrame(results_metrics)
    frames = [results_metrics_pd, results_combine]
    results_metric = pd.concat(frames, axis=1)
    results_metric = results_metric.round(3)
    return [results_metric, results_dic]
