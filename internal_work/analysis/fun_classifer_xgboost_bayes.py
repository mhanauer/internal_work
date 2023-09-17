import numpy as np
import pandas as pd
from skopt import BayesSearchCV
from sklearn.model_selection import RepeatedKFold
import xgboost as xgb


def fun_classifer_xgboost_bayes(
    x_train,
    y_train,
    n_splits=5,
    n_repeats=5,
    id_var="None",
    reg_class=["reg", "class", "count"],
):
    """Runs a classifcation xgboost model with repeated cross validation according to DS standards
    (see: https://dev.azure.com/cleanslate/DWBI/_wiki/wikis/DWBI.wiki/365/Data-Science-and-MLOps)

    Args:
        scale_pos_weight (int): scale_pos_weight value for imbalanced data
        x_train (int): x_train from testing train split.
        y_train (int): binary y_train from testing train split.
        n_splits (int): The number of splits for the repeated cross validation.
        n_repeats (int): The number of repeats for the repeated cross validation.
        id_var (str, optional): Id var you need to remove.  Likely mrn. Defaults to "None".

learning_rate: (0.01, 0.3)
    Explanation: This parameter controls how quickly or slowly a model learns during training. A smaller value like 0.01 makes the model        learn slowly, which could achieve better performance but may also lead to overfitting if not properly regulated. A larger value like        0.3 makes the model learn quickly, potentially making the learning process less accurate.
max_depth: (2, 5)
    Explanation: This parameter specifies the maximum depth of the tree in the ensemble model. A small depth can result in a simpler model      (may underfit), while a larger depth can capture more complexity but may overfit. The range here is from 2 to 5, allowing trees to be       moderately complex.
subsample: (0.3, 0.7)
    Explanation: This parameter defines the proportion of the training data to randomly sample in each boosting round. Subsampling helps in     making the model more robust by preventing overfitting. Values range between 0.3 and 0.7, meaning each boosting round uses 30% to 70%       of the data.
colsample_bytree: (0.1, 0.7)
    Explanation: This parameter indicates the fraction of features to choose for each boosting round. It's a way to perform feature         
    sampling, providing an extra layer of randomness and thus making the model more robust. The range between 0.1 and 0.7 implies using 10%     to 70% of features for each round.
reg_lambda: (1, 3.0)
    Explanation: This is the L2 regularization term on weights, used to avoid overfitting. Increasing this value will make the model more       conservative by constraining the weight values, making the model simpler but potentially underfitting. The range is between 1 and 3,        suggesting a moderate level of regularization.

"hist" (Histogram-based)
    Explanation: The histogram-based approach buckets continuous feature values into discrete bins and builds histograms during training.       It speeds up the training process considerably and is useful for datasets with a large number of rows but can result in slightly lower      accuracy due to the binning process.

    Returns:
         Model: Returns the best model based on highest auc.
    """
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=2652124)

    if reg_class == "class":
        estimator = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            n_jobs=-1,
        )

    if reg_class == "count":
        estimator = xgb.XGBClassifier(
            objective="count:poisson",
            eval_metric="mae",
            tree_method="hist",
            n_jobs=-1,
        )
    else:
        estimator = xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="mae",
            tree_method="hist",
            n_jobs=-1,
        )
    search_spaces = {
        "learning_rate": (0.01, 0.3),
        "max_depth": (2, 5),
        "subsample": (0.3, 0.7),
        "colsample_bytree": (0.1, 0.7),
        "reg_lambda": (1, 3.0),
    }
    bayes_cv_tuner = BayesSearchCV(
        estimator=estimator,
        search_spaces=search_spaces,
        cv=rkf,
        verbose=0,
        refit=True,
    )
    id_var = id_var

    if id_var == "None":
        x_train_fit = x_train
    else:
        x_train_fit = x_train.drop(columns=[id_var])
    result_bayes_tune = bayes_cv_tuner.fit(x_train_fit, np.ravel(y_train))
    best_estimator_bayes_tune = result_bayes_tune.best_estimator_
    return best_estimator_bayes_tune
