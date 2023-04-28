import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RepeatedKFold
from mlxtend.regressor import StackingCVRegressor
from skopt import BayesSearchCV
import numpy as np

def fun_regressor_stacking_grad_boost_bayes(
    model_1, model_2, n_splits, n_repeats, x_train, y_train
):

    """
    Perform hyperparameter tuning using Bayesian optimization with stacked gradient boosting regressors.

    Args:
        model_1 (xgboost.XGBRegressor): A machine learning model from XGBoost library for regression, 
            with `objective` set to `"reg:squarederror"` and `n_estimators` set to 100.
        model_2 (lightgbm.LGBMRegressor): A machine learning model from LightGBM library for regression, 
            with `objective` set to `"regression"` and `n_estimators` set to 100.
        n_splits (int): An integer indicating the number of splits for cross-validation.
        n_repeats (int): An integer indicating the number of times to repeat the cross-validation process.
        x_train (numpy.ndarray or pandas.core.frame.DataFrame): A numpy array or Pandas dataframe 
            containing the input features of the training data.
        y_train (numpy.ndarray or pandas.core.series.Series): A numpy array or Pandas series 
            containing the target values of the training data.

    Returns:
        skopt.searchcv.BayesSearchCV: A BayesSearchCV object that contains the results of 
            hyperparameter tuning using Bayesian optimization with stacked gradient boosting regressors. 
            This object can be used to obtain the best hyperparameters and the best score achieved during 
            the hyperparameter tuning process.
    """
    estimators = [
        ("model_1", model_1),
        ("model_2", model_2),
    ]

    model_hist_grad_boost = HistGradientBoostingRegressor()  # defining meta-regressor
    clf_stack = StackingCVRegressor(
        regressors=[model_1, model_2],
        meta_regressor=model_hist_grad_boost,
        use_features_in_secondary=True,
    )
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=2652124)

    search_spaces = {
        "xgbregressor__learning_rate": (0.01, 0.3),
        "xgbregressor__max_depth": (2, 5),
        "xgbregressor__subsample": (0.3, 0.7),
        "xgbregressor__colsample_bytree": (0.1, 0.7),
        "xgbregressor__reg_lambda": (1, 3.0),
        "xgbregressor__gamma": (0, 1),
        "xgbregressor__min_child_weight": (1, 10),
        "lgbmregressor__learning_rate": (0.01, 0.3),
        "lgbmregressor__max_depth": (2, 5),
        "lgbmregressor__num_leaves": (20, 40),
        "lgbmregressor__feature_fraction": (0.1, 0.7),
        "lgbmregressor__subsample": (0.3, 0.7),
        "lgbmregressor__lambda_l1": (1, 3.0),
        "lgbmregressor__min_data_in_leaf": (1, 10),
        "meta_regressor__learning_rate": (0.01, 0.3),
        "meta_regressor__max_depth": (2, 5),
    }

    bayes_cv_tuner = BayesSearchCV(
        estimator=clf_stack,
        search_spaces=search_spaces,
        cv=rkf,
        verbose=0,
        refit=True,
    )

    result_bayes_tune = bayes_cv_tuner.fit(x_train, np.ravel(y_train))
    return result_bayes_tune