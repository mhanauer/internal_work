import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, median_absolute_error


def fun_reg_accuracy_calculator(y_test, y_pred):
    """Cleans up regression accuracy measures using y_test and y_pred

    Args:
        y_test (int): y_test
        y_pred (int): y_pred

    Returns:
        DataFrame: Returns a DataFrame for r2 and mae
    """
    r2_result = r2_score(y_true=y_test, y_pred=y_pred)
    mae_result = median_absolute_error(y_true=y_test, y_pred=y_pred)
    r2_mae_result_dic = {"r2": r2_result, "mea": mae_result}
    r2_mae_result = pd.DataFrame(
        {"label": ["r2", "mae"], "result": [r2_result, mae_result]}
    )
    r2_mae_result["result"] = r2_mae_result["result"].round(2)
    return [r2_mae_result, r2_mae_result_dic]