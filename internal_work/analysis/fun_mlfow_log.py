import mlflow

def fun_mlfow_log(model, experiment_name, results_accuracy):
    """Log and register a model along with accuracy results.  Accuracy results must be in dictionary form.

    Args:
        model (model): Model to register and log.
        experiment_name (str): Experiment name.
        results_accuracy (dic): Dictionary accuracy results.
    """
    mlflow.sklearn.log_model(
        model, artifact_path="model"
    )  # Name the folder where the model will be stored and grabbed in the path below
    run = mlflow.active_run()
    model_uri = "runs:/{}/model".format(run.info.run_id)
    mlflow.register_model(model_uri, experiment_name)  # Name of the experiment / job
    mlflow.log_metrics(results_accuracy)
    mlflow.end_run()