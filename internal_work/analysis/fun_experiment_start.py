import mlflow

def fun_experiment_start(experiment_name, data_name):
    """Starts an experiment and versions the data.

    Args:
        experiment_name (str): Name of experiment.
        data_name (DataFrame): The DataFrame with no quotations.
    """
    experiment_name = experiment_name
    experiment = mlflow.set_experiment(experiment_name)
    experiment_id = experiment.experiment_id
    mlflow.start_run(experiment_id=experiment_id)
    tags = {"dataset": data_name.name, "version": data_name.version}
    mlflow.set_tags(tags)