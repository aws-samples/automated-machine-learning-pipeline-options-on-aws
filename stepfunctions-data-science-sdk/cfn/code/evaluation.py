print('evaluation....')
import json
import os
import tarfile
import pathlib
import pandas as pd
import subprocess
import numpy as np

subprocess.run("pip install xgboost", shell=True)
import xgboost

from sklearn.externals import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
)
import logging
import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

s3_client = boto3.client('s3')

if __name__ == "__main__":
    model_artifacts_url = json.loads(os.environ['model_url'])['uri']
    print('model_artifacts_url: ', model_artifacts_url)
    model_artifacts_bucket = model_artifacts_url.split('/')[3]
    model_artifacts_key = model_artifacts_url.split('/')[4:]
    model_artifacts_key = '/'.join(model_artifacts_key)

    print(model_artifacts_bucket, model_artifacts_key)

    model_dir = "/opt/ml/processing/model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, "model.tar.gz")

    s3_client.download_file(model_artifacts_bucket, model_artifacts_key, model_path)
    print("Extracting model from path: {}".format(model_path))

    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
    
    model = joblib.load("xgboost-model")
    test_data = os.path.join("/opt/ml/processing/test", "test.csv")
    
    test_data_pd = pd.read_csv(test_data, header=None)
    
    y_test = test_data_pd.iloc[:, 0].to_numpy()
    test_data_pd.drop(test_data_pd.columns[0], axis=1, inplace=True)
    X_test = xgboost.DMatrix(test_data_pd.values)
    
    prediction_probabilities = model.predict(X_test)
    predictions = np.round(prediction_probabilities)

    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    fpr, tpr, _ = roc_curve(y_test, prediction_probabilities)

    logger.debug("Accuracy: {}".format(accuracy))
    logger.debug("Precision: {}".format(precision))
    logger.debug("Recall: {}".format(recall))
    logger.debug("Confusion matrix: {}".format(conf_matrix))
    
    """
    Need to customise the formation of evaluation.json to visualize the metrics in model registry based-on tasks
    Result should coincide in format with the result by sagemaker.workflow.quality_check_step.QualityCheckStep.
    Binarry
    """
    
    report_dict = {
        "binary_classification_metrics": {
            "accuracy": {"value": accuracy, "standard_deviation": "NaN"},
            "precision": {"value": precision, "standard_deviation": "NaN"},
            "recall": {"value": recall, "standard_deviation": "NaN"},
            "confusion_matrix": {
                "0": {"0": int(conf_matrix[0][0]), "1": int(conf_matrix[0][1])},
                "1": {"0": int(conf_matrix[1][0]), "1": int(conf_matrix[1][1])},
            },
            "receiver_operating_characteristic_curve": {
                "false_positive_rates": list(fpr),
                "true_positive_rates": list(tpr),
            },
        },
    }
    
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"

    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))