import boto3
import sys
from datetime import datetime
import json

import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
)

from awsglue.utils import getResolvedOptions

s3_client = boto3.client('s3')
sagemaker_client = boto3.client('sagemaker')    
sagemaker_runtime_client = boto3.Session().client('sagemaker-runtime')
glue_client = boto3.client("glue")


class ModelRun:

    def __init__(self):
        args = getResolvedOptions(sys.argv, ['WORKFLOW_NAME', 'WORKFLOW_RUN_ID', 'train_input_path', 'model_output_path', 'algorithm_image', 'role_arn'])
        current_time = datetime.now()
        self.train_input_path = args['train_input_path']
        self.model_output_path = args['model_output_path']
        self.algorithm_image = args['algorithm_image']
        self.role_arn = args['role_arn']
        timestamp_suffix = str(current_time.month) + "-" + str(current_time.day) + "-" + str(current_time.hour) + "-" + str(current_time.minute)
        self.training_job_name = 'gw-xgb-churn-pred' + timestamp_suffix
        
        # by default, a test data set is used to evaluate the model performance
        self.evaluation_data_set_s3_uri = f"{self.train_input_path}/test/test.csv"
        
        # get run properties of the workflow
        workflow_name = args['WORKFLOW_NAME']
        workflow_run_id = args['WORKFLOW_RUN_ID']
        workflow_params = glue_client.get_workflow_run_properties(Name=workflow_name,
                                                RunId=workflow_run_id)["RunProperties"]

        self.endpoint = workflow_params['endpoint_name']
        self.evaluation_threshold = 0.95 if 'evaluation_threshold' not in workflow_params else float(workflow_params['evaluation_threshold'])
        
    def create_training_job(self):
        print("===Create Training Job===")
        
        try:
            response = sagemaker_client.create_training_job(
                TrainingJobName=self.training_job_name,
                HyperParameters={
                    'max_depth': '5',
                    'eta': '0.2',
                    'gamma': '4',
                    'min_child_weight': '6',
                    'subsample': '0.8',
                    'silent': '0',
                    'objective': 'binary:logistic',
                    'num_round': '100',
                    'eval_metric': 'auc'
                },
                AlgorithmSpecification={
                    'TrainingImage': self.algorithm_image,
                    'TrainingInputMode': 'File'
                },
                RoleArn=self.role_arn,
                InputDataConfig=[
                    {
                        'ChannelName': 'train',
                        'DataSource': {
                            'S3DataSource': {
                                'S3DataType': 'S3Prefix',
                                'S3Uri': self.train_input_path + '/train',
                                'S3DataDistributionType': 'FullyReplicated'
                            }
                        },
                        'ContentType': 'text/csv',
                        'CompressionType': 'None'
                    },
                    {
                        'ChannelName': 'validation',
                        'DataSource': {
                            'S3DataSource': {
                                'S3DataType': 'S3Prefix',
                                'S3Uri': self.train_input_path + '/validation',
                                'S3DataDistributionType': 'FullyReplicated'
                            }
                        },
                        'ContentType': 'text/csv',
                        'CompressionType': 'None'
                    }
                ],
                OutputDataConfig={
                    'S3OutputPath': self.model_output_path
                },
                ResourceConfig={
                    'InstanceType': 'ml.m5.xlarge',
                    'InstanceCount': 1,
                    'VolumeSizeInGB': 20
                },
                StoppingCondition={
                    'MaxRuntimeInSeconds': 86400
                }
            )
            print("Training job has been created...")
        except Exception as e:
            print(e)
            print('Unable to create training job')
            raise(e)
            
    def describe_training_job(self):
        print("===Describe Training Job===")
        status = sagemaker_client.describe_training_job(
            TrainingJobName=self.training_job_name
        )
        print(self.training_job_name + " job status: ", status)
        print("Waiting for " + self.training_job_name + " training job to complete...")
        sagemaker_client.get_waiter('training_job_completed_or_stopped').wait(TrainingJobName=self.training_job_name)
        resp = sagemaker_client.describe_training_job(TrainingJobName=self.training_job_name)
        status = resp['TrainingJobStatus']
        print("Training job " + self.training_job_name + " ended with status: " + status)
        if status == 'Failed':
            message = resp['FailureReason']
            print('Training job {} failed with the following error: {}'.format(self.training_job_name, message))
            raise Exception('Creation of sagemaker Training job failed')
        return status

    def create_endpoint_config(self):

        endpoint_name = self.endpoint

        print("===Create Model===")
        create_model = sagemaker_client.create_model(
            ModelName=endpoint_name,
            PrimaryContainer=
            {
                'Image': self.algorithm_image,
                'ModelDataUrl': f"{self.model_output_path}/{self.training_job_name}/output/model.tar.gz"
            },
            ExecutionRoleArn=self.role_arn
        )

        resp = sagemaker_client.create_endpoint_config(
            EndpointConfigName=endpoint_name,
            ProductionVariants=[
                {
                    'VariantName': '{}-variant-1'.format(endpoint_name),
                    'ModelName': endpoint_name,
                    'InitialInstanceCount': 1,
                    'InstanceType': 'ml.m5.large'
                }
            ])

        print(resp)
        return resp

    def create_endpoint(self):
        print("===Create Endpoint===")
        endpoint_name = self.endpoint
        response = sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_name
        )
        print(response)

    def describe_endpoint(self):
        print("===Describe Endpoint===")
        status = sagemaker_client.describe_endpoint(EndpointName=self.endpoint)['EndpointStatus']
        print(self.endpoint + "endpoint is now in status:", status)
        print("Waiting for " + self.endpoint + " to be In-service...")
        sagemaker_client.get_waiter('endpoint_in_service').wait(EndpointName=self.endpoint)
        resp = sagemaker_client.describe_endpoint(EndpointName=self.endpoint)
        status = resp['EndpointStatus']
        print(self.endpoint + " endpoint is now in status:", status)
        if status == 'Failed':
            message = resp['FailureReason']
            print('Test Endpoint {} creation failed with the following error: {}'.format(self.endpoint, message))
            raise Exception('Endpoint creation failed')
        return status

    
    def create_batch_transform_job(self):
        batch_job_name = self.batch_transform_job_name
        model_name = self.model_name
        inference_output_location = self.inference_output_location
        inference_input_location = self.inference_input_location
        
        request = {
            "TransformJobName": batch_job_name,
            "ModelName": model_name,
            "TransformOutput": {
                "S3OutputPath": inference_output_location,
                "Accept": "text/csv",
                "AssembleWith": "Line",
            },
            "TransformInput": {
                "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": inference_input_location}},
                "ContentType": "text/csv",
                "SplitType": "Line",
                "CompressionType": "None",
            },
            "TransformResources": {"InstanceType": "ml.m5.xlarge", "InstanceCount": 1},
        }
        sagemaker_client.create_transform_job(**request)
        print("Created Transform job with name: ", batch_job_name)
        
    
    def evaluate_model(self):
        # download the data
        uri_components = self.evaluation_data_set_s3_uri.split('/')
        bucket_name = uri_components[2]
        key = '/'.join(uri_components[3:])
        
        obj = s3_client.get_object(Bucket=bucket_name, Key=key)
        df = pd.read_csv(obj['Body'], header=None)

        
        # df = pd.read_csv(self.evaluation_data_set_s3_uri, header=None)

        payload = df[df.columns[1:]].to_csv(header=False, index=False).encode("utf-8")

        response = sagemaker_runtime_client.invoke_endpoint(
            EndpointName=self.endpoint, 
            ContentType='text/csv', 
            Body=payload)

        result = response['Body'].read().decode()

        prediction_probabilities = np.asarray(result.split(','), dtype=float)
        predictions = np.round(prediction_probabilities)

        y_test = df[0]

        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        accuracy = accuracy_score(y_test, predictions)
        conf_matrix = confusion_matrix(y_test, predictions)
        fpr, tpr, _ = roc_curve(y_test, prediction_probabilities)

        # prettify the evaluation result printing
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
        print("===Evaluation Result===")
        print(json.dumps(report_dict))
        
        return accuracy, precision, recall, conf_matrix
    
    def review_evaluation_result(self, accuracy):
        """
        May delete the endpoint & related configuration if accuracy metric is less than evaluation threshold.
        """
        if accuracy < self.evaluation_threshold:
            print("===Deleting the Endpoint & Endpoint Configuration===")
            sagemaker_client.delete_endpoint_config(EndpointConfigName=self.endpoint)
            sagemaker_client.delete_endpoint(EndpointName=self.endpoint)
            print("Done")
        else:
            print("===Not Delete Endpoint===")
            print(f"accuracy metric {accuracy} is larger or equal than threshold {self.evaluation_threshold}, hence, not delete the endpoint.")
        
        
    
if __name__ == '__main__':

    
    obj = ModelRun()

    # Create training job
    obj.create_training_job()

    # Describe training job
    status = obj.describe_training_job()

    # Create endpoint conf
    resp = obj.create_endpoint_config()

    # Create endpoint for model
    obj.create_endpoint()

    # Describe endpoint
    status = obj.describe_endpoint()

    # Evaluate model
    accuracy, _, _, _ = obj.evaluate_model()

    # Review evluation result
    obj.review_evaluation_result(accuracy)    