import boto3
import sys
from datetime import datetime
# todo
# from awsglue.utils import getResolvedOptions

class ModelRun:

    def __init__(self, args):
        # todo
        # args = getResolvedOptions(sys.argv, ['train_input_path', 'model_output_path', 'algorithm_image', 'role_arn', 'endpoint_name'])
        current_time = datetime.now()
        self.train_input_path = args['train_input_path']
        self.model_output_path = args['model_output_path']
        self.algorithm_image = args['algorithm_image']
        self.role_arn = args['role_arn']
        timestamp_suffix = str(current_time.month) + "-" + str(current_time.day) + "-" + str(current_time.hour) + "-" + str(current_time.minute)
        self.training_job_name = 'gw-xgb-churn-pred' + timestamp_suffix
        self.endpoint = args['endpoint_name']
        
    def create_training_job(self):
        print("Started training job...")
        
        try:
            response = sagemaker.create_training_job(
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
        status = sagemaker.describe_training_job(
            TrainingJobName=self.training_job_name
        )
        print(self.training_job_name + " job status: ", status)
        print("Waiting for " + self.training_job_name + " training job to complete...")
        sagemaker.get_waiter('training_job_completed_or_stopped').wait(TrainingJobName=self.training_job_name)
        resp = sagemaker.describe_training_job(TrainingJobName=self.training_job_name)
        status = resp['TrainingJobStatus']
        print("Training job " + self.training_job_name + " ended with status: " + status)
        if status == 'Failed':
            message = resp['FailureReason']
            print('Training job {} failed with the following error: {}'.format(self.training_job_name, message))
            raise Exception('Creation of sagemaker Training job failed')
        return status

    def create_endpoint_config(self):

        endpoint_name = self.endpoint

        print("Creating model..")
        create_model = sagemaker.create_model(
            ModelName=endpoint_name,
            PrimaryContainer=
            {
                'Image': self.algorithm_image,
                'ModelDataUrl': f"{self.model_output_path}/{self.training_job_name}/output/model.tar.gz"
            },
            ExecutionRoleArn=self.role_arn
        )

        resp = sagemaker.create_endpoint_config(
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
        print("Creating endpoint..")
        endpoint_name = self.endpoint
        response = sagemaker.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_name
        )
        print(response)

    def describe_endpoint(self):
        status = sagemaker.describe_endpoint(EndpointName=self.endpoint)['EndpointStatus']
        print(self.endpoint + "endpoint is now in status:", status)
        print("Waiting for " + self.endpoint + " to be In-service...")
        sagemaker.get_waiter('endpoint_in_service').wait(EndpointName=self.endpoint)
        resp = sagemaker.describe_endpoint(EndpointName=self.endpoint)
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
        sagemaker.create_transform_job(**request)
        print("Created Transform job with name: ", batch_job_name)
        
    
if __name__ == '__main__':

    # Configure SDK to sagemaker
    sagemaker = boto3.client('sagemaker')    
    s3 = boto3.resource('s3')

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
