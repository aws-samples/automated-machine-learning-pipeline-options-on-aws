import boto3
import json

sm_client = boto3.client('sagemaker')
def lambda_handler(event, context):
    try:
        processing_name = event['taskResult']['ProcessingJobArn'].split('/')[-1]
        response = sm_client.describe_processing_job(ProcessingJobName=processing_name)
        return {
            'ProcessingJobStatus': response['ProcessingJobStatus'],
            'RunJobName': event['RunJobName'],
            'trainTaskResult': {
                'TrainingJobName': event['trainTaskResult']['TrainingJobName'],
                'ModelArtifacts': {
                    'S3ModelArtifacts': event['trainTaskResult']['ModelArtifacts']['S3ModelArtifacts']
                }
            },
            'taskResult': {
                'ProcessingJobArn': response['ProcessingJobArn']
            }
        }
    except:
        return {
            'ProcessingJobStatus': 'Failed'
        }