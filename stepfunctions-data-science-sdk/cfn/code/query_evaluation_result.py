import json
import logging

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client('s3')

# Retrieve transform job name from event and return transform job status.
def lambda_handler(event, context):
    print(event)
    eval_res = event["EvaluationResult"]
    eval_s3uri = f"{eval_res}evaluation.json"
    
    bucket_name = eval_s3uri.replace('s3://', '').split('/')[0]
    key_name = eval_s3uri.replace(f's3://{bucket_name}/', '')
    
    s3_clientobj = s3_client.get_object(Bucket=bucket_name, Key=key_name)
    
    s3_clientdata = s3_clientobj['Body'].read().decode('utf-8')
    s3clientlist = json.loads(s3_clientdata)

    return {
        "statusCode": 200,
        "trainingMetrics": s3clientlist["binary_classification_metrics"]["accuracy"]["value"],
        "RunJobName": event["RunJobName"],
        "trainTaskResult": event["trainTaskResult"]
    }
