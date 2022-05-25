import os
import json
import boto3
from boto3.dynamodb.conditions import Key, Attr
from botocore.exceptions import ClientError
from decimal import Decimal
import time
import datetime as dt

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

# Create a client for the AWS Analytical service to use
client = boto3.client('glue')

sagemaker = boto3.client('sagemaker')

def datetimeconverter(o):
    if isinstance(o, dt.datetime):
        return o.__str__()

def lambda_handler(event, context):
    """Calls custom job waiter developed by user

    Arguments:
        event {dict} -- Dictionary with details on previous processing step
        context {dict} -- Dictionary with details on Lambda context

    Returns:
        {dict} -- Dictionary with Processed Bucket, Key(s) and Job Details
    """
    try:

        logger.info('Lambda event is [{}]'.format(event))
        
        logger.info(event['body'])
        job_name = event['body']['targetJob']
        processed_dir = event['body']['processedDir']
        input_dir = event['body']['inputDir']
        token = event['body']['token']

        # Submitting a new Glue Job
        job_response = client.start_job_run(
            JobName=job_name,
            Arguments={
                # Specify any arguments needed based on bucket and keys (e.g. input/output S3 locations)
                '--job-bookmark-option': 'job-bookmark-enable',
                '--additional-python-modules': 'pyarrow==2,awswrangler==2.9.0,fsspec==0.7.4',
                # Custom arguments below
                '--PROCESSED_DIR': processed_dir,
                '--INPUT_DIR': input_dir,
            },
            MaxCapacity=2.0
        )

        logger.info('Response is [{}]'.format(job_response))

        # Collecting details about Glue Job after submission (e.g. jobRunId for Glue)
        json_data = json.loads(json.dumps(job_response, default=datetimeconverter))

        job_details = {
            "jobName": job_name,
            "jobRunId": json_data.get('JobRunId'),
            "jobStatus": 'STARTED',
            "trainUri": processed_dir+"train/train.csv",
            "validationUri": processed_dir+"validation/validation.csv",
            "testUri": processed_dir+"test/test.csv",
            "token": token
        }

        response = {
            'jobDetails': job_details
        }

    except Exception as e:
        logger.error("Fatal error", exc_info=True)
        
        sagemaker.send_pipeline_execution_step_failure(
            CallbackToken=token,
            FailureReason="error"
        )
        
        raise e
    return response
