
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

def check_job_status(job_details):
    # This function checks the status of the currently running job
    job_response = client.get_job_run(JobName=job_details['jobName'], RunId=job_details['jobRunId'])
    json_data = json.loads(json.dumps(job_response, default=datetimeconverter))
    # IMPORTANT update the status of the job based on the job_response (e.g RUNNING, SUCCEEDED, FAILED)
    job_details['jobStatus'] = json_data.get('JobRun').get('JobRunState')

    response = {
        'jobDetails': job_details
    }
    return response

def lambda_handler(event, context):
    """Calls custom job waiter developed by user

    Arguments:
        event {dict} -- Dictionary with details on previous processing step
        context {dict} -- Dictionary with details on Lambda context

    Returns:
        {dict} -- Dictionary with Processed Bucket, Key(s) and Job Details
    """
    try:

        logger.info('Lambda event is {}'.format(event))

        job_details = event['jobDetails']

        logger.info('Checking Job Status with user custom code')
        #transform_handler = TransformHandler().stage_transform(team, dataset, stage)
        response = check_job_status(job_details)  # custom user code called
        
        if response['jobDetails']['jobStatus'] == "SUCCEEDED":
            sagemaker.send_pipeline_execution_step_success(
                CallbackToken=job_details['token'],
                OutputParameters=[
                    {
                        'Name': 'final_status',
                        'Value': 'Glue Job finished.',
                    },
                    {
                        'Name': 'trainUri',
                        'Value': job_details['trainUri'],
                    },
                    {
                        'Name': 'validationUri',
                        'Value': job_details['validationUri'],
                    },
                    {
                        'Name': 'testUri',
                        'Value': job_details['testUri'],
                    } 
                ]
            )
        elif response['jobDetails']['jobStatus'] == "FAILED":
            sagemaker.send_pipeline_execution_step_failure(
                CallbackToken=job_details['token'],
                FailureReason="unknown reason"
            )

        logger.info('Response is [{}]'.format(response))

    except Exception as e:
        logger.error("Fatal error", exc_info=True)
        sagemaker.send_pipeline_execution_step_failure(
                CallbackToken=job_details['token'],
                FailureReason=str(e)
        )
        
        raise e
    return response
