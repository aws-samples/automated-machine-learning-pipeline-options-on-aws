import os
import json
import boto3
from boto3.dynamodb.conditions import Key, Attr
from botocore.exceptions import ClientError
from decimal import Decimal
import time
from datetime import date, datetime

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

# Create a client for the AWS Analytical service to use
client = boto3.client('stepfunctions')

sagemaker = boto3.client('sagemaker')
s3 = boto3.client('s3')

sm_arn = os.environ["state_machine_arn"]

def datetimeconverter(o):
    if isinstance(o, dt.datetime):
        return o.__str__()
        
def json_serial(obj):
    """JSON serializer for objects not serializable by default"""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError("Type %s not serializable" % type(obj))

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
        # Note: For simplicity, parameters "target_job" 
        # and "target_ddb" are hardcoded values defined during deployment of thhe pipeline.
        # Other parameters can be dynamically retrieved
        for record in event['Records']:
            payload = json.loads(record["body"])
            token = payload["token"]
            arguments = payload["arguments"]
            trainUri = arguments["trainUri"]
            valUri = arguments["valUri"]
            testUri = arguments["testUri"]
            input_dir = arguments['inputDir']
            
            logger.info('Trigger execution of state machine [{}]'.format(sm_arn))

            # Prepare input to state machine
            message = {
                'statusCode': 200,
                'body': {
                    "trainUri": trainUri,
                    "valUri": valUri,
                    "testUri": testUri,
                    'inputDir': input_dir,
                    "token": token
                }
            }

            logger.info('Input Message is [{}]'.format(message))

            client.start_execution(stateMachineArn=sm_arn,input=json.dumps(message, default=json_serial))

    except Exception as e:
        logger.error("Fatal error", exc_info=True)
        sagemaker.send_pipeline_execution_step_failure(
            CallbackToken=token,
            FailureReason="Fatal error"
        )
        raise e
    return 200
