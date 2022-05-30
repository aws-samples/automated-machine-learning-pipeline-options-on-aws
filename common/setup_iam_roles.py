import json
import boto3

iam = boto3.client('iam')

def create_glue_role(role_name, bucket):
    try:
        response = iam.create_role(
            RoleName = role_name,
            AssumeRolePolicyDocument = json.dumps({
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {
                            "Service": "glue.amazonaws.com"
                        },
                        "Action": "sts:AssumeRole"
                    }
                ]
            }),
            Description='Role for Glue ETL job'
        )
        
        role_arn = response['Role']['Arn']
        
        role_policy_document = json.dumps({
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Sid": "ListObjectsInBucket",
                        "Effect": "Allow",
                        "Action": ["s3:ListBucket"],
                        "Resource": [f"arn:aws:s3:::{bucket}"]
                    },
                    {
                        "Sid": "AllObjectActions",
                        "Effect": "Allow",
                        "Action": "s3:*Object",
                        "Resource": [f"arn:aws:s3:::{bucket}/*"]
                    },
                    {
                        "Sid": "SageMakerExecution",
                        "Effect": "Allow",
                        "Action": "sagemaker:*",
                        "Resource": "*"
                    },
                    {
                        "Action": [
                            "logs:CreateLogStream",
                            "logs:DescribeLogStreams",
                            "logs:PutLogEvents",
                            "logs:CreateLogGroup"
                        ],
                        "Resource": [
                            "arn:aws:logs:*:*:log-group:/aws/sagemaker/*",
                            "arn:aws:logs:*:*:log-group:/aws/sagemaker/*:log-stream:aws-glue-*"
                        ],
                        "Effect": "Allow"
                    },
                    {
                        "Effect": "Allow",
                        "Action": [
                            "iam:PassRole"
                        ],
                        "Resource": "arn:aws:iam::*:role/*AmazonSageMaker*"
                    }
                ]
            })
        
        response = iam.put_role_policy(
            RoleName=role_name,
            PolicyName=f'{role_name}_S3SageMakerAccessPolicy',
            PolicyDocument=role_policy_document
        )
        
        return role_arn

    except iam.exceptions.EntityAlreadyExistsException:
        print(f'Using ARN from existing role: {role_name}')
        response = iam.get_role(RoleName=role_name)
        return response['Role']['Arn']

def create_sfn_role(role_name, notebook_role_arn, glue_job_prefix):
    try:
        response = iam.create_role(
            RoleName = role_name,
            AssumeRolePolicyDocument = json.dumps({
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {
                            "Service": "states.amazonaws.com"
                        },
                        "Action": "sts:AssumeRole"
                    }
                ]
            }),
            Description='Role for Lambda to step function'
        )

        role_arn = response['Role']['Arn']
        
        response = iam.attach_role_policy(
            RoleName=role_name,
            PolicyArn='arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole'
        )
        
        response = iam.attach_role_policy(
            RoleName=role_name,
            PolicyArn='arn:aws:iam::aws:policy/CloudWatchFullAccess',
        )
        
        role_policy_document = json.dumps({
            "Version": "2012-10-17",
            "Statement": [{
                    "Effect": "Allow",
                    "Action": "iam:PassRole",
                    "Resource": notebook_role_arn,
                    "Condition": {
                        "StringEquals": {
                            "iam:PassedToService": "sagemaker.amazonaws.com"
                        }
                    }
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "sagemaker:CreateModel",
                        "sagemaker:CreateProcessingJob",
                        "sagemaker:CreateEndpointConfig",
                        "sagemaker:DeleteEndpointConfig",
                        "sagemaker:CreateEndpoint",
                        "sagemaker:UpdateEndpoint",
                        "sagemaker:DeleteEndpoint",
                        "sagemaker:DescribeTrainingJob",
                        "sagemaker:StopTrainingJob",
                        "sagemaker:CreateTrainingJob"
                    ],
                    "Resource": [
                        "arn:aws:sagemaker:*:*:*"
                    ]
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "events:DescribeRule",
                        "events:PutRule",
                        "events:PutTargets"
                    ],
                    "Resource": [
                        "arn:aws:events:*:*:rule/*"
                    ]
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "lambda:InvokeFunction"
                    ],
                    "Resource": [
                        f"arn:aws:lambda:*:*:*"
                    ]
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "glue:StartJobRun",
                        "glue:GetJobRun",
                        "glue:BatchStopJobRun",
                        "glue:GetJobRuns"
                    ],
                    "Resource": f"arn:aws:glue:*:*:job/{glue_job_prefix}*"
                }
            ]
        })
        
        response = iam.put_role_policy(
            RoleName=role_name,
            PolicyName=f'{role_name}SFNWorkflowExecutionPolicy',
            PolicyDocument=role_policy_document
        )
        
        return role_arn
    except iam.exceptions.EntityAlreadyExistsException:
        print(f'Using ARN from existing role: {role_name}')
        response = iam.get_role(RoleName=role_name)
        return response['Role']['Arn']

def create_lambda_role(role_name, bucket):
    try:
        response = iam.create_role(
            RoleName = role_name,
            AssumeRolePolicyDocument = json.dumps({
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {
                            "Service": "lambda.amazonaws.com"
                        },
                        "Action": "sts:AssumeRole"
                    }
                ]
            }),
            Description='Role for Lambda to step function'
        )

        role_arn = response['Role']['Arn']

        response = iam.attach_role_policy(
            RoleName=role_name,
            PolicyArn='arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole'
        )
        
        response = iam.attach_role_policy(
            RoleName=role_name,
            PolicyArn='arn:aws:iam::aws:policy/AmazonSageMakerFullAccess',
        )

        role_policy_document = json.dumps({
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": "s3:GetObject",
                    "Resource": [f"arn:aws:s3:::{bucket}/*"]
                }
            ]
        })

        response = iam.put_role_policy(
            RoleName=role_name,
            PolicyName=f'{role_name}LambdaPolicy',
            PolicyDocument=role_policy_document
        )
        return role_arn
    except iam.exceptions.EntityAlreadyExistsException:
        print(f'Using ARN from existing role: {role_name}')
        response = iam.get_role(RoleName=role_name)
        return response['Role']['Arn']
    

def delete_role(role_name):
    role_def = iam.list_attached_role_policies(RoleName=role_name)
    policies_list = iam.list_role_policies(RoleName=role_name)['PolicyNames']
    
    try:
        for policy_name in role_def["AttachedPolicies"]:
            iam.detach_role_policy(RoleName=role_name, PolicyArn=policy_name["PolicyArn"])
        
        for policy_name in policies_list:
            iam.delete_role_policy(RoleName=role_name, PolicyName=policy_name)
            #iam.delete_policy(PolicyArn=policy_name["PolicyArn"])
        iam.delete_role(RoleName=role_name)
        print(f'Deleted {role_name} successfully.')
    except Exception as e:
        print(f'Failed to delete {role_name}. {e}')