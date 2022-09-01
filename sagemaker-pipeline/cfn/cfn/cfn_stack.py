from operator import concat
from aws_cdk import (
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as sfn_tasks,
    aws_lambda as lambda_,
    aws_lambda_event_sources as lambda_event_sources,
    aws_glue as glue,
    aws_sqs as sqs,
    aws_iam,
    core as cdk
)

from constructs import Construct

import boto3 
my_region = boto3.session.Session().region_name
my_acc_id = boto3.client('sts').get_caller_identity().get('Account')


class CfnStack(cdk.Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        execution_role_arn = cdk.CfnParameter(
            self,
            "ExecutionRoleArn",
            type="String",
            description="the arn of execution role for a SageMaker Pipeline",
            min_length=10,
        )

        # Create SageMaker Pipeline
        sagemaker_execution_role = aws_iam.Role.from_role_name(
            self, "SageMakerExecutionRole", role_name=execution_role_arn.value_as_string
        )

        
        glue_role = aws_iam.Role(self, "Role",
            assumed_by=aws_iam.ServicePrincipal("glue.amazonaws.com"),
            description="Glue role"
        )

        glue_role.add_to_policy(
            aws_iam.PolicyStatement(
                actions = ['s3:*'],
                resources = ['arn:aws:s3:::*',]
            )
        )
        
        # Create a glue job for preprocessing
        glue_job = glue.Job(
            self,
            "sagemaker-pipeline-GlueJob",
            job_name="sagemaker-pipeline-GlueJob",
            role=glue_role,
            executable=glue.JobExecutable.python_etl(
                glue_version=glue.GlueVersion.V2_0,
                python_version=glue.PythonVersion.THREE,
                script=glue.Code.from_asset(path="./code/glue_preprocessing.py"),
            ),
            description="Prepare data for SageMaker training",
            default_arguments={
                "--job-bookmark-option": "job-bookmark-enable",
                "--enable-metrics": "",
                "--additional-python-modules": "pyarrow==2,awswrangler==2.9.0,fsspec==0.7.4"
            },
            worker_count=10,
            worker_type=glue.WorkerType.STANDARD,
            max_concurrent_runs=1,
            timeout=cdk.Duration.minutes(60),
        )

        # STEP FUNCTION
        start_glue_job = sfn_tasks.GlueStartJobRun(
            self,
            "StartGlueJobTask",
            glue_job_name=glue_job.job_name,
            integration_pattern=sfn.IntegrationPattern.RUN_JOB,
            result_path="$.taskresult",
            arguments=sfn.TaskInput.from_object(
                {
                    '--job-bookmark-option': 'job-bookmark-enable',
                    '--additional-python-modules': 'pyarrow==2,awswrangler==2.9.0,fsspec==0.7.4',
                    # Custom arguments below
                    '--TRAIN_URI': sfn.JsonPath.string_at("$.body.trainUri"),
                    '--VALIDATION_URI': sfn.JsonPath.string_at("$.body.valUri"),
                    '--TEST_URI': sfn.JsonPath.string_at("$.body.testUri"),
                    '--INPUT_DIR': sfn.JsonPath.string_at("$.body.inputDir")
                }
            ),
        )

        send_success = sfn_tasks.CallAwsService(
            self,
            "SendSuccess",
            iam_resources=[f'arn:aws:sagemaker:{my_region}:{my_acc_id}:pipeline/*'],
            service="sagemaker",
            action="sendPipelineExecutionStepSuccess",
            parameters={
                "CallbackToken.$": "$.body.token",
                "OutputParameters": [
                    {
                        "Name": "trainUri",
                        "Value.$": "$.body.trainUri"
                    },
                    {
                        "Name": "valUri",
                        "Value.$": "$.body.valUri"
                    },
                    {
                        "Name": "testUri",
                        "Value.$": "$.body.testUri"
                    }
                ]
            }
        )
        send_failure = sfn_tasks.CallAwsService(
            self,
            "SendFailure",
            iam_resources=[f'arn:aws:sagemaker:{my_region}:{my_acc_id}:pipeline/*'],
            service="sagemaker",
            action="sendPipelineExecutionStepFailure",
            parameters={
                "CallbackToken.$": "$.body.token",
                "FailureReason": "Unknown reason"
                },
        )

        definition = start_glue_job.add_catch(
            send_failure,
            result_path="$.error-info",
        ).next(
            sfn.Choice(self, "Job successful?")
            .when(
                sfn.Condition.string_equals("$.taskresult.JobRunState", "SUCCEEDED"),
                send_success,
            )
            .otherwise(
                send_failure,
            )
        )

        state_machine = sfn.StateMachine(
            self, "Preprocessing",
            definition=definition,
        )

        ## Define a Lambda Functions
        with open("lambda/execute_function.py", encoding="utf8") as fp:
            lambda_exec_func_code = fp.read()

        lambdaFn1 = lambda_.Function(
            self,
            "execute_sfn_function",
            code=lambda_.InlineCode(lambda_exec_func_code),
            handler="index.lambda_handler",
            timeout=cdk.Duration.seconds(300),
            runtime=lambda_.Runtime.PYTHON_3_8,
            environment={
                "state_machine_arn": state_machine.state_machine_arn
            }
        )

        # Add perms
        lambdaFn1.add_to_role_policy(aws_iam.PolicyStatement(
            actions = ['states:StartExecution',],
            resources = [f'arn:aws:states:{my_region}:{my_acc_id}:stateMachine:*',]
            ))
        
        lambdaFn1.add_to_role_policy(aws_iam.PolicyStatement(
            actions = ['sagemaker:SendPipelineExecutionStepFailure',],
            resources = [f'arn:aws:sagemaker:{my_region}:{my_acc_id}:pipeline/*',]
            ))

        callback_queue = sqs.Queue(
            self, "pipeline_callbacks_glue_prep",
            visibility_timeout=cdk.Duration.minutes(10)
        )

        callback_queue.grant_send_messages(sagemaker_execution_role)

        lambdaFn1.add_event_source(
            lambda_event_sources.SqsEventSource(callback_queue)
        )

        cdk.CfnOutput(
            self,
            "SqsURL",
            value=callback_queue.queue_url,
            export_name="SM-Pipeline-SQS-URL",
        )
        
