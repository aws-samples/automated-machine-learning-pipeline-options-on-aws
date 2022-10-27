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


        with open("lambda/execute_function.py", encoding="utf8") as fp:
            lambda_exec_func_code = fp.read()

        lambdaFn1 = lambda_.Function(
            self,
            "execute_sfn_function",
            code=lambda_.InlineCode(lambda_exec_func_code),
            handler="index.lambda_handler",
            timeout=cdk.Duration.seconds(300),
            runtime=lambda_.Runtime.PYTHON_3_8
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
        
        lambdaFn1.add_to_role_policy(aws_iam.PolicyStatement(
            actions = [
                'sqs:ReceiveMessage',
                'sqs:DeleteMessage',
                'sqs:GetQueueAttributes'
                ],
            resources = [f'arn:aws:sqs:{my_region}:{my_acc_id}:*',]
            ))

        ## Start Glue Job
        with open("lambda/execute_glue_job.py", encoding="utf8") as fp:
            lambda_exec_glue_code = fp.read()
        
        lambdaFn2 = lambda_.Function(
            self,
            "execute_glue_job_function",
            code=lambda_.InlineCode(lambda_exec_glue_code),
            handler="index.lambda_handler",
            timeout=cdk.Duration.seconds(300),
            runtime=lambda_.Runtime.PYTHON_3_8
        )

        # Add perms
        lambdaFn2.add_to_role_policy(aws_iam.PolicyStatement(
            actions = ['glue:StartJobRun',],
            resources = [f'arn:aws:glue:{my_region}:{my_acc_id}:job/*',]
            ))
        
        lambdaFn2.add_to_role_policy(aws_iam.PolicyStatement(
            actions = ['sagemaker:SendPipelineExecutionStepFailure',],
            resources = [f'arn:aws:sagemaker:{my_region}:{my_acc_id}:pipeline/*',]
            ))

        # Create a function for checking the status of Glue job
        with open("lambda/check_glue_job.py", encoding="utf8") as fp:
            lambda_check_glue_code = fp.read()
        
        lambdaFn3 = lambda_.Function(
            self,
            "check_glue_job_function",
            code=lambda_.InlineCode(lambda_check_glue_code),
            handler="index.lambda_handler",
            timeout=cdk.Duration.seconds(300),
            runtime=lambda_.Runtime.PYTHON_3_8
        )

        # Add perms
        lambdaFn3.add_to_role_policy(aws_iam.PolicyStatement(
            actions = ['glue:GetJobRun',],
            resources = [f'arn:aws:glue:{my_region}:{my_acc_id}:job/*',]
            ))
        
        lambdaFn3.add_to_role_policy(aws_iam.PolicyStatement(
            actions = [
                'sagemaker:SendPipelineExecutionStepFailure',
                'sagemaker:SendPipelineExecutionStepSuccess'
                ],
            resources = [f'arn:aws:sagemaker:{my_region}:{my_acc_id}:pipeline/*',]
            ))

        # Create a step functions for preprocessing with Glue
        start_glue_job = sfn.Task(
            self, "execute_glue",
            task=sfn_tasks.InvokeFunction(lambdaFn2),
        )

        wait_state = sfn.Wait(
            self, "Wait 15 seconds",
            time=sfn.WaitTime.duration(cdk.Duration.seconds(15)),
        )

        get_status = sfn.Task(
            self, "Get Job Status",
            task=sfn_tasks.InvokeFunction(lambdaFn3),
        )

        is_complete = sfn.Choice(
            self, "Job Complete?"
        )
        job_failed = sfn.Fail(
            self, "Job Failed",
            cause="AWS Job Failed",
            error="DescribeJob returned FAILED"
        )

        job_succeeded = sfn.Succeed(
            self, "succeed_state"
        )

        definition = start_glue_job.next(wait_state)\
            .next(get_status)\
            .next(is_complete
                    .when(sfn.Condition.string_equals(
                      "$.jobDetails.jobStatus", "FAILED"), job_failed)
                    .when(sfn.Condition.string_equals(
                      "$.jobDetails.jobStatus", "SUCCEEDED"), job_succeeded)
                    .otherwise(wait_state))
        
        state_machine = sfn.StateMachine(
            self, "Preprocessing",
            definition=definition,
        )

        # Create a SQS for callbackstep in a SageMaker Pipelines

        # Create SageMaker Pipeline
        sagemaker_execution_role = aws_iam.Role.from_role_name(
            self, "SageMakerExecutionRole", role_name=execution_role_arn.value_as_string
        )

        callback_queue = sqs.Queue(
            self, "pipeline_callbacks_glue_prep",
            visibility_timeout=cdk.Duration.minutes(10)
        )

        callback_queue.grant_send_messages(sagemaker_execution_role)

        lambdaFn1.add_event_source(
            lambda_event_sources.SqsEventSource(callback_queue)
        )

        # Create a glue job for preprocessing
        glue_job = glue.Job(
            self,
            "sagemaker-pipeline-GlueJob",
            job_name="sagemaker-pipeline-GlueJob",
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
            worker_count=2,
            worker_type=glue.WorkerType.STANDARD,
            max_concurrent_runs=1,
            timeout=cdk.Duration.minutes(60),
        )

        #glue_job.add_to_role_policy(aws_iam.PolicyStatement(
        #    actions = ['sagemaker:SendPipelineExecutionStepFailure',],
        #    resources = [f'arn:aws:sagemaker:{my_region}:{my_acc_id}:pipeline/*',]
        #    ))

        cdk.CfnOutput(
            self,
            "SqsURL",
            value=callback_queue.queue_url,
            export_name="SM-Pipeline-SQS-URL",
        )

        cdk.CfnOutput(
            self,
            "StatesArn",
            value=state_machine.state_machine_arn,
            export_name="SM-Pipeline-States-ARN",
        )

        cdk.CfnOutput(
            self,
            "GlueJobName",
            value=glue_job.job_name,
            export_name="SM-Pipeline-Glue-Name",
        )
