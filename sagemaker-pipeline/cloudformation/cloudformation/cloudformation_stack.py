from aws_cdk import (
    Stack,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as sfn_tasks,
    aws_lambda as lambda_,
    aws_iam,
    core
)

from constructs import Construct

import boto3 
my_region = boto3.session.Session().region_name
my_acc_id = boto3.client('sts').get_caller_identity().get('Account')

class CloudformationStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        with open("lambda/execute_function.py", encoding="utf8") as fp:
            lambda_exec_func_code = fp.read()

        lambdaFn1 = lambda_.Function(
            self,
            "execute_sfn_function",
            code=lambda_.InlineCode(lambda_exec_func_code),
            handler="execute_function.lambda_handler",
            timeout=core.Duration.seconds(300),
            runtime=lambda_.Runtime.PYTHON_3_8
        )

        # Add perms
        lambdaFn1.add_to_role_policy(aws_iam.PolicyStatement(
            actions = ['states:StartExecution',],
            resources = [f'arn:aws:states:{my_region}:{my_acc_id}:execution:*',]
            ))
        
        lambdaFn1.add_to_role_policy(aws_iam.PolicyStatement(
            actions = ['sagemaker:SendPipelineExecutionStepFailure',],
            resources = [f'arn:aws:sagemaker:{my_region}:{my_acc_id}:pipeline-execution:*',]
            ))

        ## Start Glue Job
        with open("lambda/execute_glue_job.py", encoding="utf8") as fp:
            lambda_exec_glue_code = fp.read()
        
        lambdaFn2 = lambda_.Function(
            self,
            "execute_glue_job_function",
            code=lambda_.InlineCode(lambda_exec_glue_code),
            handler="execute_glue_job.lambda_handler",
            timeout=core.Duration.seconds(300),
            runtime=lambda_.Runtime.PYTHON_3_8
        )

        # Add perms
        lambdaFn2.add_to_role_policy(aws_iam.PolicyStatement(
            actions = ['glue:StartJobRun',],
            resources = [f'arn:aws:glue:{my_region}:{my_acc_id}:job:*',]
            ))
        
        lambdaFn2.add_to_role_policy(aws_iam.PolicyStatement(
            actions = ['sagemaker:SendPipelineExecutionStepFailure',],
            resources = [f'arn:aws:sagemaker:{my_region}:{my_acc_id}:pipeline-execution:*',]
            ))

        # Create a function for checking the status of Glue job
        with open("lambda/check_glue_job.py", encoding="utf8") as fp:
            lambda_check_glue_code = fp.read()
        
        lambdaFn3 = lambda_.Function(
            self,
            "check_glue_job_function",
            code=lambda_.InlineCode(lambda_check_glue_code),
            handler="check_glue_job.lambda_handler",
            timeout=core.Duration.seconds(300),
            runtime=lambda_.Runtime.PYTHON_3_8
        )

        # Add perms
        lambdaFn3.add_to_role_policy(aws_iam.PolicyStatement(
            actions = ['glue:GetJobRun',],
            resources = [f'arn:aws:glue:{my_region}:{my_acc_id}:job:*',]
            ))
        
        lambdaFn3.add_to_role_policy(aws_iam.PolicyStatement(
            actions = ['sagemaker:SendPipelineExecutionStepFailure',],
            resources = [f'arn:aws:sagemaker:{my_region}:{my_acc_id}:pipeline-execution:*',]
            ))
        
        lambdaFn3.add_to_role_policy(aws_iam.PolicyStatement(
            actions = ['sagemaker:SendPipelineExecutionStepSuccess',],
            resources = [f'arn:aws:sagemaker:{my_region}:{my_acc_id}:pipeline-execution:*',]
            ))

        # Create a step functions for preprocessing with Glue
        start_glue_job = sfn.Task(
            self, "execute_glue",
            task=sfn_tasks.InvokeFunction(lambdaFn2),
        )

        wait_state = sfn.Wait(
            self, "Wait 15 seconds",
            time=sfn.WaitTime.duration(core.Duration.seconds(15)),
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
            .next(wait_state)\
            .next(get_status)\
            .next(is_complete
                    .when(sfn.Condition.string_equals(
                      "$.jobDetails.jobStatus", "FAILED"), job_failed)
                    .when(sfn.Condition.string_equals(
                      "$.jobDetails.jobStatus", "SUCCEEDED"), job_succeeded)
                    .otherwise(wait_state))
        
        sfn.StateMachine(
            self, "Preprocessing",
            definition=definition,
        )