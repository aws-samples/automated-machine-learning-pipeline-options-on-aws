from aws_cdk import (
    # Duration,
    core as cdk,
    aws_stepfunctions as sfn,
    aws_lambda as lambda_,
    aws_glue as glue,
    aws_ec2 as ec2,
    aws_s3 as s3,
    aws_stepfunctions_tasks as sfn_tasks,
    aws_s3_deployment as s3deploy,
    aws_iam,
)
from constructs import Construct

import sagemaker

import boto3 
my_region = boto3.session.Session().region_name
my_acc_id = boto3.client('sts').get_caller_identity().get('Account')
resource_s3 = boto3.resource("s3")

class CfnStack(cdk.Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        
        bucket_name = cdk.CfnParameter(
            self,
            "BucketName",
            type="String",
            description="Bucket for project",
            min_length=3,
        )

        prefix = cdk.CfnParameter(
            self,
            "Prefix",
            type="String",
            description="Prefix for project",
            min_length=3,
        )

        artifact_bucket = s3.Bucket.from_bucket_name(
            self,
            "ArtifactBucket",
            bucket_name.value_as_string
        )

        glue_role = aws_iam.Role(self, "GlueRole",
            assumed_by=aws_iam.ServicePrincipal("glue.amazonaws.com"),
            description="Glue role"
        )

        glue_role.add_to_policy(
            aws_iam.PolicyStatement(
                actions = ['s3:ListBucket'],
                resources = [f'arn:aws:s3:::{bucket_name.value_as_string}',]
            )
        )

        glue_role.add_to_policy(
            aws_iam.PolicyStatement(
                actions = ['s3:*Object'],
                resources = [f'arn:aws:s3:::{bucket_name.value_as_string}/*',]
            )
        )

        # Create a glue job for preprocessing
        glue_job = glue.Job(
            self,
            "stepdfunctions-datascience-GlueJob",
            job_name="stepdfunctions-datascience-GlueJob",
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
            max_concurrent_runs=2,
            timeout=cdk.Duration.minutes(60),
        )

        input_dir = f"s3://{bucket_name.value_as_string}/{prefix.value_as_string}/input"
        train_dir = f"s3://{bucket_name.value_as_string}/{prefix.value_as_string}/processed/train"
        val_dir = f"s3://{bucket_name.value_as_string}/{prefix.value_as_string}/processed/val"
        test_dir = f"s3://{bucket_name.value_as_string}/{prefix.value_as_string}/processed/test"

        # STEP FUNCTION
        start_glue_job = sfn_tasks.GlueStartJobRun(
            self,
            "StartGlueJobTask",
            glue_job_name=glue_job.job_name,
            integration_pattern=sfn.IntegrationPattern.RUN_JOB,
            result_path="$.glueTaskResult",
            arguments=sfn.TaskInput.from_object(
                {
                    '--job-bookmark-option': 'job-bookmark-enable',
                    '--additional-python-modules': 'pyarrow==2,awswrangler==2.9.0,fsspec==0.7.4',
                    # Custom arguments below
                    '--INPUT_DIR': input_dir,
                    '--TRAIN_DIR': train_dir,
                    '--VAL_DIR': val_dir,
                    '--TEST_DIR': test_dir,
                }
            ),
            result_selector={
                "train_dir": train_dir,
                "val_dir": val_dir,
                "test_dir": test_dir
            }
        )

        image_uri = sagemaker.image_uris.retrieve(
            framework="xgboost",
            region=my_region,
            version="1.0-1",
            py_version="py3",
        )

        # TrainingJob
        model_prefix = f"{prefix.value_as_string}/model"
        train_task = sfn_tasks.SageMakerCreateTrainingJob(self, "TrainSagemaker",
            training_job_name=sfn.JsonPath.string_at("$.RunJobName"),
            algorithm_specification=sfn_tasks.AlgorithmSpecification(
                training_input_mode=sfn_tasks.InputMode.FILE,
                training_image=sfn_tasks.DockerImage.from_registry(image_uri),
            ),
            integration_pattern=sfn.IntegrationPattern.RUN_JOB,
            result_path="$.trainTaskResult",
            input_data_config=[
                sfn_tasks.Channel(
                    channel_name="train",
                    data_source=sfn_tasks.DataSource(
                        s3_data_source=sfn_tasks.S3DataSource(
                            s3_data_type=sfn_tasks.S3DataType.S3_PREFIX,
                            s3_location=sfn_tasks.S3Location.from_json_expression("$.glueTaskResult.train_dir")
                        )
                    ),
                    content_type="text/csv"
                ),
                sfn_tasks.Channel(
                    channel_name="validation",
                    data_source=sfn_tasks.DataSource(
                        s3_data_source=sfn_tasks.S3DataSource(
                            s3_data_type=sfn_tasks.S3DataType.S3_PREFIX,
                            s3_location=sfn_tasks.S3Location.from_json_expression("$.glueTaskResult.val_dir")
                        )
                    ),
                    content_type="text/csv"
                ),
            ],
            output_data_config=sfn_tasks.OutputDataConfig(
                s3_output_location=sfn_tasks.S3Location.from_bucket(
                    artifact_bucket,
                    model_prefix
                )
            ),
            resource_config=sfn_tasks.ResourceConfig(
                instance_count=1,
                instance_type=ec2.InstanceType(sfn.JsonPath.string_at("$.TrainInstanceType")),
                volume_size=cdk.Size.gibibytes(50)
            ),  # optional: default is 1 instance of EC2 `M4.XLarge` with `10GB` volume
            stopping_condition=sfn_tasks.StoppingCondition(
                max_runtime=cdk.Duration.hours(2)
            ),
            hyperparameters={
                "max_depth": sfn.JsonPath.string_at("$.hyperparameters.max_depth"),
                "eta": sfn.JsonPath.string_at("$.hyperparameters.eta"),
                "gamma": sfn.JsonPath.string_at("$.hyperparameters.gamma"),
                "min_child_weight": sfn.JsonPath.string_at("$.hyperparameters.min_child_weight"),
                "subsample": sfn.JsonPath.string_at("$.hyperparameters.subsample"),
                "silent": sfn.JsonPath.string_at("$.hyperparameters.silent"),
                "objective": sfn.JsonPath.string_at("$.hyperparameters.objective"),
                "num_round": sfn.JsonPath.string_at("$.hyperparameters.num_round"),
                "eval_metric": sfn.JsonPath.string_at("$.hyperparameters.eval_metric")
            }
        )

        # Create evaluation step
        code_key = f'{prefix.value_as_string}/code'
        deployment = s3deploy.BucketDeployment(self, 'DeployWebsite',
            sources=[s3deploy.Source.asset("./code")],
            destination_bucket=artifact_bucket,
            destination_key_prefix=f"{prefix.value_as_string}/code"
        )

        sm_role = train_task.role
        sm_role.add_to_policy(
            aws_iam.PolicyStatement(
                actions = ['s3:ListBucket', 's3:*Object'],
                resources = [
                    f'arn:aws:s3:::{bucket_name.value_as_string}',
                    f'arn:aws:s3:::{bucket_name.value_as_string}/*',
                ]
            )
        )

        eval_image_uri = sagemaker.image_uris.retrieve(
            framework="sklearn",
            region=my_region,
            version="0.20.0",
            py_version="py3",
        )

        output_evaluation_s3_uri = f"s3://{bucket_name.value_as_string}/{prefix.value_as_string}/evaluation/"
        run_evaluation = sfn_tasks.CallAwsService(
            self,
            "ModelEvaluation",
            iam_resources=[f'arn:aws:sagemaker:{my_region}:{my_acc_id}:processing-job/*'],
            service="sagemaker",
            action="createProcessingJob",
            result_path="$.taskResult",
            # integration_pattern=sfn.IntegrationPattern.WAIT_FOR_TASK_TOKEN, # RUN_JOB is not supported for AWS SDK service integration currently (2022/10/03)
            parameters={
                "ProcessingInputs": [
                    {
                        "InputName": "test-data",
                        "S3Input": {
                            "S3Uri": f"{test_dir}/",
                            "LocalPath":"/opt/ml/processing/test",
                            "S3DataType": "S3Prefix",
                            "S3InputMode": "File"
                        }
                    },
                    {
                        "InputName": "code",
                        "S3Input": {
                            "S3Uri": f"s3://{bucket_name.value_as_string}/{code_key}/evaluation.py",
                            "LocalPath":"/opt/ml/processing/input/code",
                            "S3DataType": "S3Prefix",
                            "S3InputMode": "File"
                        }
                    }
                ],
                "ProcessingOutputConfig": {
                    "Outputs": [
                        {
                            "OutputName": "evaluation",
                            "S3Output": {
                                "S3Uri": output_evaluation_s3_uri,
                                "LocalPath": "/opt/ml/processing/evaluation",
                                "S3UploadMode": "EndOfJob"
                            }
                        }
                    ]
                },
                "ProcessingJobName": sfn.JsonPath.string_at("$.RunJobName"),
                "ProcessingResources": {
                    "ClusterConfig": {
                        "InstanceCount": 1,
                        "InstanceType": "ml.m5.xlarge",
                        "VolumeSizeInGB": 20
                    }
                },
                "StoppingCondition": {
                    "MaxRuntimeInSeconds": 1200
                },
                "AppSpecification": {
                    "ImageUri": eval_image_uri,
                    "ContainerEntrypoint": ["python3", "/opt/ml/processing/input/code/evaluation.py"]
                },
                "RoleArn": sm_role.role_arn,
                "Environment": {
                    "model_url": sfn_tasks.S3Location.from_json_expression("$.trainTaskResult.ModelArtifacts.S3ModelArtifacts")
                }
            }
        )

        wait_state = sfn.Wait(
            self, "Wait 15 seconds",
            time=sfn.WaitTime.duration(cdk.Duration.seconds(15)),
        )

        # Create a function for checking the status of Glue job
        with open("code/check_processing_job.py", encoding="utf8") as fp:
            lambda_check_glue_code = fp.read()

        check_processing_lambda = lambda_.Function(
            self,
            "check_glue_job_function",
            code=lambda_.InlineCode(lambda_check_glue_code),
            handler="index.lambda_handler",
            timeout=cdk.Duration.seconds(300),
            runtime=lambda_.Runtime.PYTHON_3_8
        )

        # Add perms
        check_processing_lambda.add_to_role_policy(aws_iam.PolicyStatement(
            actions = ['sagemaker:DescribeProcessingJob',],
            resources = [f'arn:aws:sagemaker:{my_region}:{my_acc_id}:processing-job/*',]
        ))

        get_status = sfn.Task(
            self, "Get Processing Job Status",
            task=sfn_tasks.InvokeFunction(check_processing_lambda),
        )

        is_evaluation_complete = sfn.Choice(
            self, "Evaluation Complete?"
        )

        job_failed = sfn.Fail(
            self, "Evaluation failed",
            cause="AWS Job Failed",
            error="DescribeJob returned FAILED"
        )

        # Query evaluation result
        with open("code/query_evaluation_result.py", encoding="utf8") as fp:
            lambda_query_eval_code = fp.read()

        query_eval_lambda = lambda_.Function(
            self,
            "query_evaluation_function",
            code=lambda_.InlineCode(lambda_query_eval_code),
            handler="index.lambda_handler",
            timeout=cdk.Duration.seconds(300),
            runtime=lambda_.Runtime.PYTHON_3_8
        )

        # Add perms
        query_eval_lambda.add_to_role_policy(aws_iam.PolicyStatement(
            actions = ['s3:ListBucket', 's3:*Object'],
            resources = [
                f'arn:aws:s3:::{bucket_name.value_as_string}',
                f'arn:aws:s3:::{bucket_name.value_as_string}/*',
            ]
        ))

        query_eval_task = sfn.Task(
            self, "Query evaluation result",
            task=sfn_tasks.InvokeFunction(
                query_eval_lambda,
                payload={
                    "EvaluationResult": output_evaluation_s3_uri,
                    "RunJobName": sfn.JsonPath.string_at("$.RunJobName"),
                    "trainTaskResult": sfn.JsonPath.string_at("$.trainTaskResult")
                }
            )
        )

        check_evaluation = sfn.Choice(
            self, "Greater than metric?"
        )

        accuracy_fail_step = sfn.Fail(
            self, "Model Accuracy Too Low",
            cause="Validation accuracy lower than threshold",
            error="Model Accuracy Too Low"
        )

        # Registry model
        with open("code/register_model.py", encoding="utf8") as fp:
            lambda_registry_model_code = fp.read()

        registry_model_lambda = lambda_.Function(
            self,
            "register_model_function",
            code=lambda_.InlineCode(lambda_registry_model_code),
            handler="index.lambda_handler",
            timeout=cdk.Duration.seconds(300),
            runtime=lambda_.Runtime.PYTHON_3_8
        )

        # Add perms
        registry_model_lambda.add_to_role_policy(aws_iam.PolicyStatement(
            actions = [
                'sagemaker:ListModelPackageGroups'
            ],
            resources = ["*"]
        ))

        registry_model_lambda.add_to_role_policy(aws_iam.PolicyStatement(
            actions = [
                'sagemaker:CreateModelPackageGroup'
            ],
            resources = [
                f"arn:aws:sagemaker:{my_region}:{my_acc_id}:model-package-group/*"
            ]
        ))

        registry_model_lambda.add_to_role_policy(aws_iam.PolicyStatement(
            actions = [
                'sagemaker:CreateModelPackage',
            ],
            resources = [
                f"arn:aws:sagemaker:{my_region}:{my_acc_id}:model-package/*",
            ]
        ))

        registry_model_lambda.add_to_role_policy(aws_iam.PolicyStatement(
            actions = ['s3:ListBucket', 's3:*Object'],
            resources = [
                f'arn:aws:s3:::{bucket_name.value_as_string}',
                f'arn:aws:s3:::{bucket_name.value_as_string}/*',
            ]
        ))

        registry_model_lambda.add_to_role_policy(aws_iam.PolicyStatement(
            actions = ['ecr:*'],
            resources = [
                "*"
            ]
        ))

        registry_model_lambda.add_to_role_policy(aws_iam.PolicyStatement(
            actions = [
                'sagemaker:DescribeTrainingJob'
            ],
            resources = [
                f"arn:aws:sagemaker:{my_region}:{my_acc_id}:training-job/*"
            ]
        ))

        register_model_task = sfn.Task(
            self, "Register model",
            task=sfn_tasks.InvokeFunction(
                registry_model_lambda,
                payload={
                    "EvaluationResult": output_evaluation_s3_uri,
                    "ImageUri": image_uri,
                    "TrainingJobName": sfn.JsonPath.string_at("$.trainTaskResult.TrainingJobName"),
                    "ModelPackageGroupName": sfn.JsonPath.string_at("$.RunJobName")
                }
            )
        )
#Create a model
    
        create_model_task = sfn_tasks.SageMakerCreateModel(self, "CreateModel",
         	    model_name=sfn.JsonPath.string_at("$.TrainingJobName"),
           	    primary_container=sfn_tasks.ContainerDefinition(
               	    image=sfn_tasks.DockerImage.from_registry(image_uri),
                    mode=sfn_tasks.Mode.SINGLE_MODEL,
                    model_s3_location=sfn_tasks.S3Location.from_json_expression("$.ModelArtifacts.S3ModelArtifacts")
                ),
                result_path="$.taskResult"
            )
        
#endpoint configuration
        endpoint_configuration_task= sfn_tasks.SageMakerCreateEndpointConfig(self, "SagemakerEndpointConfig",
	           	endpoint_config_name=sfn.JsonPath.string_at("$.TrainingJobName"),
                production_variants=[
                    sfn_tasks.ProductionVariant(
                        initial_instance_count=1,
                        instance_type=ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.XLARGE),
                        model_name=sfn.JsonPath.string_at("$.TrainingJobName"),
                        variant_name="test-variant"
                    )
                ],
                result_path="$.ConfTaskResult"
        )
        
#create endpoint
        endpoint_creation_task= sfn_tasks.SageMakerCreateEndpoint(self, "SagemakerEndpoint",
	     	endpoint_name=sfn.JsonPath.string_at("$.TrainingJobName"),
        	endpoint_config_name=sfn.JsonPath.string_at("$.TrainingJobName")
        )

        definition = start_glue_job.next(
            train_task
        ).next(
            run_evaluation
        ).next(
            wait_state
        ).next(
            get_status
        ).next(
            is_evaluation_complete.when(
                    sfn.Condition.string_equals("$.ProcessingJobStatus", "Failed"), job_failed
                ).when(
                    sfn.Condition.string_equals("$.ProcessingJobStatus", "Completed"), query_eval_task
                    .next(
                        check_evaluation.when(
                             sfn.Condition.number_greater_than_equals("$.trainingMetrics", 0.9), (register_model_task
                                                                                                    .next(create_model_task
                                                                                                ).next(endpoint_configuration_task
                                                                                                ).next(endpoint_creation_task))
                        ).otherwise(
                            accuracy_fail_step
                        )
                    )
                ).when(
                    sfn.Condition.string_equals("$.ProcessingJobStatus", "Stopped"), job_failed
                ).otherwise(
                    wait_state
                )
        )
        
        state_machine = sfn.StateMachine(
            self, "STFPipeline",
            definition=definition,
        )
        state_machine.add_to_role_policy(
            aws_iam.PolicyStatement(
                actions = ['sagemaker:CreateTrainingJob'],
                resources = [
                    f'arn:aws:sagemaker:{my_region}:{my_acc_id}:training-job/*',
                ]
            )
        )

        cdk.CfnOutput(
            self,
            "sfn-pipeline-arn",
            value=state_machine.state_machine_arn,
            export_name="sfn-pipeline-arn",
        )

