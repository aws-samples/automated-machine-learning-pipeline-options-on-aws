from aws_cdk import (
    # Duration,
    core as cdk,
    aws_stepfunctions as sfn,
    aws_glue as glue,
    aws_ec2 as ec2,
    aws_s3 as s3,
    aws_stepfunctions_tasks as sfn_tasks,
    aws_iam,
)
from constructs import Construct

import sagemaker

import boto3 
my_region = boto3.session.Session().region_name
my_acc_id = boto3.client('sts').get_caller_identity().get('Account')

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

        train_instance_type = cdk.CfnParameter(
            self,
            "TrainInstanceType",
            type="String",
            description="Train Instance Type",
            min_length=3,
        )

        artifact_bucket = s3.Bucket.from_bucket_name(
            self,
            "ArtifactBucket",
            bucket_name.value_as_string
        )

        glue_role = aws_iam.Role(self, "Role",
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
        processed_dir = f"s3://{bucket_name.value_as_string}/{prefix.value_as_string}/processed"

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
                    '--INPUT_DIR': input_dir,
                    '--PROCESSED_DIR': processed_dir
                }
            ),
        )

        image_uri = sagemaker.image_uris.retrieve(
            framework="xgboost",
            region=my_region,
            version="1.0-1",
            py_version="py3",
        )

        train_task = sfn_tasks.SageMakerCreateTrainingJob(self, "TrainSagemaker",
            training_job_name="StepFunctionsDataScienceSDKTraining",
            algorithm_specification=sfn_tasks.AlgorithmSpecification(
                training_input_mode=sfn_tasks.InputMode.FILE,
                training_image=image_uri,

            ),
            input_data_config=[
                sfn_tasks.Channel(
                    channel_name="train",
                    data_source=sfn_tasks.DataSource(
                        s3_data_source=sfn_tasks.S3DataSource(
                            s3_data_type=sfn_tasks.S3DataType.S3_PREFIX,
                            s3_location=sfn_tasks.S3Location.from_bucket(
                                artifact_bucket,
                                f"{prefix.value_as_string}/processed/train"
                            )
                        )
                    )
                ),
                sfn_tasks.Channel(
                    channel_name="validation",
                    data_source=sfn_tasks.DataSource(
                        s3_data_source=sfn_tasks.S3DataSource(
                            s3_data_type=sfn_tasks.S3DataType.S3_PREFIX,
                            s3_location=sfn_tasks.S3Location.from_bucket(
                                artifact_bucket,
                                f"{prefix.value_as_string}/processed/validation"
                            )
                        )
                    )
                ),
            ],
            output_data_config=sfn_tasks.OutputDataConfig(
                s3_output_location=sfn_tasks.S3Location.from_bucket(
                    artifact_bucket,
                    f"{prefix.value_as_string}/model"
                )
            ),
            resource_config=sfn_tasks.ResourceConfig(
                instance_count=1,
                instance_type=ec2.InstanceType(train_instance_type.value_as_string),
                volume_size=cdk.Size.gibibytes(50)
            ),  # optional: default is 1 instance of EC2 `M4.XLarge` with `10GB` volume
            stopping_condition=sfn_tasks.StoppingCondition(
                max_runtime=cdk.Duration.hours(2)
            ),
            hyperparameters={
                "max_depth":5,
                "eta":0.2,
                "gamma":4,
                "min_child_weight":6,
                "subsample":0.8,
                "silent":0,
                "objective": "binary:logistic",
                "num_round": 100,
                "eval_metric": "auc"
            }
        )

        definition = start_glue_job.next(
            train_task
        )
        
        state_machine = sfn.StateMachine(
            self, "STFPipeline",
            definition=definition,
        )


