
# Deploy a machine learning pipeline with AWS Step Functions Data Science SDK

## CDK development environment setting
The `cdk.json` file tells the CDK Toolkit how to execute your app.

This project is set up like a standard Python project.  The initialization
process also creates a virtualenv within this project, stored under the `.venv`
directory.  To create the virtualenv it assumes that there is a `python3`
(or `python` for Windows) executable in your path with access to the `venv`
package. If for any reason the automatic creation of the virtualenv fails,
you can create the virtualenv manually.

To manually create a virtualenv on MacOS and Linux:

```
$ python3 -m venv .venv
```

After the init process completes and the virtualenv is created, you can use the following
step to activate your virtualenv.

```
$ source .venv/bin/activate
```

If you are a Windows platform, you would activate the virtualenv like this:

```
% .venv\Scripts\activate.bat
```

Once the virtualenv is activated, you can install the required dependencies.

```
$ pip install -r requirements.txt
```

## Deploy a Step Function Pipeline
At this point you can now synthesize the CloudFormation template for this code.

```
$ cdk synth
```

To add additional dependencies, for example other CDK libraries, just add
them to your `setup.py` file and rerun the `pip install -r requirements.txt`
command.

We define 2 parameters in the CDK code, and deploy CDK application with the following code.
- `bucket_name` is the existing S3 Bucket in your account and region should be the same region as your CDK environment.
- `prefix` is the existing directory in `bucket_name`

```
cdk deploy --StepFunctionsDataScienceStack --parameters BucketName={bucket_name} --parameters Prefix={prefix}
```
We can see the arn of Step Functions we create in the outputs followed by the above command.

## Data preparation
Once you succeed to deploy Step Functions pipe, upload the sample data to the S3 Bucket (`bucket_name` and `prefix` are same as we used in `cdk deploy`).
```bash
aws s3 cp {project_root}/data/churn_processed.csv s3://{bucket_name}/{prefix}/input/churn_processed.csv
```

## Run Step Functions Pipeline
In Step Functions console, we select the Step Functions we created and create a execution with the input parameters as below. Note that we need to use different `RunJobName` each time when executing the Step Function Pipelines.
```json
{
    "TrainInstanceType": "ml.m5.xlarge",
    "RunJobName": "stepfunctionsTraining148",
    "hyperparameters": {
        "max_depth": "5",
        "eta": "0.2",
        "gamma": "4",
        "min_child_weight": "6",
        "subsample": "0.8",
        "silent": "0",
        "objective": "binary:logistic",
        "num_round": "100",
        "eval_metric": "auc"
    }
}
```

## Check the real-time inference endpoint
Once the pipeline is finished, you can check the real-time inference endpoint created by the step function pipeline in SageMaker console.

---
## Useful commands

 * `cdk ls`          list all stacks in the app
 * `cdk synth`       emits the synthesized CloudFormation template
 * `cdk deploy`      deploy this stack to your default AWS account/region
 * `cdk diff`        compare deployed stack with current state
 * `cdk docs`        open CDK documentation

