import boto3
import logging

sm_client = boto3.client('sagemaker')

def lambda_handler(event, context):
    training_name = event["TrainingJobName"]
    training_desc = sm_client.describe_training_job(TrainingJobName=training_name)
    model_uri = training_desc['ModelArtifacts']['S3ModelArtifacts']

    image_uri = event["ImageUri"]
    model_package_group_name = event["ModelPackageGroupName"]
    
    response = sm_client.list_model_package_groups()
    model_groups = []
    for model_package_group in response["ModelPackageGroupSummaryList"]:
        model_groups.append(model_package_group["ModelPackageGroupName"])
        
    if model_package_group_name not in model_groups:
        sm_client.create_model_package_group(ModelPackageGroupName=model_package_group_name,
                                            ModelPackageGroupDescription="Churn Prediction Prediction")
    
    eval_res = event["EvaluationResult"]
    eval_s3uri = f"{eval_res}evaluation.json"
    
    modelpackage_inference_specification =  {
        "InferenceSpecification": {
          "Containers": [
             {
                "Image": image_uri,
                 "ModelDataUrl": model_uri
             }
          ],
          "SupportedContentTypes": [ "text/csv" ],
          "SupportedResponseMIMETypes": [ "text/csv" ],
       },
       "ModelMetrics": {
           "ModelQuality": {
               "Statistics": {
                   "S3Uri": eval_s3uri,
                   "ContentType": "application/json"
               },
               
           }
       }
     }

    try:
        create_model_package_input_dict = {
            "ModelPackageGroupName" : model_package_group_name,
            "ModelPackageDescription" : "Custom churn prediction",
            "ModelApprovalStatus" : "Approved"
        }
        create_model_package_input_dict.update(modelpackage_inference_specification)

        create_model_package_response = sm_client.create_model_package(**create_model_package_input_dict)
        model_package_arn = create_model_package_response["ModelPackageArn"]

        return {
            "statusCode": 200,
            "modelPackageArn": model_package_arn,
            "ModelArtifacts": {
                "S3ModelArtifacts": model_uri
            },
            "TrainingJobName": training_name
        }
    except Exception as e:
        return {"statusCode": 400, "Error": f"model registery failed! {e}"}