import boto3
import logging

sm_client = boto3.client('sagemaker')
s3_client = boto3.client('s3')

def lambda_handler(event, context):
    model_uri = event["S3ModelArtifacts"]
    image_uri = event["ImageUri"]
    model_package_group_name = event["ModelPackageGroupName"]
    
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

    # Alternatively, you can specify the model source like this:
    # modelpackage_inference_specification["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]=model_url
    try:
        create_model_package_input_dict = {
            "ModelPackageGroupName" : model_package_group_name,
            "ModelPackageDescription" : "Custom churn prediction",
            "ModelApprovalStatus" : "PendingManualApproval"
        }
        create_model_package_input_dict.update(modelpackage_inference_specification)

        create_model_package_response = sm_client.create_model_package(**create_model_package_input_dict)
        model_package_arn = create_model_package_response["ModelPackageArn"]

        return {"statusCode": 200, "modelPackageArn": model_package_arn}
    except Exception as e:
        return {"statusCode": 400, "Error": f"model registery failed! {e}"}
    #return {"statusCode": 200, "message": "OK"}