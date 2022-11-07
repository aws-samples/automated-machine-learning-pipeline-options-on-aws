## A demonstration of three different ways to build automated Machine learning workflows on AWS
### Introduction
A typical machine learning (ML) workflow involves a number of steps. Typically, it can involve steps such as data processing, feature engineering, model training and evaluation, as well as model deployment. Some more complex workflows can include additional steps such as data and model monitoring, checking for bias in various stages of the ML lifecycle and registering the model into a model registry. 

There are various orchestrator tools available on AWS and so there are various ways you can think of orchestrating your ML workflows. Often, you would want to know which orchestration tools will fit your use case the best and which to decide on using when creating your ML workflows.

In this github you can learn about three orchestration tools that you can use to design automated machine learning workflows on AWS as described in the following. Our goal is to show you how you use each of these orchestration tools native to AWS to create a ML workflow and discuss the advantages and limitations of each, as well as which of the tools is most likely the best choice for a given use case. The three approaches discussed are:

- Amazon SageMaker Pipelines:  [Amazon SageMaker Pipelines](https://aws.amazon.com/sagemaker/pipelines/) is the first purpose-built, easy-to-use continuous integration and continuous delivery (CI/CD) service for machine learning (ML). With SageMaker Pipelines, you can create, automate, and manage end-to-end ML workflows at scale. Since it is purpose-built for machine learning, SageMaker Pipelines helps you automate different steps of the ML workflow, including data loading, data transformation, training and tuning, and deployment. With SageMaker Pipelines, you can build ML models, manage massive volumes of data, thousands of training experiments, and hundreds of different model versions. You can share and re-use workflows to recreate or optimize models, helping you scale ML throughout your organization.
- AWS Step Functions: [AWS Step Functions](https://aws.amazon.com/step-functions/) is a serverless orchestration service that allows you to build resilient workflows using AWS services such as Amazon SageMaker, AWS Glue, and AWS Lambda. The AWS Step Functions Data Science SDK is an open-source library that allows you to easily create workflows that pre-process data and then train and publish machine learning models using Amazon SageMaker and AWS Step Functions. You can create machine learning workflows in Python that orchestrate AWS infrastructure at scale, without having to provision and integrate the AWS services separately. 
- AWS Glue workflows: [AWS Glue workflows](https://docs.aws.amazon.com/glue/latest/dg/orchestrate-using-workflows.html) provide a visual and programmatic tool to author data pipelines by combining AWS Glue crawlers for schema discovery and AWS Glue Spark and Python shell jobs to transform the data. A workflow consists of one or more task nodes arranged as a graph. Relationships can be defined and parameters passed between task nodes to enable you to build pipelines of varying complexity. You can trigger workflows on a schedule or on-demand. You can track the progress of each node independently or the entire workflow, making it easier to troubleshoot your workflow.

Please note If you use Kubeflow or Airflow there are specific managed tools available that can integrate with the SageMaker to create ML workflows. These options are not shown in this repo, but you can learn more about these options on [SageMaker Components for Kubeflow](https://aws.amazon.com/blogs/machine-learning/introducing-amazon-sagemaker-components-for-kubeflow-pipelines/) and building workflows using [Amazon SageMaker and Airflow](https://aws.amazon.com/blogs/machine-learning/build-end-to-end-machine-learning-workflows-with-amazon-sagemaker-and-apache-airflow/).

### Solution Overview

To demonstrate various workflow options discussed above, we will design a workflow with multiple steps including data processing, training, evaluating and deploying a classification model that predicts customer churn. The data for this use case is a synthetic dataset originally from the University of California Irvine Repository of Machine Learning Datasets, that we have further processed to use in our use case.  The processed data is available in the code repository [here]([https://github.com/aws-samples/automated-machine-learning-pipeline-options-on-aws/tree/main/data). If you are interested to learn more about the data set and details of the processing, please refer to this [GitHub](https://github.com/aws-samples/real-time-churn-prediction-with-amazon-connect-and-amazon-sagemaker) . 

At the high level our workflow consists of the following steps. A processing step that uses a PySpark  job ran in AWS Glue. For the purpose of this demo, we assume your processing job uses Spark and you would like to utilize Glue for the processing step. Our goal is to also show you how you can integrate a Glue step in SageMaker Pipelines using a call back step. Please note, that you can also implement the processing step using SageMaker processing capabilities such as  SageMaker  Processing Jobs as used in this [blog post](https://aws.amazon.com/blogs/machine-learning/building-automating-managing-and-scaling-ml-workflows-using-amazon-sagemaker-pipelines/) or using [SageMaker Data Wrangler](https://docs.aws.amazon.com/sagemaker/latest/dg/data-wrangler.html). 
Second step, is a  training step, that uses XGBoost one of the SageMaker [built-in algorithms](https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html) to run a training job. 
Third step is an evaluation step that uses SageMaker processing job to fit the trained model from the previous step and evaluate its performance based on the defined metrics. We use the evaluation,  to first evaluate the model performance before registering that model.

Next step is a conditional step that embeds three steps if the model performance from previous step met our criteria; a create model step that creates a model package used for model registration and deployment, a registration step that registers the model in SageMaker Model Registry and finally a deploy step that hosts the model on a real time endpoint. If the conditional step didn’t meet our defined criteria the pipeline will stop.

These steps are shown in the following graph: 
```
![Machine Learning workflow](https://github.com/aws-samples/automated-machine-learning-pipeline-options-on-aws/images/img1.png)
```
You can find each of the workflow detailed description in their relevant folder.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
