# MLFlow_Basics

## For Dagshub
MLFLOW_TRACKING_URI=https://dagshub.com/faridkhan5/MLFlow_Basics.mlflow \
MLFLOW_TRACKING_USERNAME=faridkhan5 \
MLFLOW_TRACKING_PASSWORD=801f7f4defd33f302dd86616acda87ee94a5d725  \
python script.py


```bash

export MLFLOW_TRACKING_URI=https://dagshub.com/faridkhan5/MLFlow_Basics.mlflow

export MLFLOW_TRACKING_USERNAME=faridkhan5

export MLFLOW_TRACKING_PASSWORD=801f7f4defd33f302dd86616acda87ee94a5d725

```


## MLFlow on AWS setup:
1. Login to AWS console
2. Create IAM user with AdministratorAccess
3. Export the credentials in your AWS CLI by running "aws configure"
4. Create an s3 bucket
5. Create EC2 machine (Ubuntu) & add Security groups 500 port

Run the following bash code on EC2 machine
```bash
sudo apt update

sudo apt install python3-pip

sudo pip3 install pipenv

sudo pip3 install virtualenv

mkdir mlflow

cd mlflow

pipenv install mlflow

pipenv install awscli

pipenv install boto3

pipenv shell

## set aws credentials
aws configure

#finally
mlflow server -h 0.0.0.0 --default-artifact-root s3://mlflow-buc1

#open Public IPv4 DNS to the port 5000

#set uri in our local terminal and in our code
export MLFLOW_TRACKING_URI=http://ec2-16-170-133-156.eu-north-1.compute.amazonaws.com:5000/

```