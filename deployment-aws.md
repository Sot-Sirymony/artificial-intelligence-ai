# 🚀 AWS Deployment Guide for Emotion Detection

This guide provides step-by-step instructions to deploy your Text Emotion Detection project on AWS using various services.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Deployment Options](#deployment-options)
3. [Option 1: EC2 Instance Deployment](#option-1-ec2-instance-deployment)
4. [Option 2: AWS Lambda + API Gateway](#option-2-aws-lambda--api-gateway)
5. [Option 3: AWS ECS/Fargate](#option-3-aws-ecsfargate)
6. [Option 4: AWS SageMaker](#option-4-aws-sagemaker)
7. [Monitoring and Logging](#monitoring-and-logging)
8. [Cost Optimization](#cost-optimization)
9. [Security Best Practices](#security-best-practices)
10. [Troubleshooting](#troubleshooting)

---

## 🔧 Prerequisites

### AWS Account Setup
- AWS Account with appropriate permissions
- AWS CLI installed and configured
- Basic knowledge of AWS services

### Required AWS Services
- **EC2** (for server deployment)
- **Lambda** (for serverless deployment)
- **API Gateway** (for REST API)
- **S3** (for model storage)
- **CloudWatch** (for monitoring)
- **IAM** (for permissions)

### Local Setup
```bash
# Install AWS CLI
pip install awscli

# Configure AWS credentials
aws configure
```

---

## 🎯 Deployment Options

| Option | Use Case | Pros | Cons |
|--------|----------|------|------|
| **EC2** | High traffic, full control | Full control, persistent | More expensive, manual scaling |
| **Lambda** | Low traffic, cost-effective | Pay-per-use, auto-scaling | Cold starts, 15min timeout |
| **ECS/Fargate** | Containerized deployment | Scalable, managed | More complex setup |
| **SageMaker** | ML-focused deployment | ML-optimized, managed | Expensive for simple apps |

---

## 🖥️ Option 1: EC2 Instance Deployment

### Step 1: Prepare Application

Create a production-ready version of your app:

```bash
# Create production requirements
cat > requirements-prod.txt << EOF
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
nltk>=3.8.0
joblib>=1.3.0
plotly>=5.15.0
gunicorn>=20.1.0
EOF

# Create production app
cat > app-prod.py << 'EOF'
import streamlit as st
import os
import sys
sys.path.append('src')
from src.predict import EmotionPredictor

# Production configuration
st.set_page_config(
    page_title="Emotion Detection",
    page_icon="🎭",
    layout="wide"
)

@st.cache_resource
def load_model():
    return EmotionPredictor("model/emotion_model.pkl")

def main():
    st.title("🎭 Emotion Detection")
    
    predictor = load_model()
    
    text = st.text_area("Enter text to analyze:")
    if st.button("Analyze"):
        if text:
            result = predictor.predict_emotion(text)
            st.success(f"Emotion: {result['emotion'].title()}")
            st.info(f"Confidence: {result['confidence']:.2%}")

if __name__ == "__main__":
    main()
EOF
```

### Step 2: Create EC2 Instance

```bash
# Create security group
aws ec2 create-security-group \
    --group-name emotion-detection-sg \
    --description "Security group for emotion detection app"

# Add rules
aws ec2 authorize-security-group-ingress \
    --group-name emotion-detection-sg \
    --protocol tcp \
    --port 22 \
    --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
    --group-name emotion-detection-sg \
    --protocol tcp \
    --port 8501 \
    --cidr 0.0.0.0/0

# Launch EC2 instance (Ubuntu 22.04)
aws ec2 run-instances \
    --image-id ami-0c02fb55956c7d316 \
    --count 1 \
    --instance-type t3.medium \
    --key-name your-key-pair \
    --security-group-ids emotion-detection-sg
```

### Step 3: Setup EC2 Instance

```bash
# SSH into your instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install python3 python3-pip python3-venv -y

# Create application directory
mkdir -p /home/ubuntu/emotion-detection
cd /home/ubuntu/emotion-detection

# Upload your application files (use scp or git)
# scp -r . ubuntu@your-ec2-ip:/home/ubuntu/emotion-detection/

# Setup virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements-prod.txt

# Download NLTK data
python3 -c "
import ssl
import nltk
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
"

# Train model (if not already trained)
python3 train_model.py
```

### Step 4: Run Application

```bash
# Create systemd service
sudo tee /etc/systemd/system/emotion-detection.service << EOF
[Unit]
Description=Emotion Detection Streamlit App
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/emotion-detection
Environment=PATH=/home/ubuntu/emotion-detection/venv/bin
ExecStart=/home/ubuntu/emotion-detection/venv/bin/streamlit run app-prod.py --server.port 8501 --server.address 0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Start service
sudo systemctl daemon-reload
sudo systemctl enable emotion-detection
sudo systemctl start emotion-detection

# Check status
sudo systemctl status emotion-detection
```

### Step 5: Configure Nginx (Optional)

```bash
# Install nginx
sudo apt install nginx -y

# Configure nginx
sudo tee /etc/nginx/sites-available/emotion-detection << EOF
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

# Enable site
sudo ln -s /etc/nginx/sites-available/emotion-detection /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

---

## ⚡ Option 2: AWS Lambda + API Gateway

### Step 1: Prepare Lambda Function

```python
# lambda_function.py
import json
import pickle
import boto3
import os
import sys
from io import BytesIO

# Add src to path
sys.path.append('src')
from src.preprocessing import TextPreprocessor

# Initialize preprocessor
preprocessor = TextPreprocessor()

def load_model_from_s3():
    """Load model from S3"""
    s3 = boto3.client('s3')
    bucket = os.environ['MODEL_BUCKET']
    key = os.environ['MODEL_KEY']
    
    response = s3.get_object(Bucket=bucket, Key=key)
    model_data = response['Body'].read()
    return pickle.loads(model_data)

def lambda_handler(event, context):
    """Lambda function handler"""
    try:
        # Parse request
        body = json.loads(event['body'])
        text = body.get('text', '')
        
        if not text:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Text is required'})
            }
        
        # Load model
        model = load_model_from_s3()
        
        # Preprocess text
        cleaned_text = preprocessor.clean_text(text)
        
        # Make prediction
        prediction = model.predict([cleaned_text])[0]
        probabilities = model.predict_proba([cleaned_text])[0]
        
        # Get confidence
        confidence = max(probabilities)
        
        # Prepare response
        response = {
            'text': text,
            'emotion': prediction,
            'confidence': float(confidence),
            'probabilities': dict(zip(model.classes_, probabilities.tolist()))
        }
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(response)
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

### Step 2: Create Deployment Package

```bash
# Create deployment directory
mkdir lambda-deployment
cd lambda-deployment

# Copy function and dependencies
cp ../lambda_function.py .
cp -r ../src .
cp ../model/emotion_model.pkl .

# Install dependencies
pip install -r ../requirements.txt -t .

# Create ZIP file
zip -r emotion-detection-lambda.zip .
```

### Step 3: Upload Model to S3

```bash
# Create S3 bucket
aws s3 mb s3://your-emotion-detection-models

# Upload model
aws s3 cp model/emotion_model.pkl s3://your-emotion-detection-models/emotion_model.pkl
```

### Step 4: Create Lambda Function

```bash
# Create Lambda function
aws lambda create-function \
    --function-name emotion-detection \
    --runtime python3.9 \
    --role arn:aws:iam::your-account:role/lambda-execution-role \
    --handler lambda_function.lambda_handler \
    --zip-file fileb://emotion-detection-lambda.zip \
    --timeout 30 \
    --memory-size 512 \
    --environment Variables='{MODEL_BUCKET=your-emotion-detection-models,MODEL_KEY=emotion_model.pkl}'
```

### Step 5: Create API Gateway

```bash
# Create REST API
aws apigateway create-rest-api \
    --name emotion-detection-api \
    --description "Emotion Detection API"

# Get API ID
API_ID=$(aws apigateway get-rest-apis --query 'items[?name==`emotion-detection-api`].id' --output text)

# Create resource
aws apigateway create-resource \
    --rest-api-id $API_ID \
    --parent-id $(aws apigateway get-resources --rest-api-id $API_ID --query 'items[?path==`/`].id' --output text) \
    --path-part "predict"

# Create POST method
aws apigateway put-method \
    --rest-api-id $API_ID \
    --resource-id $(aws apigateway get-resources --rest-api-id $API_ID --query 'items[?path==`/predict`].id' --output text) \
    --http-method POST \
    --authorization-type NONE

# Integrate with Lambda
aws apigateway put-integration \
    --rest-api-id $API_ID \
    --resource-id $(aws apigateway get-resources --rest-api-id $API_ID --query 'items[?path==`/predict`].id' --output text) \
    --http-method POST \
    --type AWS_PROXY \
    --integration-http-method POST \
    --uri arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/arn:aws:lambda:us-east-1:your-account:function:emotion-detection/invocations

# Deploy API
aws apigateway create-deployment \
    --rest-api-id $API_ID \
    --stage-name prod
```

---

## 🐳 Option 3: AWS ECS/Fargate

### Step 1: Create Dockerfile

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "
import ssl
import nltk
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
"

# Copy application code
COPY . .

# Train model
RUN python train_model.py

# Expose port
EXPOSE 8501

# Run application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Step 2: Build and Push Docker Image

```bash
# Build image
docker build -t emotion-detection .

# Tag for ECR
docker tag emotion-detection:latest your-account.dkr.ecr.us-east-1.amazonaws.com/emotion-detection:latest

# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin your-account.dkr.ecr.us-east-1.amazonaws.com

# Push image
docker push your-account.dkr.ecr.us-east-1.amazonaws.com/emotion-detection:latest
```

### Step 3: Create ECS Cluster and Service

```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name emotion-detection-cluster

# Create task definition
aws ecs register-task-definition \
    --family emotion-detection \
    --network-mode awsvpc \
    --requires-compatibilities FARGATE \
    --cpu 512 \
    --memory 1024 \
    --execution-role-arn arn:aws:iam::your-account:role/ecsTaskExecutionRole \
    --container-definitions '[
        {
            "name": "emotion-detection",
            "image": "your-account.dkr.ecr.us-east-1.amazonaws.com/emotion-detection:latest",
            "portMappings": [
                {
                    "containerPort": 8501,
                    "protocol": "tcp"
                }
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/emotion-detection",
                    "awslogs-region": "us-east-1",
                    "awslogs-stream-prefix": "ecs"
                }
            }
        }
    ]'

# Create service
aws ecs create-service \
    --cluster emotion-detection-cluster \
    --service-name emotion-detection-service \
    --task-definition emotion-detection:1 \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[subnet-12345],securityGroups=[sg-12345],assignPublicIp=ENABLED}"
```

---

## 🤖 Option 4: AWS SageMaker

### Step 1: Prepare SageMaker Model

```python
# inference.py
import os
import json
import pickle
import sys
from io import BytesIO

# Add src to path
sys.path.append('src')
from src.preprocessing import TextPreprocessor

preprocessor = TextPreprocessor()

def model_fn(model_dir):
    """Load the model"""
    model_path = os.path.join(model_dir, 'emotion_model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def input_fn(request_body, request_content_type):
    """Parse input data"""
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return input_data.get('text', '')
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Make prediction"""
    if not input_data:
        return {'error': 'No text provided'}
    
    # Preprocess text
    cleaned_text = preprocessor.clean_text(input_data)
    
    # Make prediction
    prediction = model.predict([cleaned_text])[0]
    probabilities = model.predict_proba([cleaned_text])[0]
    confidence = max(probabilities)
    
    return {
        'text': input_data,
        'emotion': prediction,
        'confidence': float(confidence),
        'probabilities': dict(zip(model.classes_, probabilities.tolist()))
    }

def output_fn(prediction, content_type):
    """Format output"""
    if content_type == 'application/json':
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")
```

### Step 2: Create SageMaker Model

```bash
# Create model archive
tar -czf model.tar.gz model/emotion_model.pkl inference.py src/

# Upload to S3
aws s3 cp model.tar.gz s3://your-sagemaker-models/emotion-detection/

# Create SageMaker model
aws sagemaker create-model \
    --model-name emotion-detection \
    --primary-container Image=your-account.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:latest,ModelDataUrl=s3://your-sagemaker-models/emotion-detection/model.tar.gz

# Create endpoint configuration
aws sagemaker create-endpoint-config \
    --endpoint-config-name emotion-detection-config \
    --production-variants VariantName=default,ModelName=emotion-detection,InitialInstanceCount=1,InstanceType=ml.t3.medium

# Create endpoint
aws sagemaker create-endpoint \
    --endpoint-name emotion-detection-endpoint \
    --endpoint-config-name emotion-detection-config
```

---

## 📊 Monitoring and Logging

### CloudWatch Setup

```bash
# Create CloudWatch dashboard
aws cloudwatch put-dashboard \
    --dashboard-name emotion-detection-dashboard \
    --dashboard-body '{
        "widgets": [
            {
                "type": "metric",
                "properties": {
                    "metrics": [
                        ["AWS/Lambda", "Invocations", "FunctionName", "emotion-detection"],
                        [".", "Errors", ".", "."],
                        [".", "Duration", ".", "."]
                    ],
                    "period": 300,
                    "stat": "Sum",
                    "region": "us-east-1",
                    "title": "Lambda Metrics"
                }
            }
        ]
    }'
```

### Application Logging

```python
# Add to your application
import logging
import boto3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CloudWatch client
cloudwatch = boto3.client('cloudwatch')

def log_prediction(text, emotion, confidence):
    """Log prediction to CloudWatch"""
    logger.info(f"Prediction: {text} -> {emotion} ({confidence:.2%})")
    
    # Send custom metric
    cloudwatch.put_metric_data(
        Namespace='EmotionDetection',
        MetricData=[
            {
                'MetricName': 'Predictions',
                'Value': 1,
                'Unit': 'Count',
                'Dimensions': [
                    {'Name': 'Emotion', 'Value': emotion}
                ]
            }
        ]
    )
```

---

## 💰 Cost Optimization

### EC2 Cost Optimization
- Use Spot Instances for non-critical workloads
- Right-size instances based on usage
- Use Reserved Instances for predictable workloads

### Lambda Cost Optimization
- Optimize function memory (affects CPU allocation)
- Use provisioned concurrency to avoid cold starts
- Implement caching strategies

### General Tips
- Monitor usage with CloudWatch
- Set up billing alerts
- Use AWS Cost Explorer for analysis

---

## 🔒 Security Best Practices

### IAM Policies

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject"
            ],
            "Resource": "arn:aws:s3:::your-emotion-detection-models/*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            "Resource": "arn:aws:logs:*:*:*"
        }
    ]
}
```

### Security Groups
- Restrict access to necessary ports only
- Use VPC for network isolation
- Implement WAF for web applications

### Data Protection
- Encrypt data at rest and in transit
- Use AWS KMS for key management
- Implement proper access controls

---

## 🛠️ Troubleshooting

### Common Issues

1. **Cold Start Issues (Lambda)**
   ```bash
   # Use provisioned concurrency
   aws lambda put-provisioned-concurrency-config \
       --function-name emotion-detection \
       --qualifier prod \
       --provisioned-concurrent-executions 5
   ```

2. **Memory Issues**
   ```bash
   # Increase Lambda memory
   aws lambda update-function-configuration \
       --function-name emotion-detection \
       --memory-size 1024
   ```

3. **Timeout Issues**
   ```bash
   # Increase Lambda timeout
   aws lambda update-function-configuration \
       --function-name emotion-detection \
       --timeout 60
   ```

### Monitoring Commands

```bash
# Check Lambda logs
aws logs describe-log-groups --log-group-name-prefix /aws/lambda/emotion-detection

# Check EC2 status
aws ec2 describe-instances --instance-ids i-1234567890abcdef0

# Check ECS service status
aws ecs describe-services --cluster emotion-detection-cluster --services emotion-detection-service
```

---

## 📝 Deployment Checklist

- [ ] AWS account configured
- [ ] Application tested locally
- [ ] Model trained and saved
- [ ] Dependencies documented
- [ ] Security groups configured
- [ ] IAM roles created
- [ ] Monitoring set up
- [ ] Logging configured
- [ ] Cost alerts enabled
- [ ] Backup strategy implemented

---

## 🎯 Next Steps

1. **Choose deployment option** based on your requirements
2. **Follow the step-by-step guide** for your chosen option
3. **Test the deployment** thoroughly
4. **Monitor performance** and costs
5. **Scale as needed** based on usage patterns

For additional support, refer to:
- [AWS Documentation](https://docs.aws.amazon.com/)
- [AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)
- [AWS Cost Optimization](https://aws.amazon.com/cost-optimization/)

---

**Happy Deploying! 🚀** 