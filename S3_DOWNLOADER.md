# Langflow AWS Integration Guide - IRSA on EKS

Complete guide for configuring Langflow to access AWS Bedrock and S3 using IAM Roles for Service Accounts (IRSA).

## Quick Links

- [EKS Setup](#eks-setup)
- [Local Development Setup](#local-development)
- [Component Code](#component-code)
- [Troubleshooting](#troubleshooting)

---

## EKS Setup

### 1. Create OIDC Provider

```bash
CLUSTER_NAME="your-cluster"
REGION="us-east-1"
ACCOUNT_ID="your-account-id"

eksctl utils associate-iam-oidc-provider \
  --cluster $CLUSTER_NAME \
  --region $REGION \
  --approve
```

### 2. Get OIDC Provider URL

```bash
OIDC_PROVIDER=$(aws eks describe-cluster \
  --name $CLUSTER_NAME \
  --query "cluster.identity.oidc.issuer" \
  --output text | sed 's|https://||')

echo $OIDC_PROVIDER
```

### 3. Create IAM Role

Create `trust-policy.json`:
```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {
      "Federated": "arn:aws:iam::ACCOUNT_ID:oidc-provider/OIDC_PROVIDER"
    },
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {
      "StringEquals": {
        "OIDC_PROVIDER:sub": "system:serviceaccount:langflow:langflow-sa",
        "OIDC_PROVIDER:aud": "sts.amazonaws.com"
      }
    }
  }]
}
```

Replace `ACCOUNT_ID` and `OIDC_PROVIDER`, then:
```bash
aws iam create-role \
  --role-name LangflowRole \
  --assume-role-policy-document file://trust-policy.json
```

### 4. Attach Permissions

```bash
# Bedrock
aws iam attach-role-policy \
  --role-name LangflowRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonBedrockFullAccess

# S3
aws iam attach-role-policy \
  --role-name LangflowRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess
```

### 5. Create Service Account

Create `serviceaccount.yaml`:
```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: langflow-sa
  namespace: langflow
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::ACCOUNT_ID:role/LangflowRole
```

Apply:
```bash
kubectl create namespace langflow
kubectl apply -f serviceaccount.yaml
```

### 6. Deploy Langflow

Create `deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: langflow
  namespace: langflow
spec:
  selector:
    matchLabels:
      app: langflow
  template:
    metadata:
      labels:
        app: langflow
    spec:
      serviceAccountName: langflow-sa
      containers:
      - name: langflow
        image: langflow/langflow:latest
        ports:
        - containerPort: 7860
```

Apply:
```bash
kubectl apply -f deployment.yaml
```

---

## Local Development

### Option 1: AWS CLI (Recommended)

```bash
aws configure
# Enter: Access Key ID, Secret Key, region (us-east-1)

aws sts get-caller-identity  # Verify
```

**In Langflow**: Leave all credential fields empty.

### Option 2: Role Assumption

Create role:
```bash
cat > local-trust.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"AWS": "arn:aws:iam::ACCOUNT_ID:user/YOUR_USER"},
    "Action": "sts:AssumeRole"
  }]
}
EOF

aws iam create-role \
  --role-name LangflowLocalRole \
  --assume-role-policy-document file://local-trust.json

aws iam attach-role-policy \
  --role-name LangflowLocalRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonBedrockFullAccess
```

**In Langflow**: 
- IAM Role ARN: `arn:aws:iam::ACCOUNT_ID:role/LangflowLocalRole`
- Leave credentials empty

---

## Component Code

### Bedrock Component

See full code in the artifacts panel → `bedrock_converse_component.py`

Key points:
- Supports IRSA (automatic on EKS)
- Supports AWS CLI credentials
- Supports role assumption
- Leave credentials empty for automatic resolution

### S3 Downloader Component

See full code in the artifacts panel → `s3_downloader_component.py`

Key points:
- Downloads single files or prefixes
- Preserves directory structure
- Same authentication as Bedrock component

---

## Verification

### EKS
```bash
POD=$(kubectl get pods -n langflow -l app=langflow -o jsonpath='{.items[0].metadata.name}')

# Check service account
kubectl exec -n langflow $POD -- env | grep AWS

# Test credentials
kubectl exec -n langflow $POD -- aws sts get-caller-identity
```

### Local
```bash
aws sts get-caller-identity
aws bedrock list-foundation-models --region us-east-1
```

---

## Troubleshooting

### Error: InvalidClientTokenId

**Cause**: Wrong or missing AWS credentials

**Fix**:
```bash
# Check region (must be us-east-1, not us-east1)
cat ~/.aws/config

# Reconfigure if needed
aws configure
```

### Error: Access Denied

**Cause**: Missing permissions

**Fix**:
```bash
# Check attached policies
aws iam list-attached-role-policies --role-name LangflowRole

# Add missing policy
aws iam attach-role-policy \
  --role-name LangflowRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonBedrockFullAccess
```

### Error: Pod can't assume role

**Cause**: IRSA misconfiguration

**Fix**:
```bash
# Verify service account annotation
kubectl describe sa langflow-sa -n langflow

# Verify trust policy
aws iam get-role --role-name LangflowRole
```

---

## Security Best Practices

1. **Least Privilege**: Use custom policies instead of `*FullAccess`
2. **Audit Logging**: Enable CloudTrail
3. **Network Policies**: Restrict pod egress
4. **Regular Reviews**: Audit permissions quarterly

---

## Quick Reference

**EKS (IRSA)**:
- ✅ No credentials needed
- ✅ Auto-rotating credentials
- ✅ Per-pod permissions

**Local (AWS CLI)**:
- ✅ Simple setup
- ✅ Uses `~/.aws/credentials`
- ⚠️ Manual credential rotation

**Local (Role Assumption)**:
- ✅ Granular permissions
- ✅ Cross-account access
- ⚠️ Requires additional setup

---

**Last Updated**: February 2026