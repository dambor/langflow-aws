from pathlib import Path
from typing import Any

from lfx.custom.custom_component.component import Component
from lfx.io import (
    BoolInput,
    DropdownInput,
    MessageTextInput,
    Output,
    SecretStrInput,
    StrInput,
)


class S3BucketDownloaderComponent(Component):
    """S3BucketDownloaderComponent downloads files from an S3 bucket.

    This component supports multiple authentication methods:
    - IAM Role ARN (for role assumption)
    - Direct AWS credentials (access key/secret key)
    - AWS CLI credentials (automatic from ~/.aws/credentials)
    - IRSA (IAM Roles for Service Accounts) when running on EKS

    Attributes:
        display_name (str): The display name of the component.
        description (str): A brief description of the component's functionality.
        icon (str): The icon representing the component.
        name (str): The internal name of the component.
        inputs (list): A list of input configurations required by the component.
        outputs (list): A list of output configurations provided by the component.

    Methods:
        download_files() -> list:
            Downloads files from S3 bucket based on the specified prefix/key.
        _get_s3_client() -> Any:
            Creates and returns an S3 client with appropriate credentials.
    """

    display_name = "S3 Bucket Downloader"
    description = "Downloads files from S3 bucket with IAM role support."
    icon = "Amazon"
    name = "s3bucketdownloader"

    inputs = [
        SecretStrInput(
            name="role_arn",
            display_name="IAM Role ARN",
            info="ARN of the IAM role to assume (e.g., arn:aws:iam::123456789012:role/MyS3Role). "
            "If provided, will use STS to assume this role. Leave empty to use direct credentials "
            "or automatic credential resolution (AWS CLI, IRSA on EKS).",
            advanced=True,
            required=False,
        ),
        MessageTextInput(
            name="role_session_name",
            display_name="Role Session Name",
            value="langflow-s3-session",
            info="Session name for the assumed role.",
            advanced=True,
        ),
        SecretStrInput(
            name="aws_access_key_id",
            display_name="AWS Access Key ID",
            info="AWS Access Key ID. Leave empty to use IAM role ARN, environment variables, "
            "or AWS CLI credentials.",
            required=False,
            password=True,
        ),
        SecretStrInput(
            name="aws_secret_access_key",
            display_name="AWS Secret Key",
            info="AWS Secret Access Key. Leave empty to use IAM role ARN, environment variables, "
            "or AWS CLI credentials.",
            required=False,
            password=True,
        ),
        SecretStrInput(
            name="aws_session_token",
            display_name="AWS Session Token",
            info="AWS Session Token (for temporary credentials). Optional.",
            advanced=True,
            required=False,
            password=True,
        ),
        DropdownInput(
            name="region_name",
            display_name="Region Name",
            options=["us-east-1", "us-east-2", "us-west-1", "us-west-2", "eu-west-1", "eu-central-1", "ap-southeast-1"],
            value="us-east-1",
            info="AWS region where the S3 bucket is located.",
        ),
        StrInput(
            name="bucket_name",
            display_name="Bucket Name",
            info="Name of the S3 bucket to download from.",
            required=True,
        ),
        StrInput(
            name="s3_key",
            display_name="S3 Key/Prefix",
            info="S3 key (file path) or prefix to download. Use prefix to download multiple files.",
            required=True,
        ),
        StrInput(
            name="local_path",
            display_name="Local Download Path",
            value="./downloads",
            info="Local directory path where files will be downloaded.",
            advanced=True,
        ),
        DropdownInput(
            name="download_mode",
            display_name="Download Mode",
            options=["Single File", "All Files with Prefix"],
            value="Single File",
            info="Download a single file or all files matching a prefix.",
        ),
        BoolInput(
            name="create_local_dir",
            display_name="Create Local Directory",
            value=True,
            info="Automatically create the local download directory if it doesn't exist.",
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Downloaded Files", name="files", method="download_files"),
    ]

    def download_files(self) -> list:
        """Downloads files from S3 bucket.

        Returns:
            list: List of dictionaries containing download information for each file.
                  Each dict has 'key', 'local_path', and 'size' fields.
        """
        # Create local directory if needed
        local_dir = Path(self.local_path)
        if self.create_local_dir and not local_dir.exists():
            local_dir.mkdir(parents=True, exist_ok=True)
            self.log(f"Created directory: {local_dir}")

        s3_client = self._get_s3_client()
        downloaded_files = []

        if self.download_mode == "Single File":
            # Download single file
            local_file_path = local_dir / Path(self.s3_key).name
            
            self.log(f"Downloading s3://{self.bucket_name}/{self.s3_key} to {local_file_path}")
            
            s3_client.download_file(
                Bucket=self.bucket_name,
                Key=self.s3_key,
                Filename=str(local_file_path)
            )
            
            # Get file size
            response = s3_client.head_object(Bucket=self.bucket_name, Key=self.s3_key)
            file_size = response['ContentLength']
            
            downloaded_files.append({
                'key': self.s3_key,
                'local_path': str(local_file_path),
                'size': file_size
            })
            
            self.log(f"Downloaded: {self.s3_key} ({file_size} bytes)")

        else:  # All Files with Prefix
            # List and download all files with prefix
            self.log(f"Listing files with prefix: {self.s3_key}")
            
            paginator = s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=self.s3_key)
            
            file_count = 0
            for page in pages:
                if 'Contents' not in page:
                    continue
                    
                for obj in page['Contents']:
                    s3_key = obj['Key']
                    
                    # Skip directories (keys ending with /)
                    if s3_key.endswith('/'):
                        continue
                    
                    # Preserve directory structure
                    relative_path = s3_key[len(self.s3_key):].lstrip('/')
                    local_file_path = local_dir / relative_path
                    
                    # Create subdirectories if needed
                    local_file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    self.log(f"Downloading: {s3_key}")
                    
                    s3_client.download_file(
                        Bucket=self.bucket_name,
                        Key=s3_key,
                        Filename=str(local_file_path)
                    )
                    
                    downloaded_files.append({
                        'key': s3_key,
                        'local_path': str(local_file_path),
                        'size': obj['Size']
                    })
                    
                    file_count += 1
            
            self.log(f"Downloaded {file_count} files from s3://{self.bucket_name}/{self.s3_key}")

        return downloaded_files

    def _get_s3_client(self) -> Any:
        """Creates and returns an S3 client with appropriate credentials.

        This method supports multiple authentication methods:
        1. IAM Role assumption (if role_arn is provided)
        2. Direct credentials (if aws_access_key_id and aws_secret_access_key are provided)
        3. Automatic credential resolution (AWS CLI, environment variables, IRSA)

        Returns:
            Any: A boto3 S3 client instance.
        """
        try:
            import boto3
        except ImportError as e:
            msg = "boto3 is not installed. Please install it using `pip install boto3`."
            raise ImportError(msg) from e

        # Handle role assumption if role_arn is provided
        if hasattr(self, "role_arn") and self.role_arn:
            try:
                self.log(f"Assuming role: {self.role_arn}")
                
                # Create STS client with the same region
                sts_client = boto3.client('sts', region_name=self.region_name)
                
                assumed_role = sts_client.assume_role(
                    RoleArn=self.role_arn,
                    RoleSessionName=getattr(self, "role_session_name", "langflow-s3-session")
                )
                
                # Extract temporary credentials
                credentials = assumed_role['Credentials']
                
                return boto3.client(
                    's3',
                    region_name=self.region_name,
                    aws_access_key_id=credentials['AccessKeyId'],
                    aws_secret_access_key=credentials['SecretAccessKey'],
                    aws_session_token=credentials['SessionToken']
                )
                
            except Exception as e:
                msg = f"Failed to assume role {self.role_arn}: {str(e)}"
                raise ValueError(msg) from e
        
        # Use direct credentials if provided (and not empty strings)
        elif (hasattr(self, "aws_access_key_id") and self.aws_access_key_id and 
              hasattr(self, "aws_secret_access_key") and self.aws_secret_access_key):
            self.log("Using provided AWS credentials")
            
            client_params = {
                'region_name': self.region_name,
                'aws_access_key_id': self.aws_access_key_id,
                'aws_secret_access_key': self.aws_secret_access_key,
            }
            
            # Add session token if provided and not empty
            if (hasattr(self, "aws_session_token") and self.aws_session_token and 
                self.aws_session_token.strip()):
                client_params['aws_session_token'] = self.aws_session_token
            
            return boto3.client('s3', **client_params)
        
        # Use automatic credential resolution (AWS CLI, env vars, IRSA)
        else:
            self.log("Using automatic credential resolution (AWS CLI/IRSA)")
            return boto3.client('s3', region_name=self.region_name)