from functools import lru_cache
from typing import Any, List

from langflow.field_typing import LanguageModel
from langflow.inputs.inputs import BoolInput, FloatInput, IntInput, MessageTextInput, SecretStrInput
from langflow.io import DictInput, DropdownInput

from lfx.base.models.aws_constants import AWS_REGIONS
from lfx.base.models.model import LCModelComponent
from lfx.log.logger import logger
from lfx.schema.dotdict import dotdict


# Fallback hardcoded models if API call fails
FALLBACK_MODELS = [
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "anthropic.claude-3-5-haiku-20241022-v1:0",
    "anthropic.claude-3-opus-20240229-v1:0",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
]


@lru_cache(maxsize=10)
def get_bedrock_models(region: str = "us-east-1") -> List[str]:
    """Dynamically fetch available Bedrock foundation models.
    
    Args:
        region: AWS region to query
    
    Returns:
        List of available model IDs for Converse API, sorted by provider and name
    """
    try:
        import boto3
        
        # Create Bedrock client
        bedrock_client = boto3.client('bedrock', region_name=region)
        
        # Fetch available models
        response = bedrock_client.list_foundation_models()
        
        # Extract model IDs for active models that support Converse API
        model_ids = []
        for model in response.get('modelSummaries', []):
            # Only include active models
            if model.get('modelLifecycle', {}).get('status') != 'ACTIVE':
                continue
                
            model_id = model.get('modelId')
            if not model_id:
                continue
                
            # Check if model supports Converse API
            supported_features = model.get('inferenceTypesSupported', [])
            if 'ON_DEMAND' in supported_features:
                model_ids.append(model_id)
        
        # Sort models by provider, then name
        return sorted(model_ids) if model_ids else FALLBACK_MODELS
        
    except Exception as e:
        # If API call fails, return fallback list
        logger.warning(f"Failed to fetch Bedrock models dynamically: {e}. Using fallback list.")
        return FALLBACK_MODELS


class AmazonBedrockConverseComponent(LCModelComponent):
    display_name: str = "Amazon Bedrock Converse (Dynamic)"
    description: str = (
        "Generate text using Amazon Bedrock LLMs with the modern Converse API. "
        "Dynamically loads available models from AWS Bedrock."
    )
    icon = "Amazon"
    name = "AmazonBedrockConverseModelDynamic"
    beta = True

    inputs = [
        *LCModelComponent.get_base_inputs(),
        DropdownInput(
            name="region_name",
            display_name="Region Name",
            value="us-east-1",
            options=AWS_REGIONS,
            info="The AWS region where your Bedrock resources are located. "
                 "Changing this will update available models.",
            real_time_refresh=True,  # This enables dynamic updates
        ),
        DropdownInput(
            name="model_id",
            display_name="Model ID",
            options=FALLBACK_MODELS,
            value="anthropic.claude-3-5-sonnet-20241022-v2:0",
            info="Select from available Bedrock models. List updates based on region.",
            refresh_button=True,  # Adds a refresh button
        ),
        SecretStrInput(
            name="role_arn",
            display_name="IAM Role ARN",
            info="ARN of the IAM role to assume (e.g., arn:aws:iam::123456789012:role/MyBedrockRole). "
            "If provided, will use STS to assume this role. Leave empty to use direct credentials "
            "or automatic credential resolution (AWS CLI, IRSA on EKS).",
            advanced=True,
            required=False,
        ),
        MessageTextInput(
            name="role_session_name",
            display_name="Role Session Name",
            value="langflow-bedrock-session",
            info="Session name for the assumed role.",
            advanced=True,
        ),
        SecretStrInput(
            name="aws_access_key_id",
            display_name="AWS Access Key ID",
            info="The access key for your AWS account. "
            "Leave empty to use IAM role ARN, environment variables, or AWS CLI credentials.",
            value="AWS_ACCESS_KEY_ID",
            required=False,
        ),
        SecretStrInput(
            name="aws_secret_access_key",
            display_name="AWS Secret Access Key",
            info="The secret key for your AWS account. "
            "Leave empty to use IAM role ARN, environment variables, or AWS CLI credentials.",
            value="AWS_SECRET_ACCESS_KEY",
            required=False,
        ),
        SecretStrInput(
            name="aws_session_token",
            display_name="AWS Session Token",
            advanced=True,
            info="The session key for your AWS account. "
            "Only needed for temporary credentials.",
            load_from_db=False,
        ),
        SecretStrInput(
            name="credentials_profile_name",
            display_name="Credentials Profile Name",
            advanced=True,
            info="The name of the profile to use from your "
            "~/.aws/credentials file. "
            "If not provided, the default profile will be used.",
            load_from_db=False,
        ),
        MessageTextInput(
            name="endpoint_url",
            display_name="Endpoint URL",
            advanced=True,
            info="The URL of the Bedrock endpoint to use.",
        ),
        FloatInput(
            name="temperature",
            display_name="Temperature",
            value=0.7,
            info="Controls randomness in output. Higher values make output more random.",
            advanced=True,
        ),
        IntInput(
            name="max_tokens",
            display_name="Max Tokens",
            value=4096,
            info="Maximum number of tokens to generate.",
            advanced=True,
        ),
        FloatInput(
            name="top_p",
            display_name="Top P",
            value=0.9,
            info="Nucleus sampling parameter. Controls diversity of output.",
            advanced=True,
        ),
        IntInput(
            name="top_k",
            display_name="Top K",
            value=250,
            info="Limits the number of highest probability vocabulary tokens to consider. "
            "Note: Not all models support top_k. Use 'Additional Model Fields' for manual configuration if needed.",
            advanced=True,
        ),
        BoolInput(
            name="disable_streaming",
            display_name="Disable Streaming",
            value=False,
            info="If True, disables streaming responses. Useful for batch processing.",
            advanced=True,
        ),
        DictInput(
            name="additional_model_fields",
            display_name="Additional Model Fields",
            advanced=True,
            is_list=True,
            info="Additional model-specific parameters for fine-tuning behavior.",
        ),
    ]

    async def update_build_config(
        self, build_config: dotdict, field_value: Any, field_name: str | None = None
    ) -> dotdict:
        """
        Dynamically update the build configuration based on field changes.
        
        This method is called by Langflow when a field value changes, allowing
        dynamic updates to dropdown options and field visibility.
        """
        
        # When region changes, update the model list
        if field_name == "region_name":
            try:
                logger.info(f"Fetching Bedrock models for region: {field_value}")
                
                # Fetch models for the new region
                models = get_bedrock_models(region=field_value)
                
                # Update the model_id dropdown options
                build_config["model_id"]["options"] = models
                
                # If current model isn't in new list, reset to first model
                current_model = build_config["model_id"].get("value")
                if current_model not in models:
                    build_config["model_id"]["value"] = models[0] if models else FALLBACK_MODELS[0]
                
                logger.info(f"Updated model list with {len(models)} models")
                
            except Exception as e:
                logger.error(f"Error fetching Bedrock models: {e}")
                # On error, use fallback models
                build_config["model_id"]["options"] = FALLBACK_MODELS
                build_config["model_id"]["value"] = FALLBACK_MODELS[0]
        
        # When the refresh button is clicked on model_id
        elif field_name == "model_id":
            try:
                # Clear cache to force fresh fetch
                get_bedrock_models.cache_clear()
                
                region = build_config["region_name"].get("value", "us-east-1")
                logger.info(f"Refreshing Bedrock models for region: {region}")
                
                # Fetch fresh models
                models = get_bedrock_models(region=region)
                
                # Update the dropdown
                build_config["model_id"]["options"] = models
                
                logger.info(f"Refreshed model list with {len(models)} models")
                
            except Exception as e:
                logger.error(f"Error refreshing Bedrock models: {e}")
                build_config["model_id"]["options"] = FALLBACK_MODELS
        
        return build_config

    def build_model(self) -> LanguageModel:
        try:
            from langchain_aws.chat_models.bedrock_converse import ChatBedrockConverse
        except ImportError as e:
            msg = "langchain_aws is not installed. Please install it with `pip install langchain_aws`."
            raise ImportError(msg) from e

        import boto3

        init_params = {
            "model": self.model_id,
            "region_name": self.region_name,
        }

        # Handle role assumption if role_arn is provided
        if hasattr(self, "role_arn") and self.role_arn:
            try:
                region = self.region_name if hasattr(self, "region_name") and self.region_name else "us-east-1"
                sts_client = boto3.client('sts', region_name=region)
                
                assumed_role = sts_client.assume_role(
                    RoleArn=self.role_arn,
                    RoleSessionName=getattr(self, "role_session_name", "langflow-bedrock-session")
                )
                
                credentials = assumed_role['Credentials']
                init_params["aws_access_key_id"] = credentials['AccessKeyId']
                init_params["aws_secret_access_key"] = credentials['SecretAccessKey']
                init_params["aws_session_token"] = credentials['SessionToken']
                
            except Exception as e:
                msg = f"Failed to assume role {self.role_arn}: {str(e)}"
                raise ValueError(msg) from e
        else:
            # Use provided credentials or let boto3 use default credential chain
            if hasattr(self, "aws_access_key_id") and self.aws_access_key_id:
                init_params["aws_access_key_id"] = self.aws_access_key_id
            if hasattr(self, "aws_secret_access_key") and self.aws_secret_access_key:
                init_params["aws_secret_access_key"] = self.aws_secret_access_key
            if hasattr(self, "aws_session_token") and self.aws_session_token:
                init_params["aws_session_token"] = self.aws_session_token
            if hasattr(self, "credentials_profile_name") and self.credentials_profile_name:
                init_params["credentials_profile_name"] = self.credentials_profile_name
        
        if hasattr(self, "endpoint_url") and self.endpoint_url:
            init_params["endpoint_url"] = self.endpoint_url

        if hasattr(self, "temperature") and self.temperature is not None:
            init_params["temperature"] = self.temperature
        if hasattr(self, "max_tokens") and self.max_tokens is not None:
            init_params["max_tokens"] = self.max_tokens
        if hasattr(self, "top_p") and self.top_p is not None:
            init_params["top_p"] = self.top_p

        if hasattr(self, "disable_streaming") and self.disable_streaming:
            init_params["disable_streaming"] = True

        additional_model_request_fields = {}

        if hasattr(self, "additional_model_fields") and self.additional_model_fields:
            for field in self.additional_model_fields:
                if isinstance(field, dict):
                    additional_model_request_fields.update(field)

        if additional_model_request_fields:
            init_params["additional_model_request_fields"] = additional_model_request_fields

        try:
            output = ChatBedrockConverse(**init_params)
        except Exception as e:
            error_details = str(e)
            if "validation error" in error_details.lower():
                msg = (
                    f"ChatBedrockConverse validation error: {error_details}. "
                    f"This may be due to incompatible parameters for model '{self.model_id}'. "
                    f"Consider adjusting the model parameters or trying the legacy Amazon Bedrock component."
                )
            elif "converse api" in error_details.lower():
                msg = (
                    f"Converse API error: {error_details}. "
                    f"The model '{self.model_id}' may not support the Converse API. "
                    f"Try using the legacy Amazon Bedrock component instead."
                )
            else:
                msg = f"Could not initialize ChatBedrockConverse: {error_details}"
            raise ValueError(msg) from e

        return output


# Utility function for testing
def list_bedrock_models(region: str = "us-east-1"):
    """Utility function to list all available Bedrock models in a region.
    
    Usage:
        python bedrock_component.py --region us-east-1
    """
    models = get_bedrock_models(region=region)
    print(f"\n=== Available Bedrock Models in {region} ===\n")
    
    # Group by provider
    providers = {}
    for model_id in models:
        provider = model_id.split('.')[0] if '.' in model_id else 'other'
        if provider not in providers:
            providers[provider] = []
        providers[provider].append(model_id)
    
    for provider, model_list in sorted(providers.items()):
        print(f"\n{provider.upper()}:")
        for model in model_list:
            print(f"  - {model}")
    
    print(f"\nTotal: {len(models)} models\n")


if __name__ == "__main__":
    import sys
    
    # Allow running as script to list models
    region = "us-east-1"
    if len(sys.argv) > 1:
        if sys.argv[1] == "--region" and len(sys.argv) > 2:
            region = sys.argv[2]
    
    list_bedrock_models(region=region)