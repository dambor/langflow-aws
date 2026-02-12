from typing import Any

import requests
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain_openai import OpenAIEmbeddings

from lfx.base.embeddings.model import LCEmbeddingsModel
from lfx.base.models.aws_constants import AWS_REGIONS
from lfx.base.models.model_utils import get_ollama_models, is_valid_ollama_url
from lfx.base.models.openai_constants import OPENAI_EMBEDDING_MODEL_NAMES
from lfx.base.models.watsonx_constants import (
    IBM_WATSONX_URLS,
    WATSONX_EMBEDDING_MODEL_NAMES,
)
from lfx.field_typing import Embeddings
from lfx.io import (
    BoolInput,
    DictInput,
    DropdownInput,
    FloatInput,
    IntInput,
    MessageTextInput,
    SecretStrInput,
)
from lfx.log.logger import logger
from lfx.schema.dotdict import dotdict
from lfx.utils.util import transform_localhost_url

# Ollama API constants
HTTP_STATUS_OK = 200
JSON_MODELS_KEY = "models"
JSON_NAME_KEY = "name"
JSON_CAPABILITIES_KEY = "capabilities"
DESIRED_CAPABILITY = "embedding"
DEFAULT_OLLAMA_URL = "http://localhost:11434"

# Amazon Bedrock embedding model IDs
AWS_EMBEDDING_MODEL_IDS = [
    "amazon.titan-embed-text-v1",
    "amazon.titan-embed-text-v2:0",
    "amazon.titan-embed-image-v1",
    "cohere.embed-english-v3",
    "cohere.embed-multilingual-v3",
]


class EmbeddingModelComponent(LCEmbeddingsModel):
    display_name = "Embedding Model"
    description = "Generate embeddings using a specified provider."
    documentation: str = "https://docs.langflow.org/components-embedding-models"
    icon = "binary"
    name = "EmbeddingModel"
    category = "models"

    inputs = [
        DropdownInput(
            name="provider",
            display_name="Model Provider",
            options=["OpenAI", "Ollama", "IBM watsonx.ai", "Amazon Bedrock"],
            value="OpenAI",
            info="Select the embedding model provider",
            real_time_refresh=True,
            options_metadata=[
                {"icon": "OpenAI"},
                {"icon": "Ollama"},
                {"icon": "WatsonxAI"},
                {"icon": "Amazon"},
            ],
        ),
        MessageTextInput(
            name="api_base",
            display_name="API Base URL",
            info="Base URL for the API. Leave empty for default.",
            advanced=True,
        ),
        MessageTextInput(
            name="ollama_base_url",
            display_name="Ollama API URL",
            info=f"Endpoint of the Ollama API (Ollama only). Defaults to {DEFAULT_OLLAMA_URL}",
            value="",
            show=False,
        ),
        DropdownInput(
            name="base_url_ibm_watsonx",
            display_name="watsonx API Endpoint",
            info="The base URL of the API (IBM watsonx.ai only)",
            options=IBM_WATSONX_URLS,
            value=IBM_WATSONX_URLS[0],
            show=False,
            real_time_refresh=True,
        ),
        DropdownInput(
            name="model",
            display_name="Model Name",
            options=OPENAI_EMBEDDING_MODEL_NAMES,
            value=OPENAI_EMBEDDING_MODEL_NAMES[0],
            info="Select the embedding model to use",
            refresh_button=True,
        ),
        SecretStrInput(
            name="api_key",
            display_name="OpenAI API Key",
            info="Model Provider API key",
            required=False,
            show=True,
        ),
        # Watson-specific inputs
        MessageTextInput(
            name="project_id",
            display_name="Project ID",
            info="IBM watsonx.ai Project ID (required for IBM watsonx.ai)",
            show=False,
        ),
        # Amazon Bedrock-specific inputs
        DropdownInput(
            name="bedrock_region",
            display_name="AWS Region",
            options=AWS_REGIONS,
            value="us-east-1",
            info="The AWS region where your Bedrock resources are located. "
            "Uses IRSA for authentication — no explicit credentials needed.",
            show=False,
        ),
        IntInput(
            name="dimensions",
            display_name="Dimensions",
            info="The number of dimensions the resulting output embeddings should have. "
            "Only supported by certain models.",
            advanced=True,
        ),
        BoolInput(
            name="normalize",
            display_name="Normalize Embeddings",
            info="Whether to normalize the embedding vectors (Amazon Bedrock Titan v2 only).",
            value=False,
            advanced=True,
            show=False,
        ),
        IntInput(name="chunk_size", display_name="Chunk Size", advanced=True, value=1000),
        FloatInput(name="request_timeout", display_name="Request Timeout", advanced=True),
        IntInput(name="max_retries", display_name="Max Retries", advanced=True, value=3),
        BoolInput(name="show_progress_bar", display_name="Show Progress Bar", advanced=True),
        DictInput(
            name="model_kwargs",
            display_name="Model Kwargs",
            advanced=True,
            info="Additional keyword arguments to pass to the model.",
        ),
        IntInput(
            name="truncate_input_tokens",
            display_name="Truncate Input Tokens",
            advanced=True,
            value=200,
            show=False,
        ),
        BoolInput(
            name="input_text",
            display_name="Include the original text in the output",
            value=True,
            advanced=True,
            show=False,
        ),
    ]

    @staticmethod
    def fetch_ibm_models(base_url: str) -> list[str]:
        """Fetch available models from the watsonx.ai API."""
        try:
            endpoint = f"{base_url}/ml/v1/foundation_model_specs"
            params = {
                "version": "2024-09-16",
                "filters": "function_embedding,!lifecycle_withdrawn:and",
            }
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            models = [model["model_id"] for model in data.get("resources", [])]
            return sorted(models)
        except Exception:  # noqa: BLE001
            logger.exception("Error fetching models")
            return WATSONX_EMBEDDING_MODEL_NAMES

    def _get_provider(self) -> str:
        """Safely get the current provider value."""
        return getattr(self, "provider", "OpenAI") or "OpenAI"

    def build_embeddings(self) -> Embeddings:
        provider = self.provider
        model = self.model
        api_key = self.api_key
        api_base = self.api_base
        base_url_ibm_watsonx = self.base_url_ibm_watsonx
        dimensions = self.dimensions
        chunk_size = self.chunk_size
        request_timeout = self.request_timeout
        max_retries = self.max_retries
        show_progress_bar = self.show_progress_bar
        model_kwargs = self.model_kwargs or {}

        if provider == "OpenAI":
            if not api_key:
                msg = "OpenAI API key is required when using OpenAI provider"
                raise ValueError(msg)
            return OpenAIEmbeddings(
                model=model,
                dimensions=dimensions or None,
                base_url=api_base or None,
                api_key=api_key,
                chunk_size=chunk_size,
                max_retries=max_retries,
                timeout=request_timeout or None,
                show_progress_bar=show_progress_bar,
                model_kwargs=model_kwargs,
            )

        if provider == "Ollama":
            try:
                from langchain_ollama import OllamaEmbeddings
            except ImportError:
                try:
                    from langchain_community.embeddings import OllamaEmbeddings
                except ImportError:
                    msg = "Please install langchain-ollama: pip install langchain-ollama"
                    raise ImportError(msg) from None

            ollama_base_url = self.ollama_base_url or DEFAULT_OLLAMA_URL
            transformed_base_url = transform_localhost_url(ollama_base_url)

            if transformed_base_url and transformed_base_url.rstrip("/").endswith("/v1"):
                transformed_base_url = transformed_base_url.rstrip("/").removesuffix("/v1")
                logger.warning(
                    "Detected '/v1' suffix in base URL. The Ollama component uses the native Ollama API, "
                    "not the OpenAI-compatible API. The '/v1' suffix has been automatically removed. "
                    "If you want to use the OpenAI-compatible API, please use the OpenAI component instead. "
                    "Learn more at https://docs.ollama.com/openai#openai-compatibility"
                )

            return OllamaEmbeddings(
                model=model,
                base_url=transformed_base_url or DEFAULT_OLLAMA_URL,
                **model_kwargs,
            )

        if provider == "IBM watsonx.ai":
            try:
                from langchain_ibm import WatsonxEmbeddings
            except ImportError:
                msg = "Please install langchain-ibm: pip install langchain-ibm"
                raise ImportError(msg) from None

            if not api_key:
                msg = "IBM watsonx.ai API key is required when using IBM watsonx.ai provider"
                raise ValueError(msg)

            project_id = self.project_id
            if not project_id:
                msg = "Project ID is required for IBM watsonx.ai provider"
                raise ValueError(msg)

            from ibm_watsonx_ai import APIClient, Credentials

            credentials = Credentials(
                api_key=self.api_key,
                url=base_url_ibm_watsonx or "https://us-south.ml.cloud.ibm.com",
            )
            api_client = APIClient(credentials)

            params = {
                EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: self.truncate_input_tokens,
                EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": self.input_text},
            }

            return WatsonxEmbeddings(
                model_id=model,
                params=params,
                watsonx_client=api_client,
                project_id=project_id,
            )

        if provider == "Amazon Bedrock":
            try:
                from langchain_aws.embeddings import BedrockEmbeddings
            except ImportError:
                msg = "Please install langchain-aws: pip install langchain-aws"
                raise ImportError(msg) from None

            init_params: dict[str, Any] = {
                "model_id": model,
                "region_name": self.bedrock_region,
            }

            bedrock_model_kwargs: dict[str, Any] = {}
            if model_kwargs:
                bedrock_model_kwargs.update(model_kwargs)
            if dimensions:
                bedrock_model_kwargs["dimensions"] = dimensions
            if self.normalize:
                bedrock_model_kwargs["normalize"] = True
            if bedrock_model_kwargs:
                init_params["model_kwargs"] = bedrock_model_kwargs

            try:
                output = BedrockEmbeddings(**init_params)
            except Exception as e:
                msg = f"Could not initialize BedrockEmbeddings: {e}"
                raise ValueError(msg) from e

            return output

        msg = f"Unknown provider: {provider}"
        raise ValueError(msg)

    async def update_build_config(
        self, build_config: dotdict, field_value: Any, field_name: str | None = None
    ) -> dotdict:
        # ── Early exit: only process fields we care about ──
        provider = self._get_provider()

        # Block any field that would trigger Ollama localhost calls when not using Ollama
        if field_name not in ("provider", "base_url_ibm_watsonx") and provider != "Ollama":
            if field_name in ("ollama_base_url", "model", "api_key"):
                return build_config

        if field_name == "provider":
            # --- Reset ALL provider-specific fields ---
            build_config["ollama_base_url"]["show"] = False
            build_config["base_url_ibm_watsonx"]["show"] = False
            build_config["project_id"]["show"] = False
            build_config["truncate_input_tokens"]["show"] = False
            build_config["input_text"]["show"] = False
            build_config["bedrock_region"]["show"] = False
            build_config["normalize"]["show"] = False
            build_config["api_base"]["show"] = False
            build_config["api_key"]["show"] = False
            build_config["api_key"]["required"] = False
            build_config["api_key"]["value"] = ""

            # ── OpenAI ──
            if field_value == "OpenAI":
                build_config["model"]["options"] = OPENAI_EMBEDDING_MODEL_NAMES
                build_config["model"]["value"] = OPENAI_EMBEDDING_MODEL_NAMES[0]
                build_config["api_key"]["display_name"] = "OpenAI API Key"
                build_config["api_key"]["required"] = True
                build_config["api_key"]["show"] = True
                build_config["api_base"]["display_name"] = "OpenAI API Base URL"
                build_config["api_base"]["advanced"] = True
                build_config["api_base"]["show"] = True

            # ── Ollama ──
            elif field_value == "Ollama":
                build_config["ollama_base_url"]["show"] = True
                if not build_config["ollama_base_url"].get("value"):
                    build_config["ollama_base_url"]["value"] = DEFAULT_OLLAMA_URL
                build_config["api_key"]["display_name"] = "API Key (Optional)"

                ollama_url = build_config["ollama_base_url"].get("value") or DEFAULT_OLLAMA_URL
                if await is_valid_ollama_url(url=ollama_url):
                    try:
                        models = await get_ollama_models(
                            base_url_value=ollama_url,
                            desired_capability=DESIRED_CAPABILITY,
                            json_models_key=JSON_MODELS_KEY,
                            json_name_key=JSON_NAME_KEY,
                            json_capabilities_key=JSON_CAPABILITIES_KEY,
                        )
                        build_config["model"]["options"] = models
                        build_config["model"]["value"] = models[0] if models else ""
                    except ValueError:
                        build_config["model"]["options"] = []
                        build_config["model"]["value"] = ""
                else:
                    build_config["model"]["options"] = []
                    build_config["model"]["value"] = ""

            # ── IBM watsonx.ai ──
            elif field_value == "IBM watsonx.ai":
                ibm_models = self.fetch_ibm_models(base_url=self.base_url_ibm_watsonx)
                build_config["model"]["options"] = ibm_models
                build_config["model"]["value"] = ibm_models[0] if ibm_models else ""
                build_config["api_key"]["display_name"] = "IBM watsonx.ai API Key"
                build_config["api_key"]["required"] = True
                build_config["api_key"]["show"] = True
                build_config["base_url_ibm_watsonx"]["show"] = True
                build_config["project_id"]["show"] = True
                build_config["truncate_input_tokens"]["show"] = True
                build_config["input_text"]["show"] = True

            # ── Amazon Bedrock ──
            elif field_value == "Amazon Bedrock":
                build_config["model"]["options"] = AWS_EMBEDDING_MODEL_IDS
                build_config["model"]["value"] = AWS_EMBEDDING_MODEL_IDS[0]
                build_config["api_key"]["display_name"] = "API Key (Not required — uses IRSA)"
                build_config["bedrock_region"]["show"] = True
                build_config["normalize"]["show"] = True

        elif field_name == "base_url_ibm_watsonx":
            ibm_models = self.fetch_ibm_models(base_url=field_value)
            build_config["model"]["options"] = ibm_models
            build_config["model"]["value"] = ibm_models[0] if ibm_models else ""

        elif field_name == "ollama_base_url" and provider == "Ollama":
            ollama_url = field_value or self.ollama_base_url or DEFAULT_OLLAMA_URL
            if await is_valid_ollama_url(url=ollama_url):
                try:
                    models = await get_ollama_models(
                        base_url_value=ollama_url,
                        desired_capability=DESIRED_CAPABILITY,
                        json_models_key=JSON_MODELS_KEY,
                        json_name_key=JSON_NAME_KEY,
                        json_capabilities_key=JSON_CAPABILITIES_KEY,
                    )
                    build_config["model"]["options"] = models
                    build_config["model"]["value"] = models[0] if models else ""
                except ValueError:
                    await logger.awarning("Failed to fetch Ollama embedding models.")
                    build_config["model"]["options"] = []
                    build_config["model"]["value"] = ""

        elif field_name == "model" and provider == "Ollama":
            ollama_url = self.ollama_base_url or DEFAULT_OLLAMA_URL
            if await is_valid_ollama_url(url=ollama_url):
                try:
                    models = await get_ollama_models(
                        base_url_value=ollama_url,
                        desired_capability=DESIRED_CAPABILITY,
                        json_models_key=JSON_MODELS_KEY,
                        json_name_key=JSON_NAME_KEY,
                        json_capabilities_key=JSON_CAPABILITIES_KEY,
                    )
                    build_config["model"]["options"] = models
                except ValueError:
                    await logger.awarning("Failed to refresh Ollama embedding models.")
                    build_config["model"]["options"] = []

        return build_config