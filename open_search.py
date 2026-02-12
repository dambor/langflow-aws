"""
AWS OpenSearch Vector Store Component for Langflow.

Extends the existing OpenSearch component with AWS-native authentication:
  1. IRSA / IAM (SigV4) — credential-free on EKS via AWSV4SignerAuth
  2. Basic auth          — username/password
  3. JWT                 — Bearer token

On EKS with IRSA, boto3.Session().get_credentials() auto-discovers the
web identity token (AWS_ROLE_ARN + AWS_WEB_IDENTITY_TOKEN_FILE).
For local dev it falls back to ~/.aws/credentials or env vars.

Supports both Amazon OpenSearch Service (service='es') and
Amazon OpenSearch Serverless (service='aoss').

Requirements beyond Langflow defaults: None.
  opensearchpy, boto3 are already bundled.
"""

from __future__ import annotations

import json
import uuid
from typing import Any
from urllib.parse import urlparse

import boto3
from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection, helpers

from lfx.base.vectorstores.model import LCVectorStoreComponent, check_cached_vector_store
from lfx.base.vectorstores.vector_store_connection_decorator import vector_store_connection
from lfx.io import (
    BoolInput,
    DropdownInput,
    HandleInput,
    IntInput,
    MultilineInput,
    SecretStrInput,
    StrInput,
    TableInput,
)
from lfx.log import logger
from lfx.schema.data import Data


@vector_store_connection
class AWSOpenSearchVectorStoreComponent(LCVectorStoreComponent):
    """AWS OpenSearch Vector Store with IRSA / IAM SigV4 Auth.

    Supports Amazon OpenSearch Service and OpenSearch Serverless.
    Uses AWSV4SignerAuth from opensearch-py (bundled) for credential-free
    authentication on EKS via IRSA.
    """

    display_name: str = "AWS OpenSearch"
    icon: str = "Amazon"
    description: str = (
        "Store and search documents using Amazon OpenSearch Service with "
        "IAM/IRSA (SigV4) authentication. Supports hybrid semantic and keyword search."
    )

    default_keys: list[str] = [
        "opensearch_url",
        "index_name",
        *[i.name for i in LCVectorStoreComponent.inputs],
        "embedding",
        "vector_field",
        "number_of_results",
        "auth_mode",
        "aws_region",
        "aws_service",
        "username",
        "password",
        "jwt_token",
        "jwt_header",
        "bearer_prefix",
        "use_ssl",
        "verify_certs",
        "filter_expression",
        "engine",
        "space_type",
        "ef_construction",
        "m",
        "docs_metadata",
    ]

    inputs = [
        TableInput(
            name="docs_metadata",
            display_name="Document Metadata",
            info="Additional metadata key-value pairs added to all ingested documents.",
            table_schema=[
                {"name": "key", "display_name": "Key", "type": "str", "description": "Key name"},
                {"name": "value", "display_name": "Value", "type": "str", "description": "Value"},
            ],
            value=[],
            input_types=["Data"],
        ),
        StrInput(
            name="opensearch_url",
            display_name="OpenSearch Endpoint",
            value="https://search-mydomain.us-east-1.es.amazonaws.com",
            info=(
                "AWS OpenSearch domain endpoint "
                "(e.g., https://search-mydomain.us-east-1.es.amazonaws.com) or "
                "Serverless endpoint (e.g., https://xyz.us-east-1.aoss.amazonaws.com)."
            ),
        ),
        StrInput(
            name="index_name",
            display_name="Index Name",
            value="langflow",
            info="OpenSearch index where documents are stored. Created automatically if missing.",
        ),
        # ----- Auth mode (dynamic) -----
        DropdownInput(
            name="auth_mode",
            display_name="Authentication Mode",
            value="iam",
            options=["iam", "basic", "jwt"],
            info=(
                "iam: IRSA / IAM SigV4 (recommended on AWS — zero static creds on EKS). "
                "basic: Username/password. "
                "jwt: Bearer token."
            ),
            real_time_refresh=True,
        ),
        # ----- IAM / IRSA fields -----
        DropdownInput(
            name="aws_region",
            display_name="AWS Region",
            options=[
                "us-east-1", "us-east-2", "us-west-1", "us-west-2",
                "eu-west-1", "eu-west-2", "eu-central-1",
                "ap-southeast-1", "ap-southeast-2", "ap-northeast-1",
                "sa-east-1",
            ],
            value="us-east-1",
            info="AWS region of your OpenSearch domain. Used for SigV4 signing.",
        ),
        DropdownInput(
            name="aws_service",
            display_name="AWS Service Type",
            options=["es", "aoss"],
            value="es",
            info=(
                "'es' for Amazon OpenSearch Service (managed domains). "
                "'aoss' for Amazon OpenSearch Serverless collections."
            ),
        ),
        # ----- Basic auth fields -----
        StrInput(
            name="username",
            display_name="Username",
            value="admin",
            show=False,
        ),
        SecretStrInput(
            name="password",
            display_name="Password",
            value="admin",
            show=False,
        ),
        # ----- JWT fields -----
        SecretStrInput(
            name="jwt_token",
            display_name="JWT Token",
            value="",
            load_from_db=False,
            show=False,
            info="JSON Web Token for Bearer authentication.",
        ),
        StrInput(
            name="jwt_header",
            display_name="JWT Header Name",
            value="Authorization",
            show=False,
            advanced=True,
        ),
        BoolInput(
            name="bearer_prefix",
            display_name="Prefix 'Bearer '",
            value=True,
            show=False,
            advanced=True,
        ),
        # ----- Vector engine settings -----
        DropdownInput(
            name="engine",
            display_name="Vector Engine",
            options=["jvector", "nmslib", "faiss", "lucene"],
            value="nmslib",
            info=(
                "Vector search engine. Amazon OpenSearch Serverless (aoss) "
                "only supports 'nmslib' or 'faiss'."
            ),
            advanced=True,
        ),
        DropdownInput(
            name="space_type",
            display_name="Distance Metric",
            options=["l2", "l1", "cosinesimil", "linf", "innerproduct"],
            value="l2",
            advanced=True,
        ),
        IntInput(
            name="ef_construction",
            display_name="EF Construction",
            value=512,
            advanced=True,
        ),
        IntInput(
            name="m",
            display_name="M Parameter",
            value=16,
            advanced=True,
        ),
        *LCVectorStoreComponent.inputs,
        HandleInput(name="embedding", display_name="Embedding", input_types=["Embeddings"]),
        StrInput(
            name="vector_field",
            display_name="Vector Field Name",
            value="chunk_embedding",
            advanced=True,
        ),
        IntInput(
            name="number_of_results",
            display_name="Default Result Limit",
            value=10,
            advanced=True,
        ),
        MultilineInput(
            name="filter_expression",
            display_name="Search Filters (JSON)",
            value="",
            info=(
                "Optional JSON for search filtering.\n"
                'Format 1: {"filter": [{"term": {"filename":"doc.pdf"}}], "limit": 10, "score_threshold": 1.6}\n'
                'Format 2: {"data_sources":["file.pdf"], "owners":["user123"]}'
            ),
        ),
        # ----- TLS -----
        BoolInput(
            name="use_ssl",
            display_name="Use SSL/TLS",
            value=True,
            advanced=True,
        ),
        BoolInput(
            name="verify_certs",
            display_name="Verify SSL Certificates",
            value=True,
            advanced=True,
        ),
    ]

    # ------------------------------------------------------------------ #
    # Index mapping
    # ------------------------------------------------------------------ #

    def _default_text_mapping(
        self,
        dim: int,
        engine: str = "nmslib",
        space_type: str = "l2",
        ef_search: int = 512,
        ef_construction: int = 100,
        m: int = 16,
        vector_field: str = "vector_field",
    ) -> dict[str, Any]:
        return {
            "settings": {"index": {"knn": True, "knn.algo_param.ef_search": ef_search}},
            "mappings": {
                "properties": {
                    vector_field: {
                        "type": "knn_vector",
                        "dimension": dim,
                        "method": {
                            "name": "hnsw",
                            "space_type": space_type,
                            "engine": engine,
                            "parameters": {"ef_construction": ef_construction, "m": m},
                        },
                    }
                }
            },
        }

    def _validate_aoss_with_engines(self, *, is_aoss: bool, engine: str) -> None:
        if is_aoss and engine not in {"nmslib", "faiss"}:
            msg = "Amazon OpenSearch Serverless only supports 'nmslib' or 'faiss' engines."
            raise ValueError(msg)

    # ------------------------------------------------------------------ #
    # Auth / client — the IRSA-aware part
    # ------------------------------------------------------------------ #

    def _build_auth_kwargs(self) -> dict[str, Any]:
        """Build auth kwargs for the OpenSearch client based on auth_mode.

        For 'iam': uses AWSV4SignerAuth with boto3 credentials.
        On EKS with IRSA, boto3 auto-discovers the web identity token
        (AWS_ROLE_ARN + AWS_WEB_IDENTITY_TOKEN_FILE).
        For local dev, falls back to ~/.aws/credentials or env vars.
        """
        mode = (self.auth_mode or "iam").strip().lower()

        if mode == "iam":
            # IRSA: boto3 automatically picks up IRSA credentials from the pod.
            # No access keys needed when running on EKS with the annotated service account.
            credentials = boto3.Session().get_credentials()
            if credentials is None:
                msg = (
                    "No AWS credentials found. On EKS, ensure IRSA is configured "
                    "(service account annotated with eks.amazonaws.com/role-arn). "
                    "Locally, configure aws credentials via 'aws configure' or env vars."
                )
                raise ValueError(msg)

            service = getattr(self, "aws_service", "es") or "es"
            region = getattr(self, "aws_region", "us-east-1") or "us-east-1"
            auth = AWSV4SignerAuth(credentials, region, service)

            return {
                "http_auth": auth,
                "connection_class": RequestsHttpConnection,
            }

        if mode == "jwt":
            token = (self.jwt_token or "").strip()
            if not token:
                msg = "Auth mode is 'jwt' but no JWT token was provided."
                raise ValueError(msg)
            header_name = (self.jwt_header or "Authorization").strip()
            header_value = f"Bearer {token}" if self.bearer_prefix else token
            return {"headers": {header_name: header_value}}

        # basic
        user = (self.username or "").strip()
        pwd = (self.password or "").strip()
        if not user or not pwd:
            msg = "Auth mode is 'basic' but username/password are missing."
            raise ValueError(msg)
        return {"http_auth": (user, pwd)}

    def build_client(self) -> OpenSearch:
        auth_kwargs = self._build_auth_kwargs()

        # Parse the endpoint to extract host properly
        parsed = urlparse(self.opensearch_url)
        host = parsed.hostname or self.opensearch_url
        port = parsed.port or 443
        scheme = parsed.scheme or "https"

        client_kwargs = {
            "hosts": [{"host": host, "port": port}],
            "use_ssl": self.use_ssl if scheme == "https" else False,
            "verify_certs": self.verify_certs,
            "ssl_assert_hostname": False,
            "ssl_show_warn": False,
            **auth_kwargs,
        }

        # IAM mode already sets connection_class=RequestsHttpConnection
        # For other modes, set it if not already present
        if "connection_class" not in client_kwargs:
            client_kwargs["connection_class"] = RequestsHttpConnection

        return OpenSearch(**client_kwargs)

    # ------------------------------------------------------------------ #
    # Ingest
    # ------------------------------------------------------------------ #

    def _bulk_ingest_embeddings(
        self,
        client: OpenSearch,
        index_name: str,
        embeddings: list[list[float]],
        texts: list[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        vector_field: str = "vector_field",
        text_field: str = "text",
        mapping: dict | None = None,
        max_chunk_bytes: int | None = 1 * 1024 * 1024,
        *,
        is_aoss: bool = False,
    ) -> list[str]:
        requests = []
        return_ids = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            _id = ids[i] if ids else str(uuid.uuid4())
            request = {
                "_op_type": "index",
                "_index": index_name,
                vector_field: embeddings[i],
                text_field: text,
                **metadata,
            }
            if is_aoss:
                request["id"] = _id
            else:
                request["_id"] = _id
            requests.append(request)
            return_ids.append(_id)
        helpers.bulk(client, requests, max_chunk_bytes=max_chunk_bytes)
        return return_ids

    @check_cached_vector_store
    def build_vector_store(self) -> OpenSearch:
        self.log(self.ingest_data)
        client = self.build_client()
        self._add_documents_to_vector_store(client=client)
        return client

    def _add_documents_to_vector_store(self, client: OpenSearch) -> None:
        self.ingest_data = self._prepare_ingest_data()
        docs = self.ingest_data or []
        if not docs:
            self.log("No documents to ingest.")
            return

        texts = []
        metadatas = []

        additional_metadata = {}
        if hasattr(self, "docs_metadata") and self.docs_metadata:
            if isinstance(self.docs_metadata[-1], Data):
                self.docs_metadata = self.docs_metadata[-1].data
                additional_metadata.update(self.docs_metadata)
            else:
                for item in self.docs_metadata:
                    if isinstance(item, dict) and "key" in item and "value" in item:
                        additional_metadata[item["key"]] = item["value"]
        for key, value in additional_metadata.items():
            if value == "None":
                additional_metadata[key] = None

        for doc_obj in docs:
            data_copy = json.loads(doc_obj.model_dump_json())
            text = data_copy.pop(doc_obj.text_key, doc_obj.default_value)
            texts.append(text)
            data_copy.update(additional_metadata)
            metadatas.append(data_copy)

        if not self.embedding:
            msg = "Embedding handle is required to embed documents."
            raise ValueError(msg)

        vectors = self.embedding.embed_documents(texts)
        if not vectors:
            self.log("No vectors generated.")
            return

        dim = len(vectors[0]) if vectors else 768

        is_aoss = (getattr(self, "aws_service", "es") == "aoss")
        engine = getattr(self, "engine", "nmslib")
        self._validate_aoss_with_engines(is_aoss=is_aoss, engine=engine)

        space_type = getattr(self, "space_type", "l2")
        ef_construction = getattr(self, "ef_construction", 512)
        m = getattr(self, "m", 16)

        mapping = self._default_text_mapping(
            dim=dim,
            engine=engine,
            space_type=space_type,
            ef_construction=ef_construction,
            m=m,
            vector_field=self.vector_field,
        )

        self.log(f"Indexing {len(texts)} documents into '{self.index_name}'...")

        return_ids = self._bulk_ingest_embeddings(
            client=client,
            index_name=self.index_name,
            embeddings=vectors,
            texts=texts,
            metadatas=metadatas,
            vector_field=self.vector_field,
            text_field="text",
            mapping=mapping,
            is_aoss=is_aoss,
        )
        self.log(f"Successfully indexed {len(return_ids)} documents.")

    # ------------------------------------------------------------------ #
    # Search filter helpers
    # ------------------------------------------------------------------ #

    def _is_placeholder_term(self, term_obj: dict) -> bool:
        return any(v == "__IMPOSSIBLE_VALUE__" for v in term_obj.values())

    def _coerce_filter_clauses(self, filter_obj: dict | None) -> list[dict]:
        if not filter_obj:
            return []
        if isinstance(filter_obj, str):
            try:
                filter_obj = json.loads(filter_obj)
            except json.JSONDecodeError:
                return []

        if "filter" in filter_obj:
            raw = filter_obj["filter"]
            if isinstance(raw, dict):
                raw = [raw]
            clauses: list[dict] = []
            for f in raw or []:
                if "term" in f and isinstance(f["term"], dict) and not self._is_placeholder_term(f["term"]):
                    clauses.append(f)
                elif "terms" in f and isinstance(f["terms"], dict):
                    field, vals = next(iter(f["terms"].items()))
                    if isinstance(vals, list) and len(vals) > 0:
                        clauses.append(f)
            return clauses

        field_mapping = {
            "data_sources": "filename",
            "document_types": "mimetype",
            "owners": "owner",
        }
        context_clauses: list[dict] = []
        for k, values in filter_obj.items():
            if not isinstance(values, list):
                continue
            field = field_mapping.get(k, k)
            if len(values) == 0:
                context_clauses.append({"term": {field: "__IMPOSSIBLE_VALUE__"}})
            elif len(values) == 1:
                if values[0] != "__IMPOSSIBLE_VALUE__":
                    context_clauses.append({"term": {field: values[0]}})
            else:
                context_clauses.append({"terms": {field: values}})
        return context_clauses

    # ------------------------------------------------------------------ #
    # Hybrid search
    # ------------------------------------------------------------------ #

    def search(self, query: str | None = None) -> list[dict[str, Any]]:
        client = self.build_client()
        q = (query or "").strip()

        filter_obj = None
        if getattr(self, "filter_expression", "") and self.filter_expression.strip():
            try:
                filter_obj = json.loads(self.filter_expression)
            except json.JSONDecodeError as e:
                msg = f"Invalid filter_expression JSON: {e}"
                raise ValueError(msg) from e

        if not self.embedding:
            msg = "Embedding is required for hybrid search."
            raise ValueError(msg)

        vec = self.embedding.embed_query(q)
        filter_clauses = self._coerce_filter_clauses(filter_obj)

        limit = (filter_obj or {}).get("limit", self.number_of_results)
        score_threshold = (filter_obj or {}).get("score_threshold", 0)

        body = {
            "query": {
                "bool": {
                    "should": [
                        {
                            "knn": {
                                self.vector_field: {
                                    "vector": vec,
                                    "k": 10,
                                    "boost": 0.7,
                                }
                            }
                        },
                        {
                            "multi_match": {
                                "query": q,
                                "fields": ["text^2", "filename^1.5"],
                                "type": "best_fields",
                                "fuzziness": "AUTO",
                                "boost": 0.3,
                            }
                        },
                    ],
                    "minimum_should_match": 1,
                }
            },
            "aggs": {
                "data_sources": {"terms": {"field": "filename", "size": 20}},
                "document_types": {"terms": {"field": "mimetype", "size": 10}},
                "owners": {"terms": {"field": "owner", "size": 10}},
            },
            "_source": [
                "filename", "mimetype", "page", "text",
                "source_url", "owner", "allowed_users", "allowed_groups",
            ],
            "size": limit,
        }
        if filter_clauses:
            body["query"]["bool"]["filter"] = filter_clauses
        if isinstance(score_threshold, (int, float)) and score_threshold > 0:
            body["min_score"] = score_threshold

        resp = client.search(index=self.index_name, body=body)
        hits = resp.get("hits", {}).get("hits", [])
        return [
            {
                "page_content": hit["_source"].get("text", ""),
                "metadata": {k: v for k, v in hit["_source"].items() if k != "text"},
                "score": hit.get("_score"),
            }
            for hit in hits
        ]

    def search_documents(self) -> list[Data]:
        try:
            raw = self.search(self.search_query or "")
            return [Data(text=hit["page_content"], **hit["metadata"]) for hit in raw]
        except Exception as e:
            self.log(f"search_documents error: {e}")
            raise

    # ------------------------------------------------------------------ #
    # Dynamic UI — show/hide auth fields based on auth_mode
    # ------------------------------------------------------------------ #

    async def update_build_config(
        self, build_config: dict, field_value: str, field_name: str | None = None
    ) -> dict:
        try:
            if field_name == "auth_mode":
                mode = (field_value or "iam").strip().lower()
                is_iam = mode == "iam"
                is_basic = mode == "basic"
                is_jwt = mode == "jwt"

                # IAM fields
                build_config["aws_region"]["show"] = is_iam
                build_config["aws_service"]["show"] = is_iam
                build_config["aws_region"]["required"] = is_iam
                build_config["aws_service"]["required"] = is_iam

                # Basic fields
                build_config["username"]["show"] = is_basic
                build_config["password"]["show"] = is_basic
                build_config["username"]["required"] = is_basic
                build_config["password"]["required"] = is_basic

                # JWT fields
                build_config["jwt_token"]["show"] = is_jwt
                build_config["jwt_header"]["show"] = is_jwt
                build_config["bearer_prefix"]["show"] = is_jwt
                build_config["jwt_token"]["required"] = is_jwt

                # Clear irrelevant values
                if is_iam:
                    build_config["jwt_token"]["value"] = ""
                if is_jwt:
                    build_config["username"]["value"] = ""
                    build_config["password"]["value"] = ""

                return build_config

        except (KeyError, ValueError) as e:
            self.log(f"update_build_config error: {e}")

        return build_config