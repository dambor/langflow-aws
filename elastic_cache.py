"""
AWS ElastiCache (Valkey 8.2) Vector Store Component for Langflow.

Uses redis-py (bundled with Langflow) to connect to ElastiCache Valkey.
Supports three auth modes:
  1. IRSA / IAM Auth  — SigV4 pre-signed token via pod identity (no static creds)
  2. Password (AUTH)  — static password / RBAC token
  3. None             — no auth (VPC-only access)

Avoids the LangChain Redis VectorStore wrapper which uses TEXT fields
unsupported on ElastiCache Valkey.

Requirements beyond Langflow defaults: None.
  redis-py, numpy, botocore are already bundled.
"""

import json
import logging
import uuid
from typing import Any
from urllib.parse import ParseResult, urlencode, urlunparse

import numpy as np
import redis
from redis.commands.search.field import TagField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

from lfx.base.vectorstores.model import LCVectorStoreComponent, check_cached_vector_store
from lfx.helpers.data import docs_to_data
from lfx.io import BoolInput, DropdownInput, HandleInput, IntInput, SecretStrInput, StrInput
from lfx.schema.data import Data

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IRSA / IAM Auth — SigV4 credential provider for ElastiCache
# ---------------------------------------------------------------------------
# Uses botocore (already installed with Langflow via boto3/langchain-aws)
# to generate short-lived IAM auth tokens. On EKS with IRSA, botocore
# automatically picks up the web identity token from the pod's service
# account — zero static credentials needed.
# ---------------------------------------------------------------------------

class ElastiCacheIAMProvider(redis.CredentialProvider):
    """redis-py CredentialProvider that generates SigV4 IAM auth tokens.

    When running on EKS with IRSA, botocore automatically uses the
    web identity token injected into the pod (AWS_ROLE_ARN and
    AWS_WEB_IDENTITY_TOKEN_FILE env vars). No access keys required.

    For local dev, it falls back to ~/.aws/credentials or env vars.
    """

    def __init__(self, user: str, cache_name: str, region: str = "us-east-1"):
        self.user = user
        self.cache_name = cache_name
        self.region = region

        import botocore.session
        from botocore.model import ServiceId
        from botocore.signers import RequestSigner

        session = botocore.session.get_session()
        self.request_signer = RequestSigner(
            ServiceId("elasticache"),
            self.region,
            "elasticache",
            "v4",
            session.get_credentials(),
            session.get_component("event_emitter"),
        )

    def get_credentials(self):
        """Generate a short-lived IAM auth token (valid 15 min)."""
        query_params = {"Action": "connect", "User": self.user}
        url = urlunparse(
            ParseResult(
                scheme="https",
                netloc=self.cache_name,
                path="/",
                query=urlencode(query_params),
                params="",
                fragment="",
            )
        )
        signed_url = self.request_signer.generate_presigned_url(
            {"method": "GET", "url": url, "body": {}, "headers": {}, "context": {}},
            operation_name="connect",
            expires_in=900,
            region_name=self.region,
        )
        # Strip the protocol — ElastiCache expects just the signed portion
        token = signed_url.removeprefix("https://")
        return self.user, token


# ---------------------------------------------------------------------------
# Langflow Component
# ---------------------------------------------------------------------------

class ElastiCacheValkeyVectorStoreComponent(LCVectorStoreComponent):
    """Vector Store using AWS ElastiCache for Valkey 8.2+ with native vector search."""

    display_name: str = "AWS ElastiCache (Valkey)"
    description: str = (
        "Vector Store using AWS ElastiCache for Valkey 8.2+ with native vector search. "
        "Supports IRSA (IAM Auth), password auth, or no-auth. "
    )
    name = "ElastiCacheValkey"
    icon = "Amazon"

    inputs = [
        StrInput(
            name="elasticache_endpoint",
            display_name="ElastiCache Cluster Endpoint",
            info="Discovery endpoint for your ElastiCache Valkey cluster "
            "(e.g., mycluster.cnxa6h.clustercfg.use1.cache.amazonaws.com)",
            required=True,
        ),
        IntInput(
            name="elasticache_port",
            display_name="Port",
            value=6379,
            info="ElastiCache cluster port.",
            advanced=True,
        ),
        DropdownInput(
            name="auth_mode",
            display_name="Auth Mode",
            options=["iam", "password", "none"],
            value="iam",
            info=(
                "iam: IRSA / IAM Auth via SigV4 (recommended on EKS, requires TLS). "
                "password: Static AUTH token or RBAC password. "
                "none: No authentication (VPC-only access)."
            ),
        ),
        StrInput(
            name="iam_user_id",
            display_name="IAM User ID",
            info="ElastiCache IAM user ID (must match the user-id created with "
            "'aws elasticache create-user --authentication-mode Type=iam'). "
            "Only used when Auth Mode is 'iam'.",
            value="iam-user-01",
        ),
        StrInput(
            name="cache_name",
            display_name="Cache / Replication Group Name",
            info="The replication-group-id or serverless-cache-name. "
            "Used to generate the IAM auth token. Only for 'iam' auth mode.",
        ),
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
            info="AWS region for IAM auth token signing. Only for 'iam' auth mode.",
        ),
        SecretStrInput(
            name="elasticache_password",
            display_name="Password",
            info="Static AUTH password or RBAC token. Only for 'password' auth mode.",
            required=False,
        ),
        StrInput(
            name="elasticache_username",
            display_name="Username",
            info="RBAC username. Only for 'password' auth mode.",
            required=False,
            advanced=True,
        ),
        BoolInput(
            name="use_tls",
            display_name="Use TLS",
            value=True,
            info="Enable TLS. Required for IAM auth. Recommended for all modes.",
        ),
        BoolInput(
            name="cluster_mode",
            display_name="Cluster Mode",
            value=True,
            info="Use RedisCluster client (multi-shard). Disable for single-node.",
        ),
        StrInput(
            name="index_name",
            display_name="Vector Index Name",
            value="langflow_vector_idx",
            info="Name of the vector search index in Valkey.",
        ),
        StrInput(
            name="key_prefix",
            display_name="Key Prefix",
            value="doc:",
            info="Prefix for document hash keys.",
            advanced=True,
        ),
        IntInput(
            name="vector_dimensions",
            display_name="Vector Dimensions",
            value=1536,
            info="Must match your embedding model "
            "(1536=OpenAI, 1024=Titan v2, 768=Cohere).",
        ),
        DropdownInput(
            name="distance_metric",
            display_name="Distance Metric",
            options=["COSINE", "L2", "IP"],
            value="COSINE",
            advanced=True,
        ),
        DropdownInput(
            name="index_algorithm",
            display_name="Index Algorithm",
            options=["HNSW", "FLAT"],
            value="HNSW",
            advanced=True,
        ),
        IntInput(name="hnsw_m", display_name="HNSW M", value=16, advanced=True),
        IntInput(name="hnsw_ef_construction", display_name="HNSW EF Construction", value=200, advanced=True),
        IntInput(name="hnsw_ef_runtime", display_name="HNSW EF Runtime", value=100, advanced=True),
        IntInput(name="chunk_size", display_name="Chunk Size", value=1000, advanced=True),
        IntInput(name="chunk_overlap", display_name="Chunk Overlap", value=0, advanced=True),
        *LCVectorStoreComponent.inputs,
        IntInput(
            name="number_of_results",
            display_name="Number of Results",
            value=4,
            advanced=True,
        ),
        HandleInput(name="embedding", display_name="Embedding", input_types=["Embeddings"]),
    ]

    def _get_client(self) -> Any:
        """Create a redis-py client with the selected auth mode."""
        conn_kwargs: dict[str, Any] = {
            "host": self.elasticache_endpoint,
            "port": self.elasticache_port,
            "decode_responses": False,
        }

        if self.use_tls:
            conn_kwargs["ssl"] = True
            conn_kwargs["ssl_cert_reqs"] = None

        # --- Auth modes ---
        if self.auth_mode == "iam":
            # IRSA path: botocore picks up credentials from the pod identity
            # (AWS_ROLE_ARN + AWS_WEB_IDENTITY_TOKEN_FILE) automatically.
            # For local dev, it uses ~/.aws/credentials or env vars.
            if not self.use_tls:
                logger.warning("IAM auth requires TLS. Forcing use_tls=True.")
                conn_kwargs["ssl"] = True
                conn_kwargs["ssl_cert_reqs"] = None

            creds_provider = ElastiCacheIAMProvider(
                user=self.iam_user_id,
                cache_name=self.cache_name,
                region=self.aws_region,
            )
            conn_kwargs["credential_provider"] = creds_provider

        elif self.auth_mode == "password":
            if self.elasticache_password:
                conn_kwargs["password"] = self.elasticache_password
            if self.elasticache_username:
                conn_kwargs["username"] = self.elasticache_username

        # auth_mode == "none" → no credentials

        if self.cluster_mode:
            return redis.RedisCluster(**conn_kwargs)
        return redis.Redis(**conn_kwargs)

    def _ensure_index(self, client: Any) -> None:
        """Create the vector index if it doesn't exist."""
        try:
            client.ft(index_name=self.index_name).info()
            logger.info(f"Index '{self.index_name}' already exists.")
            return
        except Exception:
            logger.info(f"Creating index '{self.index_name}'...")

        algo_params: dict[str, Any] = {
            "TYPE": "FLOAT32",
            "DIM": self.vector_dimensions,
            "DISTANCE_METRIC": self.distance_metric,
        }
        if self.index_algorithm == "HNSW":
            algo_params["M"] = self.hnsw_m
            algo_params["EF_CONSTRUCTION"] = self.hnsw_ef_construction
            algo_params["EF_RUNTIME"] = self.hnsw_ef_runtime

        # Only vector + tag fields. NO TextField — Valkey rejects TEXT.
        fields = [
            VectorField("embedding", self.index_algorithm, algo_params),
            TagField("source", separator="|"),
        ]

        definition = IndexDefinition(
            prefix=[self.key_prefix],
            index_type=IndexType.HASH,
        )

        client.ft(index_name=self.index_name).create_index(fields, definition=definition)
        logger.info(f"Index '{self.index_name}' created.")

    @staticmethod
    def _to_bytes(vector: list[float]) -> bytes:
        return np.array(vector, dtype=np.float32).tobytes()

    def _ingest_documents(self, client: Any, documents: list[Document]) -> int:
        splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = splitter.split_documents(documents)
        if not chunks:
            return 0

        texts = [c.page_content for c in chunks]
        embeddings = self.embedding.embed_documents(texts)

        pipe = client.pipeline(transaction=False)
        for chunk, emb in zip(chunks, embeddings):
            key = f"{self.key_prefix}{uuid.uuid4().hex[:12]}"
            source = chunk.metadata.get("source", "langflow")
            meta_json = json.dumps(chunk.metadata) if chunk.metadata else "{}"

            pipe.hset(key, mapping={
                "embedding": self._to_bytes(emb),
                "content": chunk.page_content.encode("utf-8"),
                "source": source.encode("utf-8") if isinstance(source, str) else source,
                "metadata": meta_json.encode("utf-8"),
            })

        pipe.execute()
        logger.info(f"Ingested {len(chunks)} chunks.")
        return len(chunks)

    @check_cached_vector_store
    def build_vector_store(self) -> Any:
        self.ingest_data = self._prepare_ingest_data()

        documents = []
        for _input in self.ingest_data or []:
            if isinstance(_input, Data):
                documents.append(_input.to_lc_document())
            else:
                documents.append(_input)

        client = self._get_client()
        self._ensure_index(client)

        if documents:
            self._ingest_documents(client, documents)

        return _ValkeyVectorStoreWrapper(
            client=client,
            index_name=self.index_name,
            embedding=self.embedding,
            distance_metric=self.distance_metric,
        )

    def search_documents(self) -> list[Data]:
        vector_store = self.build_vector_store()

        if self.search_query and isinstance(self.search_query, str) and self.search_query.strip():
            docs = vector_store.similarity_search(
                query=self.search_query,
                k=self.number_of_results,
            )
            data = docs_to_data(docs)
            self.status = data
            return data
        return []


# ---------------------------------------------------------------------------
# Wrapper providing LangChain-compatible similarity_search()
# ---------------------------------------------------------------------------

class _ValkeyVectorStoreWrapper:
    """Wraps native Valkey FT.SEARCH in a LangChain-compatible interface."""

    def __init__(self, client: Any, index_name: str, embedding: Any, distance_metric: str = "COSINE"):
        self.client = client
        self.index_name = index_name
        self.embedding = embedding
        self.distance_metric = distance_metric

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        query_vector = self.embedding.embed_query(query)
        query_bytes = np.array(query_vector, dtype=np.float32).tobytes()

        q = (
            Query(f"*=>[KNN {k} @embedding $vec as score]")
            .return_fields("content", "metadata", "source", "score")
            .sort_by("score")
            .dialect(2)
        )

        results = self.client.ft(index_name=self.index_name).search(
            q, query_params={"vec": query_bytes}
        )

        documents = []
        for doc in results.docs:
            content = getattr(doc, "content", b"")
            if isinstance(content, bytes):
                content = content.decode("utf-8")

            metadata = {}
            raw_meta = getattr(doc, "metadata", None)
            if raw_meta:
                if isinstance(raw_meta, bytes):
                    raw_meta = raw_meta.decode("utf-8")
                try:
                    metadata = json.loads(raw_meta)
                except (json.JSONDecodeError, TypeError):
                    metadata = {"raw_metadata": raw_meta}

            raw_score = getattr(doc, "score", None)
            if raw_score is not None:
                if isinstance(raw_score, bytes):
                    raw_score = raw_score.decode("utf-8")
                metadata["score"] = float(raw_score)

            raw_source = getattr(doc, "source", None)
            if raw_source:
                if isinstance(raw_source, bytes):
                    raw_source = raw_source.decode("utf-8")
                metadata["source"] = raw_source

            documents.append(Document(page_content=content, metadata=metadata))

        return documents

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        docs = self.similarity_search(query, k=k, **kwargs)
        return [(doc, doc.metadata.get("score", 0.0)) for doc in docs]

    def as_retriever(self, **kwargs: Any):
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.callbacks import CallbackManagerForRetrieverRun

        wrapper = self
        search_k = kwargs.get("search_kwargs", {}).get("k", 4)

        class _ValkeyRetriever(BaseRetriever):
            def _get_relevant_documents(
                self, query: str, *, run_manager: CallbackManagerForRetrieverRun
            ) -> list[Document]:
                return wrapper.similarity_search(query, k=search_k)

        return _ValkeyRetriever()