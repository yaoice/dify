import json
from typing import Any, Optional

from pydantic import BaseModel
from pymochow import MochowClient, exception
from pymochow.auth.bce_credentials import BceCredentials
from pymochow.configuration import Configuration
from pymochow.model import enum
from pymochow.model.database import Database
from pymochow.model.schema import AutoBuildRowCountIncrement, Field, HNSWParams, Schema, SecondaryIndex, VectorIndex
from pymochow.model.table import AnnSearch, HNSWSearchParams, Partition, Row

from configs import dify_config
from core.rag.datasource.entity.embedding import Embeddings
from core.rag.datasource.vdb.field import Field as vdb_Field
from core.rag.datasource.vdb.vector_base import BaseVector
from core.rag.datasource.vdb.vector_factory import AbstractVectorFactory
from core.rag.datasource.vdb.vector_type import VectorType
from core.rag.models.document import Document
from extensions.ext_database import db
from extensions.ext_redis import redis_client
from models.dataset import Dataset, DatasetCollectionBinding


class BaiduVectorDBConfig(BaseModel):
    account: str
    api_key: str
    endpoint: str
    database: str
    instance_id: str
    index_type: str = enum.IndexType.HNSW
    metric_type: str = enum.MetricType.L2
    partition_num: int = 1
    replicas: int = 1
    hnsw_params: tuple[int, int] = (32, 200)
    row_count_inc_ratio: tuple[int, float] = (10000, 0.3)


class BaiduVector(BaseVector):
    def __init__(self, collection_name: str, group_id: str, config: BaiduVectorDBConfig):
        super().__init__(collection_name)
        self._group_id = group_id
        self._client_config = config
        try:
            self._client = MochowClient(
                Configuration(
                    credentials=BceCredentials(config.account, config.api_key),
                    endpoint=config.endpoint,
                )
            )
        except exception.Error as e:
            raise ValueError(f"failed to initialize client: {e}")
        self._db = self._initialize_vector_database()

    def _initialize_vector_database(self) -> Database:
        db: Optional[Database] = None
        exists = False

        for database in self._client.list_databases():
            if database.database_name == self._client_config.database:
                exists = True
                break

        if exists:
            return self._client.database(self._client_config.database)

        try:
            db = self._client.create_database(database_name=self._client_config.database)
        except exception.Error as e:
            raise ValueError(f"Failed to create database {self._client_config.database}: {e}")

        return db

    def _has_collection(self) -> bool:
        collections = self._db.list_table()
        return any(collection.table_name == self._collection_name for collection in collections)

    def _create_collection(self, dimension: int):
        lock_name = f"vector_indexing_lock_{self._collection_name}"
        with redis_client.lock(lock_name, timeout=20):
            collection_exist_cache_key = f"vector_indexing_{self._collection_name}"
            if redis_client.get(collection_exist_cache_key):
                return

            if self._has_collection():
                return

            fields = [
                Field(
                    vdb_Field.PRIMARY_KEY.value,
                    enum.FieldType.STRING,
                    primary_key=True,
                    partition_key=True,
                    not_null=True,
                ),
                Field(vdb_Field.METADATA_KEY.value, enum.FieldType.STRING, not_null=True),
                Field(vdb_Field.GROUP_KEY.value, enum.FieldType.STRING, not_null=True),
                Field(vdb_Field.CONTENT_KEY.value, enum.FieldType.TEXT_GB18030, not_null=True),
                Field(vdb_Field.VECTOR.value, enum.FieldType.FLOAT_VECTOR, not_null=True, dimension=dimension),
            ]

            indexes = [
                VectorIndex(
                    index_name=f"{vdb_Field.VECTOR.value}_idx",
                    index_type=self._client_config.index_type,
                    field=vdb_Field.VECTOR.value,
                    metric_type=self._client_config.metric_type,
                    params=HNSWParams(*self._client_config.hnsw_params),
                    auto_build=True,
                    auto_build_index_policy=AutoBuildRowCountIncrement(*self._client_config.row_count_inc_ratio),
                ),
                SecondaryIndex(index_name=f"{vdb_Field.METADATA_KEY.value}_idx", field=vdb_Field.METADATA_KEY.value),
                SecondaryIndex(index_name=f"{vdb_Field.CONTENT_KEY.value}_idx", field=vdb_Field.CONTENT_KEY.value),
                SecondaryIndex(index_name=f"{vdb_Field.GROUP_KEY.value}_idx", field=vdb_Field.GROUP_KEY.value),
            ]
            try:
                self._db.create_table(
                    table_name=self._collection_name,
                    replication=self._client_config.replicas,
                    partition=Partition(self._client_config.partition_num),
                    schema=Schema(fields=fields, indexes=indexes),
                    description="Collection For Dify",
                )
            except exception.Error as e:
                raise ValueError(f"failed to create collection {self._collection_name}: {e}")
            redis_client.set(collection_exist_cache_key, 1, ex=3600)

    def get_type(self) -> str:
        return VectorType.BAIDU

    def create(self, texts: list[Document], embeddings: list[list[float]], **kwargs):
        dimension = len(embeddings[0])
        self._create_collection(dimension)
        self.add_texts(texts, embeddings, **kwargs)

    def add_texts(self, documents: list[Document], embeddings: list[list[float]], **kwargs):
        page_contents = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        docs = []

        for i, page_content in enumerate(page_contents):
            metadata = {}
            if metadatas is not None:
                for key, val in metadatas[i].items():
                    metadata[key] = val
            doc = Row(
                id=metadatas[i]["doc_id"],
                vector=embeddings[i] if embeddings else None,
                page_content=page_content,
                metadata=json.dumps(metadata),
                group_id=self._group_id,
            )
            docs.append(doc)

        self._db.table(self._collection_name).upsert(docs)

    def text_exists(self, id: str) -> bool:
        docs = self._db.table(self._collection_name).query(primary_key={vdb_Field.PRIMARY_KEY.value: id})
        if docs and docs.code == 0:
            if not docs.row:
                return False
            else:
                return True
        return False

    def delete_by_ids(self, ids: list[str]) -> None:
        for id in ids:
            self._db.table(self._collection_name).delete(primary_key={vdb_Field.PRIMARY_KEY.value: id})

    def get_ids_by_metadata_field(self, key: str, value: str):
        # Note: Metadata field value is an dict, but
        # baidu vectordb field not support json type
        results = self._db.table(self._collection_name).select(
            filter=f"{vdb_Field.GROUP_KEY.value}=='{self._group_id}'",
            # limit max value is 1000
            limit=1000,
        )

        if (results is None) or (results.code != 0) or (len(results.rows) == 0):
            return []

        ids = []
        for result in results.rows:
            metadata = result.get(vdb_Field.METADATA_KEY.value)
            if metadata is not None:
                metadata = json.loads(metadata)
                if metadata.get(key) == value:
                    primary_key = result.get(vdb_Field.PRIMARY_KEY.value)
                    if primary_key is not None:
                        ids.append(primary_key)
        return ids

    def delete_by_metadata_field(self, key: str, value: str) -> None:
        ids = self.get_ids_by_metadata_field(key, value)
        self.delete_by_ids(ids)

    def search_by_vector(self, query_vector: list[float], **kwargs: Any) -> list[Document]:
        results = self._db.table(self._collection_name).search(
            AnnSearch(
                vector_field=vdb_Field.VECTOR.value,
                vector_floats=query_vector,
                params=HNSWSearchParams(ef=kwargs.get("ef", 10), limit=kwargs.get("top_k", 50)),
            )
        )
        score_threshold = float(kwargs.get("score_threshold") or 0.0)
        return self._get_search_res(results, score_threshold)

    def _get_search_res(self, results, score_threshold):
        docs = []
        if (results is None) or (results.code != 0) or (len(results.rows) == 0):
            return docs

        for result in results.rows:
            metadata = result["row"].get(vdb_Field.METADATA_KEY.value)
            if metadata is not None:
                metadata = json.loads(metadata)
            score = result.get("score", 0.0)
            if score > score_threshold:
                metadata["score"] = score
                doc = Document(page_content=result["row"].get(vdb_Field.CONTENT_KEY.value), metadata=metadata)
                docs.append(doc)
        docs = sorted(docs, key=lambda x: x.metadata["score"], reverse=True)
        return docs

    def search_by_full_text(self, query: str, **kwargs: Any) -> list[Document]:
        return []

    def delete(self) -> None:
        if self._has_collection:
            self._db.drop_table(self._collection_name)


class BaiduVectorFactory(AbstractVectorFactory):
    def init_vector(self, dataset: Dataset, attributes: list, embeddings: Embeddings) -> BaiduVector:
        if dataset.collection_binding_id:
            dataset_collection_binding = (
                db.session.query(DatasetCollectionBinding)
                .filter(DatasetCollectionBinding.id == dataset.collection_binding_id)
                .one_or_none()
            )
            if dataset_collection_binding:
                collection_name = dataset_collection_binding.collection_name
            else:
                raise ValueError("Dataset Collection Bindings is not exist!")
        else:
            if dataset.index_struct_dict:
                class_prefix: str = dataset.index_struct_dict["vector_store"]["class_prefix"]
                collection_name = class_prefix
            else:
                dataset_id = dataset.id
                collection_name = Dataset.gen_collection_name_by_id(dataset_id)

        if not dataset.index_struct_dict:
            dataset.index_struct = json.dumps(self.gen_index_struct_dict(VectorType.BAIDU, collection_name))

        # handle optional params
        if dify_config.BAIDU_VECTOR_DB_ACCOUNT is None:
            raise ValueError("BAIDU_VECTOR_DB_ACCOUNT should not be None")
        if dify_config.BAIDU_VECTOR_DB_API_KEY is None:
            raise ValueError("BAIDU_VECTOR_DB_API_KEY should not be None")
        if dify_config.BAIDU_VECTOR_DB_ENDPOINT is None:
            raise ValueError("BAIDU_VECTOR_DB_ENDPOINT should not be None")
        if dify_config.BAIDU_VECTOR_DB_DATABASE is None:
            raise ValueError("BAIDU_VECTOR_DB_DATABASE should not be None")
        if dify_config.BAIDU_VECTOR_DB_INSTANCE_ID is None:
            raise ValueError("BAIDU_VECTOR_DB_INSTANCE_ID should not be None")
        return BaiduVector(
            collection_name=collection_name,
            group_id=dataset.id,
            config=BaiduVectorDBConfig(
                account=dify_config.BAIDU_VECTOR_DB_ACCOUNT,
                api_key=dify_config.BAIDU_VECTOR_DB_API_KEY,
                endpoint=dify_config.BAIDU_VECTOR_DB_ENDPOINT,
                database=dify_config.BAIDU_VECTOR_DB_DATABASE,
                instance_id=dify_config.BAIDU_VECTOR_DB_INSTANCE_ID,
            ),
        )
