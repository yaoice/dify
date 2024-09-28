import os
from typing import Any
from unittest.mock import MagicMock

import pytest
from _pytest.monkeypatch import MonkeyPatch
from pymochow import MochowClient
from pymochow.model.database import Database
from pymochow.model.schema import Schema
from pymochow.model.table import AnnSearch, Partition, Row, Table

from core.rag.datasource.vdb.field import Field as vdb_Field


class AttrDict(dict):
    def __getattr__(self, item):
        return self.get(item)


class MockBaiduVectorDBClass:
    def list_databases(self) -> list[Database]:
        return [
            Database(
                conn=MagicMock(),
                database_name="test_database",
            )
        ]

    def database(self, database_name: str) -> Database:
        return Database(
            conn=MagicMock(),
            database_name=database_name,
        )

    def create_database(self, database_name: str) -> Database:
        return Database(
            conn=MagicMock(),
            database_name=database_name,
        )

    def table(self, table_name: str) -> Table:
        return Table(
            db=MagicMock(), name=table_name, replication=MagicMock(), partition=MagicMock(), schema=MagicMock()
        )

    def list_table(self) -> list:
        return [
            Table(
                db=MagicMock(),
                name="test_collection",
                replication=MagicMock(),
                partition=MagicMock(),
                schema=MagicMock(),
            )
        ]

    def drop_table(self, table_name: str):
        # delete call drop_table function does not return anything
        assert len([table for table in self.list_table() if table._table_name == table_name]) > 0

    def create_table(
        self, table_name, replication, partition, schema, enable_dynamic_field=False, description=None, config=None
    ):
        # _create_collection call create_table function does not return anything
        assert replication > 0
        assert table_name is not None
        assert description == "Collection For Dify"
        assert isinstance(schema, Schema) == True
        assert isinstance(partition, Partition) == True

    def upsert(self, rows: Any):
        # add_texts call upsert function does not return anything
        assert len(rows) > 0
        assert isinstance(rows, list) == True
        assert isinstance(rows[0], Row) == True

    def delete(self, primary_key: dict):
        # delete_by_ids call delete function does not return anything
        assert (vdb_Field.PRIMARY_KEY.value in primary_key) == True

    def select(self, filter, limit):
        result = [
            {
                vdb_Field.PRIMARY_KEY.value: "test_id_01",
                vdb_Field.CONTENT_KEY.value: "test_content",
                vdb_Field.GROUP_KEY.value: "test_group",
                vdb_Field.METADATA_KEY.value: '{\
                            "doc_id": "test_id_01",\
                            "doc_hash": "test_doc_hash",\
                            "document_id": "test_document_id",\
                            "dataset_id": "test_dataset_id"\
                        }',
            },
            {
                vdb_Field.PRIMARY_KEY.value: "test_id_02",
                vdb_Field.CONTENT_KEY.value: "test_content",
                vdb_Field.GROUP_KEY.value: "test_group",
                vdb_Field.METADATA_KEY.value: '{\
                            "doc_id": "test_id_02",\
                            "doc_hash": "test_doc_hash_02",\
                            "document_id": "test_document_id_02",\
                            "dataset_id": "test_dataset_id"\
                        }',
            },
        ]
        key, value = filter.split("==")
        rows = [row for row in result if row.get(key) == value.strip("'")]
        return AttrDict(
            {
                "code": 0,
                "msg": "Success",
                "rows": rows[:limit],
            }
        )

    def search(self, anns: AnnSearch):
        assert anns is not None
        result = [
            {
                "row": {
                    vdb_Field.PRIMARY_KEY.value: "test_id_01",
                    vdb_Field.CONTENT_KEY.value: "test_content",
                    vdb_Field.GROUP_KEY.value: "test_group",
                    vdb_Field.VECTOR.value: [1.001 * i for i in range(128)],
                    vdb_Field.METADATA_KEY.value: '{\
                                "doc_id": "test_id_01",\
                                "doc_hash": "test_doc_hash",\
                                "document_id": "test_document_id",\
                                "dataset_id": "test_dataset_id"\
                            }',
                },
                "distance": 0.0,
                "score": 1.0,
            },
            {
                "row": {
                    vdb_Field.PRIMARY_KEY.value: "test_id_02",
                    vdb_Field.CONTENT_KEY.value: "test_content2",
                    vdb_Field.GROUP_KEY.value: "test_group",
                    vdb_Field.VECTOR.value: [1.001 * i for i in range(128)],
                    vdb_Field.METADATA_KEY.value: '{\
                                "doc_id": "test_id_02",\
                                "doc_hash": "test_doc_hash",\
                                "document_id": "test_document_id",\
                                "dataset_id": "test_dataset_id"\
                            }',
                },
                "distance": 0.0,
                "score": 2.0,
            },
        ]

        rows = [item for item in result if item["row"].get(vdb_Field.VECTOR.value) == anns._vector_floats]
        return AttrDict(
            {
                "code": 0,
                "msg": "Success",
                "rows": rows,
            }
        )

    def query(self, primary_key: dict):
        if primary_key.get(vdb_Field.PRIMARY_KEY.value) == "exists_id":
            result = AttrDict(
                {
                    "code": 0,
                    "msg": "Success",
                    "row": {
                        vdb_Field.PRIMARY_KEY.value: primary_key,
                        vdb_Field.CONTENT_KEY.value: "test_content",
                        vdb_Field.GROUP_KEY.value: "test_group",
                        vdb_Field.METADATA_KEY.value: {
                            "doc_id": primary_key,
                            "doc_hash": "test_doc_hash",
                            "document_id": "test_document_id",
                            "dataset_id": "test_dataset_id",
                        },
                    },
                }
            )
        else:
            result = AttrDict({"code": 0, "msg": "Success", "row": {}})
        return result


MOCK = os.getenv("MOCK_SWITCH", "false").lower() == "true"


@pytest.fixture
def setup_baidu_vectordb_mock(monkeypatch: MonkeyPatch):
    if MOCK:
        monkeypatch.setattr(MochowClient, "list_databases", MockBaiduVectorDBClass.list_databases)
        monkeypatch.setattr(MochowClient, "database", MockBaiduVectorDBClass.database)
        monkeypatch.setattr(MochowClient, "create_database", MockBaiduVectorDBClass.create_database)
        monkeypatch.setattr(Database, "table", MockBaiduVectorDBClass.table)
        monkeypatch.setattr(Database, "list_table", MockBaiduVectorDBClass.list_table)
        monkeypatch.setattr(Database, "create_table", MockBaiduVectorDBClass.create_table)
        monkeypatch.setattr(Database, "drop_table", MockBaiduVectorDBClass.drop_table)
        monkeypatch.setattr(Table, "upsert", MockBaiduVectorDBClass.upsert)
        monkeypatch.setattr(Table, "select", MockBaiduVectorDBClass.select)
        monkeypatch.setattr(Table, "search", MockBaiduVectorDBClass.search)
        monkeypatch.setattr(Table, "query", MockBaiduVectorDBClass.query)
        monkeypatch.setattr(Table, "delete", MockBaiduVectorDBClass.delete)

    yield

    if MOCK:
        monkeypatch.undo()
