import pytest

from core.rag.datasource.vdb.baidu.baidu_vector import BaiduVector, BaiduVectorDBConfig
from core.rag.models.document import Document
from tests.integration_tests.vdb.__mock.baidu_vectordb import setup_baidu_vectordb_mock
from tests.integration_tests.vdb.test_vector_store import setup_mock_redis


@pytest.mark.parametrize("setup_baidu_vectordb_mock", [["none"]], indirect=True)
def test_init_without_database_created(setup_baidu_vectordb_mock):
    vector = BaiduVector(
        collection_name="test_collection",
        group_id="test_group",
        config=BaiduVectorDBConfig(
            account="test_account",
            api_key="test_api_key",
            database="test_database",
            endpoint="test_endpoint",
            instance_id="test_instance_id",
        ),
    )

    assert isinstance(vector, BaiduVector)
    assert vector._client is not None
    assert vector._db is not None


@pytest.mark.parametrize("setup_baidu_vectordb_mock", [["none"]], indirect=True)
def test_init_with_database_created(setup_baidu_vectordb_mock):
    vector = BaiduVector(
        collection_name="test_collection",
        group_id="test_group",
        config=BaiduVectorDBConfig(
            account="test_account",
            api_key="test_api_key",
            database="new_database",
            endpoint="test_endpoint",
            instance_id="test_instance_id",
        ),
    )

    assert isinstance(vector, BaiduVector)
    assert vector._client is not None
    assert vector._db is not None
    assert vector._db.database_name == "new_database"


@pytest.mark.parametrize("setup_baidu_vectordb_mock", [["none"]], indirect=True)
def test_add_texts(setup_baidu_vectordb_mock):
    vector = BaiduVector(
        collection_name="test_collection",
        group_id="test_group",
        config=BaiduVectorDBConfig(
            account="test_account",
            api_key="test_api_key",
            database="new_database",
            endpoint="test_endpoint",
            instance_id="test_instance_id",
        ),
    )

    documents = [
        Document(page_content="test_content_1", metadata={"doc_id": "1"}),
        Document(page_content="test_content_2", metadata={"doc_id": "2"}),
        Document(page_content="test_content_3", metadata={"doc_id": "3"}),
    ]
    embeddings = [[1.001 * i for i in range(128)] for _ in range(len(documents))]
    vector.add_texts(documents=documents, embeddings=embeddings)


@pytest.mark.parametrize(("setup_baidu_vectordb_mock", "setup_mock_redis"), [("none", "none")], indirect=True)
def test_create(setup_baidu_vectordb_mock, setup_mock_redis):
    vector = BaiduVector(
        collection_name="test_collection",
        group_id="test_group",
        config=BaiduVectorDBConfig(
            account="test_account",
            api_key="test_api_key",
            database="new_database",
            endpoint="test_endpoint",
            instance_id="test_instance_id",
        ),
    )
    texts = [
        Document(page_content="test_content_1", metadata={"doc_id": "1"}),
        Document(page_content="test_content_2", metadata={"doc_id": "2"}),
        Document(page_content="test_content_3", metadata={"doc_id": "3"}),
    ]

    embeddings = [[1.001 * i for i in range(128)] for _ in range(len(texts))]
    vector.create(texts=texts, embeddings=embeddings)


@pytest.mark.parametrize("setup_baidu_vectordb_mock", [["none"]], indirect=True)
def test_text_exists(setup_baidu_vectordb_mock):
    vector = BaiduVector(
        collection_name="test_collection",
        group_id="test_group",
        config=BaiduVectorDBConfig(
            account="test_account",
            api_key="test_api_key",
            database="new_database",
            endpoint="test_endpoint",
            instance_id="test_instance_id",
        ),
    )

    id = "exists_id"
    assert vector.text_exists(id) == True

    id = "not_exists_id"
    assert vector.text_exists(id) == False


@pytest.mark.parametrize("setup_baidu_vectordb_mock", [["none"]], indirect=True)
def test_delete_by_ids(setup_baidu_vectordb_mock):
    vector = BaiduVector(
        collection_name="test_collection",
        group_id="test_group",
        config=BaiduVectorDBConfig(
            account="test_account",
            api_key="test_api_key",
            database="new_database",
            endpoint="test_endpoint",
            instance_id="test_instance_id",
        ),
    )

    vector.delete_by_ids(["exists_id"])


@pytest.mark.parametrize("setup_baidu_vectordb_mock", [["none"]], indirect=True)
def test_get_ids_by_metadata_field(setup_baidu_vectordb_mock):
    vector = BaiduVector(
        collection_name="test_collection",
        group_id="test_group",
        config=BaiduVectorDBConfig(
            account="test_account",
            api_key="test_api_key",
            database="new_database",
            endpoint="test_endpoint",
            instance_id="test_instance_id",
        ),
    )

    ids = vector.get_ids_by_metadata_field("dataset_id", "test_dataset_id")
    assert len(ids) == 2

    ids = vector.get_ids_by_metadata_field("doc_id", "test_id_01")
    assert len(ids) == 1

    ids = vector.get_ids_by_metadata_field("dataset_id", "no_exits_dataset_id")
    assert len(ids) == 0


@pytest.mark.parametrize("setup_baidu_vectordb_mock", [["none"]], indirect=True)
def test_delete_by_metadata_field(setup_baidu_vectordb_mock):
    vector = BaiduVector(
        collection_name="test_collection",
        group_id="test_group",
        config=BaiduVectorDBConfig(
            account="test_account",
            api_key="test_api_key",
            database="new_database",
            endpoint="test_endpoint",
            instance_id="test_instance_id",
        ),
    )

    vector.delete_by_metadata_field("dataset_id", "test_dataset_id")


@pytest.mark.parametrize("setup_baidu_vectordb_mock", [["none"]], indirect=True)
def test_search_by_vector(setup_baidu_vectordb_mock):
    vector = BaiduVector(
        collection_name="test_collection",
        group_id="test_group",
        config=BaiduVectorDBConfig(
            account="test_account",
            api_key="test_api_key",
            database="new_database",
            endpoint="test_endpoint",
            instance_id="test_instance_id",
        ),
    )

    # set score_threshold higher than the vector actual score
    docs = vector.search_by_vector(
        query_vector=[1.001 * i for i in range(128)], **{"ef": 10, "top_k": 50, "score_threshold": 3.0}
    )
    assert len(docs) == 0

    # without score_threshold
    docs = vector.search_by_vector(query_vector=[1.001 * i for i in range(128)], **{"ef": 10, "top_k": 50})
    assert len(docs) > 0

    # search vector not exists
    docs = vector.search_by_vector(query_vector=[2.001 * i for i in range(128)], **{"ef": 10, "top_k": 50})
    assert len(docs) == 0


@pytest.mark.parametrize("setup_baidu_vectordb_mock", [["none"]], indirect=True)
def test_search_by_full_text(setup_baidu_vectordb_mock):
    vector = BaiduVector(
        collection_name="test_collection",
        group_id="test_group",
        config=BaiduVectorDBConfig(
            account="test_account",
            api_key="test_api_key",
            database="new_database",
            endpoint="test_endpoint",
            instance_id="test_instance_id",
        ),
    )

    assert len(vector.search_by_full_text("")) == 0


@pytest.mark.parametrize("setup_baidu_vectordb_mock", [["none"]], indirect=True)
def test_delete(setup_baidu_vectordb_mock):
    vector = BaiduVector(
        collection_name="test_collection",
        group_id="test_group",
        config=BaiduVectorDBConfig(
            account="test_account",
            api_key="test_api_key",
            database="new_database",
            endpoint="test_endpoint",
            instance_id="test_instance_id",
        ),
    )

    vector.delete()
