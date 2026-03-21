import pytest
from unittest.mock import patch

from backend.app.vectorstore.factory import VectorStoreFactory


@patch("backend.app.vectorstore.elasticsearch_store.ElasticsearchStore")
def test_factory_creates_elasticsearch(mock_es_cls):
    """Factory uses lazy import — patch at the source module."""
    VectorStoreFactory.create("elasticsearch")
    mock_es_cls.assert_called_once()


def test_factory_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="Unknown vectorstore strategy"):
        VectorStoreFactory.create("pinecone")


@patch("backend.app.vectorstore.elasticsearch_store.ElasticsearchStore")
def test_factory_uses_config_default(mock_es_cls):
    VectorStoreFactory.create()
    mock_es_cls.assert_called_once()
