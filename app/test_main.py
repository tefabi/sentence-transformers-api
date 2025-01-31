from fastapi.testclient import TestClient
from models import EmbeddingModel, ModelSettings
from sentence_transformers import SentenceTransformer

from .main import app

client = TestClient(app)


def test_can_view_docs_through_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.url.path == "/docs"


def test_can_check_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"message": "Ok"}


def test_can_get_embeddings():
    body = {
        "text": "That no man knows, let it suffice",
        "model": EmbeddingModel.all_mpnet_base_v2.value,
    }

    model = SentenceTransformer(body["model"])
    embeddings = model.encode(body["text"]).tolist()

    response = client.post("/embeddings", json=body)
    assert response.status_code == 200
    assert response.json() == {
        "model": body["model"],
        "text": body["text"],
        "embeddings": embeddings,
    }
