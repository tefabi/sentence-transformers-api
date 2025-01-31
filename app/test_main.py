from fastapi.testclient import TestClient

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
