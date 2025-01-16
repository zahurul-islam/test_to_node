import pytest
from fastapi.testclient import TestClient
from src.api.main import app

@pytest.fixture
def client():
    return TestClient(app)

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_generate_nodes(client):
    test_prompt = "Create a form with validation and submission"
    response = client.post(
        "/generate",
        json={"prompt": test_prompt}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "prompt" in data
    assert "nodes" in data
    assert isinstance(data["nodes"], str)
    assert len(data["nodes"]) > 0
