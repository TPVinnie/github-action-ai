import sys
from unittest.mock import MagicMock

# Mock transformers and torch before app import so CI does not need them installed
_mock_clf = MagicMock(return_value=[{"label": "POSITIVE", "score": 0.9998}])
_mock_transformers = MagicMock()
_mock_transformers.pipeline.return_value = _mock_clf
sys.modules["transformers"] = _mock_transformers
sys.modules["torch"] = MagicMock()

from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_label_and_score():
    response = client.post("/predict", json={"text": "I love this product!"})
    assert response.status_code == 200
    body = response.json()
    assert "label" in body
    assert "score" in body


def test_predict_empty_text_returns_400():
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 400
