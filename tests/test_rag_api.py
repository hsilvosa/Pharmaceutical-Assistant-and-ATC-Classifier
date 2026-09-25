from fastapi.testclient import TestClient

from src.rag.api import create_app
from src.rag.config import Settings


def test_api_contracts(fixture_components, tmp_path) -> None:
    _, _, _, service = fixture_components
    settings = Settings(
        development_mode=True,
        evaluations_dir=tmp_path / "evaluations",
        baseline_path=tmp_path / "baseline.json",
    )
    client = TestClient(create_app(service, settings))
    assert client.get("/api/v1/health").json()["ready"] is True
    response = client.post(
        "/api/v1/query",
        json={"question": "Contraindicaciones de Ácido Ejemplo 100 mg", "language": "es"},
    )
    assert response.status_code == 200
    assert response.json()["citations"][0]["url"].startswith("https://")
    medicines = client.get("/api/v1/medicines/search", params={"q": "700001"})
    assert medicines.status_code == 200
    assert medicines.json()[0]["registration_number"] == "10001"
    assert client.get("/").status_code == 200


