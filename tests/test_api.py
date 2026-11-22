import os

os.environ["SMARTRECRUIT_DB"] = "/tmp/smartrecruit_test.db"
if os.path.exists(os.environ["SMARTRECRUIT_DB"]):
    os.remove(os.environ["SMARTRECRUIT_DB"])

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402

client = TestClient(app)


def test_health():
    assert client.get("/api/health").json() == {"status": "ok"}


def test_offers_seeded_and_created():
    offers = client.get("/api/offers").json()
    assert len(offers) == 24
    created = client.post("/api/offers", json={"title": "Data Analyst",
                                                "text": "Build dashboards in SQL and Python for the sales team."})
    assert created.status_code == 201
    assert client.get(f"/api/offers/{created.json()['id']}").json()["title"] == "Data Analyst"


def test_upload_rejects_unknown_type():
    r = client.post("/api/candidates", files=[("files", ("cv.txt", b"hello", "text/plain"))])
    assert r.status_code == 415


def test_validation():
    assert client.post("/api/offers", json={"title": "x", "text": "short"}).status_code == 422
    assert client.get("/api/offers/9999").status_code == 404
