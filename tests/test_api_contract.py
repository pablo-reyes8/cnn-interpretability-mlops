
def test_health_contract(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "device" in body
    assert body["classes"] == ["cat", "dog"]
    assert body["version"] == "test-model"

def test_predict_method_not_allowed(client):
    r = client.get("/predict")  
    assert r.status_code == 405

def test_predict_with_file_ok(client, dummy_image_bytes):
    files = {"file": ("cat.png", dummy_image_bytes, "image/png")}
    r = client.post("/predict", files=files)
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"label", "scores", "meta"}
    assert body["label"] in {"cat", "dog"}
    assert "cat" in body["scores"] and "dog" in body["scores"]
    assert body["meta"]["model_version"] == "test-model"
    assert body["meta"]["device"] == "cpu"
    assert body["meta"]["inference_ms"] >= 0
    assert body["meta"]["total_latency_ms"] >= body["meta"]["inference_ms"]

def test_predict_rejects_file_and_url_together(client, dummy_image_bytes):
    files = {"file": ("cat.png", dummy_image_bytes, "image/png")}
    r = client.post("/predict", files=files, data={"url": "https://example.com/cat.png"})
    assert r.status_code == 400
    assert "exactamente uno" in r.json()["detail"].lower()

def test_predict_with_bad_mime(client):
    files = {"file": ("note.txt", b"hello", "text/plain")}
    r = client.post("/predict", files=files)
    assert r.status_code in (400, 415)  

def test_predict_with_corrupt_image_bytes(client):
    files = {"file": ("cat.png", b"not-a-real-image", "image/png")}
    r = client.post("/predict", files=files)
    assert r.status_code == 400

def test_predict_advanced_gradcam_only(client, dummy_image_bytes):
    files = {"file": ("cat.png", dummy_image_bytes, "image/png")}
    data = {"what": "gradcam"}
    r = client.post("/predict/advanced", files=files, data=data)
    assert r.status_code == 200
    body = r.json()
    assert "artifacts" in body
    artifacts = body["artifacts"]
    assert ("gradcam_panel" in artifacts) or ("gradcam_panel_error" in artifacts)
