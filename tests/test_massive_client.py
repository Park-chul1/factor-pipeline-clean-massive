from factor_pipeline.massive_client import MassiveClient


class FakeResponse:
    status_code = 200
    text = ""

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


def test_massive_client_reuses_cached_get_response(tmp_path, monkeypatch):
    calls = []

    def fake_get(url, params, timeout):
        calls.append((url, params, timeout))
        return FakeResponse({"results": [{"call": len(calls)}]})

    monkeypatch.setattr("factor_pipeline.massive_client.requests.get", fake_get)

    client = MassiveClient("key-1", sleep_sec=0, cache_dir=tmp_path)
    first = client.get("/v3/reference/tickers", {"market": "stocks"})
    second = client.get("/v3/reference/tickers", {"market": "stocks"})

    assert first == second
    assert len(calls) == 1


def test_massive_client_cache_key_ignores_api_key(tmp_path, monkeypatch):
    calls = []

    def fake_get(url, params, timeout):
        calls.append((url, params, timeout))
        return FakeResponse({"results": [{"ticker": "AAPL"}]})

    monkeypatch.setattr("factor_pipeline.massive_client.requests.get", fake_get)

    client_a = MassiveClient("key-1", sleep_sec=0, cache_dir=tmp_path)
    client_b = MassiveClient("key-2", sleep_sec=0, cache_dir=tmp_path)

    assert client_a.get("/v3/reference/tickers", {"market": "stocks"}) == client_b.get(
        "/v3/reference/tickers",
        {"market": "stocks"},
    )
    assert len(calls) == 1


def test_massive_client_can_disable_cache(tmp_path, monkeypatch):
    calls = []

    def fake_get(url, params, timeout):
        calls.append((url, params, timeout))
        return FakeResponse({"results": [{"call": len(calls)}]})

    monkeypatch.setattr("factor_pipeline.massive_client.requests.get", fake_get)

    client = MassiveClient("key-1", sleep_sec=0, cache_dir=tmp_path, use_cache=False)
    first = client.get("/v3/reference/tickers", {"market": "stocks"})
    second = client.get("/v3/reference/tickers", {"market": "stocks"})

    assert first != second
    assert len(calls) == 2
