URL = "/api/v1/summarize"
BODY = {"github_url": "https://github.com/owner/repo"}


def test_missing_api_key_returns_401(client):
    response = client.post(URL, json=BODY)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "ApiKey"


def test_wrong_api_key_returns_401(client):
    response = client.post(URL, json=BODY, headers={"X-API-Key": "wrong"})
    assert response.status_code == 401
