from fastapi.testclient import TestClient


def test_list_libraries_returns_seeded_openmfp_libraries(client: TestClient) -> None:
    response = client.get("/libraries")

    assert response.status_code == 200
    names = {library["name"] for library in response.json()}
    assert names == {"portal-ui-lib", "portal-server-lib", "webcomponents", "create-micro-frontend"}


def test_list_libraries_filters_by_kind(client: TestClient) -> None:
    response = client.get("/libraries", params={"kind": "server"})

    assert response.status_code == 200
    assert [library["name"] for library in response.json()] == ["portal-server-lib"]


def test_list_libraries_rejects_unknown_kind(client: TestClient) -> None:
    response = client.get("/libraries", params={"kind": "mobile"})

    assert response.status_code == 422


def test_get_library_returns_library(client: TestClient) -> None:
    response = client.get("/libraries/portal-ui-lib")

    assert response.status_code == 200
    assert response.json() == {
        "name": "portal-ui-lib",
        "kind": "ui",
        "repository_url": "https://github.com/openmfp/portal-ui-lib",
    }


def test_get_library_returns_404_for_unknown_name(client: TestClient) -> None:
    response = client.get("/libraries/does-not-exist")

    assert response.status_code == 404
    assert response.json() == {"detail": "Library 'does-not-exist' not found"}


def test_create_library_adds_it_to_the_list(client: TestClient) -> None:
    payload = {
        "name": "example-ui",
        "kind": "ui",
        "repository_url": "https://github.com/example/example-ui",
    }

    created = client.post("/libraries", json=payload)
    fetched = client.get("/libraries/example-ui")

    assert created.status_code == 201
    assert fetched.json() == payload


def test_create_library_rejects_duplicate_name(client: TestClient) -> None:
    payload = {
        "name": "portal-ui-lib",
        "kind": "ui",
        "repository_url": "https://github.com/openmfp/portal-ui-lib",
    }

    response = client.post("/libraries", json=payload)

    assert response.status_code == 409


def test_create_library_validates_payload(client: TestClient) -> None:
    payload = {"name": "Not A Slug", "kind": "ui", "repository_url": "not-a-url"}

    response = client.post("/libraries", json=payload)

    assert response.status_code == 422
    invalid_fields = {error["loc"][-1] for error in response.json()["detail"]}
    assert invalid_fields == {"name", "repository_url"}
