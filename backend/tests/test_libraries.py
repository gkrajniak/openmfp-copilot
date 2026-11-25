from fastapi.testclient import TestClient


def test_list_libraries_returns_known_openmfp_and_platform_mesh_libraries(
    client: TestClient,
) -> None:
    response = client.get("/libraries")

    assert response.status_code == 200
    ids = {f"{library['organization']}/{library['name']}" for library in response.json()}
    assert ids == {
        "openmfp/portal-ui-lib",
        "openmfp/portal-server-lib",
        "openmfp/webcomponents",
        "openmfp/create-micro-frontend",
        "openmfp/portal",
        "platform-mesh/portal-ui-lib",
        "platform-mesh/portal-server-lib",
        "platform-mesh/portal",
    }


def test_list_libraries_filters_by_organization(client: TestClient) -> None:
    response = client.get("/libraries", params={"organization": "platform-mesh"})

    assert response.status_code == 200
    assert {library["name"] for library in response.json()} == {
        "portal-ui-lib",
        "portal-server-lib",
        "portal",
    }


def test_list_libraries_filters_by_kind(client: TestClient) -> None:
    response = client.get("/libraries", params={"kind": "app"})

    assert response.status_code == 200
    assert {library["organization"] for library in response.json()} == {
        "openmfp",
        "platform-mesh",
    }


def test_list_libraries_combines_organization_and_kind_filters(client: TestClient) -> None:
    response = client.get("/libraries", params={"organization": "openmfp", "kind": "server"})

    assert response.status_code == 200
    assert [library["name"] for library in response.json()] == ["portal-server-lib"]


def test_list_libraries_rejects_unknown_kind(client: TestClient) -> None:
    response = client.get("/libraries", params={"kind": "mobile"})

    assert response.status_code == 422


def test_list_libraries_rejects_unknown_organization(client: TestClient) -> None:
    response = client.get("/libraries", params={"organization": "unknown-org"})

    assert response.status_code == 422


def test_get_library_distinguishes_same_name_across_organizations(client: TestClient) -> None:
    openmfp = client.get("/libraries/openmfp/portal-ui-lib")
    platform_mesh = client.get("/libraries/platform-mesh/portal-ui-lib")

    assert openmfp.status_code == 200
    assert openmfp.json() == {
        "organization": "openmfp",
        "name": "portal-ui-lib",
        "kind": "ui",
        "repository_url": "https://github.com/openmfp/portal-ui-lib",
    }
    assert platform_mesh.status_code == 200
    assert platform_mesh.json()["repository_url"] == (
        "https://github.com/platform-mesh/portal-ui-lib"
    )


def test_get_library_returns_404_for_unknown_name(client: TestClient) -> None:
    response = client.get("/libraries/openmfp/does-not-exist")

    assert response.status_code == 404
    assert response.json() == {"detail": "Library 'openmfp/does-not-exist' not found"}


def test_get_library_returns_404_for_library_missing_in_that_organization(
    client: TestClient,
) -> None:
    response = client.get("/libraries/platform-mesh/webcomponents")

    assert response.status_code == 404


def test_create_library_adds_it_to_the_list(client: TestClient) -> None:
    payload = {
        "organization": "platform-mesh",
        "name": "marketplace-ui",
        "kind": "ui",
        "repository_url": "https://github.com/platform-mesh/marketplace-ui",
    }

    created = client.post("/libraries", json=payload)
    fetched = client.get("/libraries/platform-mesh/marketplace-ui")

    assert created.status_code == 201
    assert fetched.json() == payload


def test_create_library_rejects_duplicate_in_same_organization(client: TestClient) -> None:
    payload = {
        "organization": "platform-mesh",
        "name": "portal-ui-lib",
        "kind": "ui",
        "repository_url": "https://github.com/platform-mesh/portal-ui-lib",
    }

    response = client.post("/libraries", json=payload)

    assert response.status_code == 409
    assert response.json() == {"detail": "Library 'platform-mesh/portal-ui-lib' already exists"}


def test_create_library_allows_same_name_in_other_organization(client: TestClient) -> None:
    payload = {
        "organization": "platform-mesh",
        "name": "webcomponents",
        "kind": "ui",
        "repository_url": "https://github.com/platform-mesh/webcomponents",
    }

    response = client.post("/libraries", json=payload)

    assert response.status_code == 201


def test_create_library_validates_payload(client: TestClient) -> None:
    payload = {
        "organization": "unknown-org",
        "name": "Not A Slug",
        "kind": "ui",
        "repository_url": "not-a-url",
    }

    response = client.post("/libraries", json=payload)

    assert response.status_code == 422
    invalid_fields = {error["loc"][-1] for error in response.json()["detail"]}
    assert invalid_fields == {"organization", "name", "repository_url"}
