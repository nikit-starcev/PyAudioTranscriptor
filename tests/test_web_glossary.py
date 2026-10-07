"""Тесты REST API глоссария веб-интерфейса (``/api/glossary``).

БД глоссария изолируется в ``tmp_path`` через настройки; конвейер и модели не
задействованы.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from audio_transcriber.web.app import create_app
from audio_transcriber.web.paths import WebPaths


@pytest.fixture
def web_paths(tmp_path: Path) -> WebPaths:
    return WebPaths(tmp_path / "web-data")


@pytest.fixture
def settings_payload(tmp_path: Path) -> dict[str, object]:
    return {"glossary_db": str(tmp_path / "custom-glossary.db")}


@pytest.fixture
def client(
    web_paths: WebPaths, settings_payload: dict[str, object]
) -> Iterator[TestClient]:
    web_paths.ensure()
    web_paths.settings_json.write_text(
        json.dumps(settings_payload, ensure_ascii=False), encoding="utf-8"
    )
    app = create_app(paths=web_paths, heartbeat=0.05)
    with TestClient(app) as test_client:
        yield test_client


def test_glossary_stats_start_empty(client: TestClient) -> None:
    assert client.get("/api/glossary/stats").json() == {"sources": 0, "entries": 0, "enabled": 0}


def test_create_list_patch_delete_entry(client: TestClient) -> None:
    created = client.post(
        "/api/glossary/entries",
        json={"canonical": "КИСУСС", "variant": "кисус", "note": "система"},
    )
    assert created.status_code == 201
    entry = created.json()
    assert entry["canonical"] == "КИСУСС"
    assert entry["variant"] == "кисус"
    assert entry["source"] == "manual"
    assert entry["enabled"] is True
    entry_id = entry["id"]

    listing = client.get("/api/glossary/entries").json()
    assert listing["total"] == 1
    assert [item["id"] for item in listing["entries"]] == [entry_id]

    patched = client.patch(
        f"/api/glossary/entries/{entry_id}",
        json={"canonical": "КИСУСС v2", "note": "новое", "enabled": False},
    )
    assert patched.status_code == 200
    body = patched.json()
    assert body["canonical"] == "КИСУСС v2"
    assert body["note"] == "новое"
    assert body["enabled"] is False

    assert client.get("/api/glossary/entries", params={"enabled_only": True}).json()["total"] == 0

    deleted = client.delete(f"/api/glossary/entries/{entry_id}")
    assert deleted.status_code == 200
    assert client.delete(f"/api/glossary/entries/{entry_id}").status_code == 404


def test_create_entry_rejects_empty_canonical(client: TestClient) -> None:
    response = client.post("/api/glossary/entries", json={"canonical": "   "})

    assert response.status_code == 400
    assert "пустым" in response.json()["detail"]


def test_patch_entry_requires_field(client: TestClient) -> None:
    entry = client.post("/api/glossary/entries", json={"canonical": "Термин"}).json()

    assert client.patch(f"/api/glossary/entries/{entry['id']}", json={}).status_code == 400


def test_search_and_pagination(client: TestClient) -> None:
    for name in ("Альфа", "Бета", "Гамма"):
        assert client.post("/api/glossary/entries", json={"canonical": name}).status_code == 201

    page = client.get("/api/glossary/entries", params={"limit": 2, "offset": 1}).json()

    assert page["total"] == 3
    assert len(page["entries"]) == 2
    found = client.get("/api/glossary/entries", params={"search": "бета"}).json()
    assert found["total"] == 1


def test_pagination_without_search_reports_total_and_page(client: TestClient) -> None:
    """Без поиска пагинация идёт в SQL, но total и срез остаются верными (#89)."""
    names = [f"Термин {index:02d}" for index in range(5)]
    for name in names:
        assert client.post("/api/glossary/entries", json={"canonical": name}).status_code == 201

    page = client.get("/api/glossary/entries", params={"limit": 2, "offset": 1}).json()

    assert page["total"] == 5
    assert [entry["canonical"] for entry in page["entries"]] == sorted(names)[1:3]


def test_import_txt_and_source_management(client: TestClient) -> None:
    content = "кисус = КИСУСС\n# комментарий\nОтдельный термин\n".encode()
    response = client.post(
        "/api/glossary/import",
        files={"file": ("terms.txt", content, "text/plain")},
        data={"source": "ТЗ"},
    )

    assert response.status_code == 200
    report = response.json()
    assert report["source"] == "ТЗ"
    assert report["kind"] == "txt"
    assert report["added"] == 2

    sources = client.get("/api/glossary/sources").json()
    assert len(sources) == 1
    assert sources[0]["name"] == "ТЗ"
    assert sources[0]["count"] == 2
    assert sources[0]["enabled"] is True

    disabled = client.patch("/api/glossary/sources/ТЗ", json={"enabled": False})
    assert disabled.status_code == 200
    assert disabled.json()["enabled"] is False

    assert client.delete("/api/glossary/sources/ТЗ").status_code == 200
    assert client.get("/api/glossary/sources").json() == []
    assert client.get("/api/glossary/entries").json()["total"] == 0


def test_import_csv_with_kind(client: TestClient) -> None:
    content = "canonical,wrong,note\nКИСУСС,кисус,система\n".encode()
    response = client.post(
        "/api/glossary/import",
        files={"file": ("terms.csv", content, "text/csv")},
        data={"kind": "csv"},
    )

    assert response.status_code == 200
    assert response.json()["kind"] == "csv"
    entries = client.get("/api/glossary/entries").json()["entries"]
    assert entries[0]["canonical"] == "КИСУСС"
    assert entries[0]["variant"] == "кисус"


def test_import_rejects_bad_kind_and_empty_file(client: TestClient) -> None:
    bad = client.post(
        "/api/glossary/import",
        files={"file": ("x.txt", b"a=b", "text/plain")},
        data={"kind": "xml"},
    )
    assert bad.status_code == 400

    empty = client.post(
        "/api/glossary/import",
        files={"file": ("x.txt", b"", "text/plain")},
    )
    assert empty.status_code == 400


def test_missing_source_returns_404(client: TestClient) -> None:
    assert client.patch("/api/glossary/sources/none", json={"enabled": False}).status_code == 404
    assert client.delete("/api/glossary/sources/none").status_code == 404


def test_quick_add_selection_becomes_variant(client: TestClient) -> None:
    response = client.post(
        "/api/glossary/quick",
        json={
            "term": "кисус",
            "canonical": "КИСУСС",
            "note": "система",
            "source": "тест_артефакты.mp4",
        },
    )

    assert response.status_code == 201
    entry = response.json()
    assert entry["canonical"] == "КИСУСС"
    assert entry["variant"] == "кисус"
    assert entry["note"] == "система"
    assert entry["source"] == "тест_артефакты.mp4"


def test_quick_add_empty_canonical_returns_400(client: TestClient) -> None:
    response = client.post(
        "/api/glossary/quick",
        json={"term": "кисус", "canonical": "   ...  "},
    )

    assert response.status_code == 400
    assert "канон" in response.json()["detail"].casefold()
    assert client.get("/api/glossary/entries").json()["total"] == 0


def test_quick_add_missing_canonical_returns_400(client: TestClient) -> None:
    response = client.post("/api/glossary/quick", json={"term": "кисус"})

    assert response.status_code == 400
    assert "канон" in response.json()["detail"].casefold()


def test_quick_add_source_defaults_to_manual(client: TestClient) -> None:
    response = client.post(
        "/api/glossary/quick",
        json={"term": "кисус", "canonical": "КИСУСС"},
    )

    assert response.status_code == 201
    assert response.json()["source"] == "manual"


def test_quick_add_strips_whitespace_and_punctuation(client: TestClient) -> None:
    response = client.post(
        "/api/glossary/quick",
        json={"term": "«кисус»,", "canonical": "  КИСУСС  "},
    )

    assert response.status_code == 201
    entry = response.json()
    assert entry["canonical"] == "КИСУСС"
    assert entry["variant"] == "кисус"


def test_quick_add_explicit_variant(client: TestClient) -> None:
    response = client.post(
        "/api/glossary/quick",
        json={"term": "КИСУСС", "canonical": "КИСУСС", "variant": "кисусс"},
    )

    assert response.status_code == 201
    assert response.json()["variant"] == "кисусс"


def test_quick_add_empty_term_returns_400(client: TestClient) -> None:
    response = client.post(
        "/api/glossary/quick", json={"term": "   ...  ", "canonical": "КИСУСС"}
    )

    assert response.status_code == 400
    assert "пустым" in response.json()["detail"]


def test_quick_add_deduplicates_existing_entry(client: TestClient) -> None:
    body = {"term": "кисус", "canonical": "КИСУСС"}
    first = client.post("/api/glossary/quick", json=body)
    second = client.post("/api/glossary/quick", json=body)

    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["id"] == second.json()["id"]
    assert client.get("/api/glossary/entries").json()["total"] == 1
