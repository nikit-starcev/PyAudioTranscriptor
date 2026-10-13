"""Тесты SQLite-хранилища архива стенограмм (#99): метаданные и поиск."""

from __future__ import annotations

from pathlib import Path

import pytest

from audio_transcriber.storage.archive_db import ArchiveDB


def _entries() -> list[tuple[int, str | None, str]]:
    return [
        (0, "Иван", "Договорились до пятницы по поставке оборудования"),
        (1, "Мария", "Пятница выходной, перенесём встречу на понедельник"),
        (2, "Иван", "Хорошо, я подготовлю акт приёмки"),
    ]


def test_meta_crud_roundtrip(tmp_path: Path) -> None:
    with ArchiveDB(tmp_path / "archive.db") as db:
        record = db.ensure_record("job-1", default_date="2026-06-04")
        assert record.meeting_date == "2026-06-04"
        assert record.participants == ()
        assert record.title is None

        updated = db.upsert_record(
            "job-1",
            title="Планёрка",
            participants=["Иван", "Мария", "Иван"],
            notes="Обсудили сроки",
        )
        assert updated.title == "Планёрка"
        assert updated.participants == ("Иван", "Мария")  # дубли и пустые убраны
        assert updated.notes == "Обсудили сроки"

        # Частичное обновление не трогает остальные поля.
        cleared = db.upsert_record("job-1", notes="")
        assert cleared.notes is None
        assert cleared.title == "Планёрка"

        assert db.get_record("job-1") is not None
        assert [r.job_id for r in db.list_records()] == ["job-1"]
        assert db.delete_record("job-1") is True
        assert db.get_record("job-1") is None
        assert db.delete_record("job-1") is False


def test_meta_date_validation(tmp_path: Path) -> None:
    with ArchiveDB(tmp_path / "archive.db") as db:
        with pytest.raises(ValueError):
            db.upsert_record("job-1", meeting_date="04.06.2026")
        record = db.upsert_record("job-1", meeting_date="2026-06-04")
        assert record.meeting_date == "2026-06-04"


def test_meta_persists_between_opens(tmp_path: Path) -> None:
    path = tmp_path / "archive.db"
    with ArchiveDB(path) as db:
        db.ensure_record("job-1", default_date="2026-06-04")
        db.upsert_record("job-1", title="Сохранённая", notes="заметка")
    with ArchiveDB(path) as db:
        record = db.get_record("job-1")
        assert record is not None
        assert record.title == "Сохранённая"
        assert record.notes == "заметка"


def test_index_and_fts_search(tmp_path: Path) -> None:
    with ArchiveDB(tmp_path / "archive.db") as db:
        assert db.fts5 is True
        count = db.index_job("job-1", _entries(), result_mtime=111)
        assert count == 3
        assert db.is_current("job-1", 111)
        assert not db.is_current("job-1", 222)

        hits, total = db.search("пятниц")
        assert total == 2
        assert {(hit.job_id, hit.entry_index) for hit in hits} == {("job-1", 0), ("job-1", 1)}
        assert "пятниц" in hits[0].snippet.lower()

        # Поиск по префиксу и регистронезависимо (включая кириллицу).
        prefix_hits, _ = db.search("поставк")
        assert [hit.entry_index for hit in prefix_hits] == [0]
        case_hits, _ = db.search("ПОСТАВК")
        assert [hit.entry_index for hit in case_hits] == [0]

        # Несколько слов — все должны встретиться в одной реплике.
        multi, _ = db.search("акт приёмки")
        assert [hit.entry_index for hit in multi] == [2]
        empty, _ = db.search("нет-такого-слова")
        assert empty == []


def test_search_speaker_filter(tmp_path: Path) -> None:
    with ArchiveDB(tmp_path / "archive.db") as db:
        db.index_job("job-1", _entries(), result_mtime=1)
        hits, total = db.search("пятница", speaker="Мария")
        assert total == 1
        assert hits[0].entry_index == 1


def test_search_job_ids_filter(tmp_path: Path) -> None:
    with ArchiveDB(tmp_path / "archive.db") as db:
        db.index_job("job-1", _entries(), result_mtime=1)
        db.index_job("job-2", [(0, "Пётр", "Пятница и отпуск")], result_mtime=1)
        hits, total = db.search("пятница", job_ids=["job-2"])
        assert total == 1
        assert hits[0].job_id == "job-2"
        none_hits, none_total = db.search("пятница", job_ids=[])
        assert (none_hits, none_total) == ([], 0)


def test_index_replaces_previous_rows(tmp_path: Path) -> None:
    with ArchiveDB(tmp_path / "archive.db") as db:
        db.index_job("job-1", _entries(), result_mtime=1)
        db.index_job("job-1", [(0, "Иван", "Совсем другой текст")], result_mtime=2)
        assert db.index_state("job-1") == (True, 2, 1)
        _, total = db.search("пятниц")
        assert total == 0
        _, total = db.search("другой")
        assert total == 1


def test_speakers_and_remove_job(tmp_path: Path) -> None:
    with ArchiveDB(tmp_path / "archive.db") as db:
        db.index_job("job-1", _entries(), result_mtime=1)
        db.ensure_record("job-1", default_date="2026-06-04")
        assert db.speakers() == ["Иван", "Мария"]
        db.remove_job("job-1")
        assert db.speakers() == []
        assert db.get_record("job-1") is None
        assert db.indexed_job_ids() == set()


def test_like_fallback_search(tmp_path: Path) -> None:
    with ArchiveDB(tmp_path / "archive.db") as db:
        # Принудительно включаем фолбэк: индекс идёт в обычную таблицу.
        db.fts5 = False
        db.index_job("job-1", _entries(), result_mtime=5)
        hits, total = db.search("пятниц")
        assert total == 2
        assert {hit.entry_index for hit in hits} == {0, 1}
        hits, total = db.search("акт приёмки")
        assert [hit.entry_index for hit in hits] == [2]
        hits, total = db.search("пятница", speaker="Иван")
        assert total == 0
        assert db.speakers() == ["Иван", "Мария"]
