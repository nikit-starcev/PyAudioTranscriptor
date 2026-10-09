"""Тесты опционального LLM-прохода семантической правки (#75, suggest-only).

Проверяется детерминированная часть (мок LLM): разбор ответа, локализация
``before``, отброс невалидных правок, фильтр по уверенности и лимиты, а также
что при включённом флаге предложения только сохраняются — ничего не
применяется к стенограмме автоматически.
"""

from __future__ import annotations

import json
from pathlib import Path

from audio_transcriber.config.settings import AppConfig
from audio_transcriber.domain.models import Speaker, TranscriptEntry
from audio_transcriber.llm.postprocess import run_llm_postprocess
from audio_transcriber.llm.semantic import (
    SemanticEdit,
    collect_semantic_edits,
    load_stored_edits,
    locate_before,
    parse_semantic_edits,
    semantic_suggestions_path,
    suggestions_from_stored_edits,
    write_semantic_edits,
)


def _speaker() -> Speaker:
    return Speaker(id="SPEAKER_00", display_name="Спикер 1")


class _EditClient:
    """LLM-заглушка, возвращающая заданные правки на каждый запрос."""

    def __init__(self, edits: list[dict[str, object]]) -> None:
        self._edits = edits
        self.calls = 0
        self.prompts: list[str] = []

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.calls += 1
        self.prompts.append(messages[-1]["content"])
        return json.dumps({"edits": self._edits})

    def close(self) -> None:
        pass


# --- Разбор ответа LLM -------------------------------------------------------


def test_parse_semantic_edits_keeps_valid_and_drops_invalid() -> None:
    raw = json.dumps(
        {
            "edits": [
                {"before": "кароче", "after": "короче", "reason": "ASR", "confidence": 0.9},
                {"before": "x", "after": "x", "confidence": 0.9},  # after == before
                {"before": "", "after": "y", "confidence": 0.9},  # пустой before
                {"before": "a", "after": "b", "confidence": "0.5"},  # строка
                "garbage",
            ]
        }
    )

    edits = parse_semantic_edits(raw)

    assert [(edit.before, edit.after) for edit in edits] == [("кароче", "короче"), ("a", "b")]
    assert edits[0].confidence == 0.9
    assert edits[1].confidence == 0.5


def test_parse_semantic_edits_garbage_returns_empty() -> None:
    assert parse_semantic_edits("совсем не JSON") == []
    assert parse_semantic_edits('{"edits": "нет"}') == []
    assert parse_semantic_edits('{"edits": []}') == []


def test_parse_semantic_edits_missing_confidence_gets_zero() -> None:
    edits = parse_semantic_edits('{"edits": [{"before": "a", "after": "b"}]}')

    assert len(edits) == 1
    assert edits[0].confidence == 0.0


# --- Детерминированная локализация ------------------------------------------


def test_locate_before_first_occurrence() -> None:
    assert locate_before("abcabc", "bc") == (1, 3)
    assert locate_before("abc", "zz") is None
    assert locate_before("abc", "") is None


def test_collect_applies_confidence_and_location() -> None:
    entries = [TranscriptEntry(0.0, 1.0, "это кароче полный бред", speaker=_speaker())]
    client = _EditClient(
        [
            {"before": "кароче", "after": "короче", "confidence": 0.9},
            {"before": "полный бред", "after": "понятный текст", "confidence": 0.5},
            {"before": "нет такого", "after": "что-то", "confidence": 0.99},
        ]
    )

    edits = collect_semantic_edits(entries, llm=client, max_chunk_chars=1000)

    # 0.5 < порога 0.8, «нет такого» не найден — остаётся одна правка.
    assert [(e.index, e.start, e.end, e.before, e.after) for e in edits] == [
        (0, 4, 10, "кароче", "короче")
    ]


def test_collect_drops_over_long_edits() -> None:
    entries = [TranscriptEntry(0.0, 1.0, "ааа ббб", speaker=_speaker())]
    client = _EditClient(
        [{"before": "ааа", "after": "б" * 200, "confidence": 0.9}]
    )

    assert collect_semantic_edits(entries, llm=client, max_chunk_chars=1000) == []


def test_collect_chunks_long_transcript() -> None:
    entries = [
        TranscriptEntry(float(i), float(i) + 1, f"фраза {i} кароче далее", speaker=_speaker())
        for i in range(8)
    ]
    client = _EditClient([{"before": "кароче", "after": "короче", "confidence": 0.9}])

    edits = collect_semantic_edits(entries, llm=client, max_chunk_chars=40)

    assert client.calls > 1
    assert len(edits) == len(entries)
    assert all(edit.before == "кароче" for edit in edits)


# --- Хранение и пересборка ---------------------------------------------------


def test_write_and_load_roundtrip(tmp_path: Path) -> None:
    edits = [
        SemanticEdit(
            before="кароче", after="короче", reason="ASR", confidence=0.9, index=0, start=1, end=7
        )
    ]
    path = semantic_suggestions_path(tmp_path, "stem")

    assert write_semantic_edits(path, edits) == path
    stored = load_stored_edits(path)
    assert stored[0]["before"] == "кароче"
    assert stored[0]["confidence"] == 0.9
    # Пустой список ничего не пишет, отсутствие файла — пустой результат.
    assert write_semantic_edits(path, []) is None
    assert load_stored_edits(tmp_path / "nope.json") == []


def test_suggestions_from_stored_relocates_against_current_text() -> None:
    texts = ["это кароче полный бред"]
    stored = [
        {"index": 0, "before": "кароче", "after": "короче", "confidence": 0.9, "reason": "r"}
    ]

    suggestions = suggestions_from_stored_edits(texts, stored, min_confidence=0.0)

    assert len(suggestions) == 1
    assert suggestions[0].kind == "semantic"
    assert (suggestions[0].start, suggestions[0].end) == (4, 10)


def test_suggestions_from_stored_drops_not_found() -> None:
    stored = [{"index": 0, "before": "zzz", "after": "yyy", "confidence": 0.9}]

    assert suggestions_from_stored_edits(["abc"], stored, min_confidence=0.0) == []


def test_suggestions_from_stored_ignores_out_of_range_index() -> None:
    stored = [{"index": 9, "before": "a", "after": "b", "confidence": 0.9}]

    assert suggestions_from_stored_edits(["abc"], stored, min_confidence=0.0) == []


# --- Интеграция с run_llm_postprocess ---------------------------------------


def test_run_llm_postprocess_flag_off_makes_no_semantic_call(
    tmp_path: Path, audio_file: Path
) -> None:
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        llm_enabled=True,
        llm_model=Path("llm.gguf"),
        llm_summary=False,
    )
    client = _EditClient([{"before": "бред", "after": "смысл", "confidence": 0.9}])
    entries = [TranscriptEntry(0.0, 1.0, "полный бред", speaker=_speaker())]

    run_llm_postprocess(config, entries, [_speaker()], client=client)

    assert client.calls == 0
    assert not semantic_suggestions_path(config.output_dir, audio_file.stem).exists()


def test_run_llm_postprocess_flag_on_saves_but_does_not_apply(
    tmp_path: Path, audio_file: Path
) -> None:
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        llm_enabled=True,
        llm_model=Path("llm.gguf"),
        llm_summary=False,
        llm_correct_semantic=True,
    )
    client = _EditClient([{"before": "бред", "after": "смысл", "confidence": 0.9}])
    entries = [TranscriptEntry(0.0, 1.0, "полный бред", speaker=_speaker())]

    out_entries, _speakers, _participants, _summary = run_llm_postprocess(
        config, entries, [_speaker()], client=client
    )

    assert client.calls == 1
    # Ничего не применено автоматически.
    assert out_entries[0].text == "полный бред"
    stored = load_stored_edits(
        semantic_suggestions_path(config.output_dir, audio_file.stem)
    )
    assert len(stored) == 1
    assert stored[0]["before"] == "бред"
    assert stored[0]["after"] == "смысл"
