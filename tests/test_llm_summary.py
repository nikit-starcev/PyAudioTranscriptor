"""Тесты резюме встречи и прозрачности LLM-промптов (без реальной LLM).

Проверяются: map-reduce по фрагментам, мягкая деградация, подмешивание доп.
инструкций пользователя и запись фактических промптов.
"""

from __future__ import annotations

from pathlib import Path

from audio_transcriber.domain.models import Speaker, TranscriptEntry
from audio_transcriber.llm.chunking import human_name, named_speaker_labels
from audio_transcriber.llm.prompts import (
    PromptRecorder,
    PromptRecordingClient,
    append_extra_instructions,
    apply_extra_to_messages,
    load_extra_instructions,
    max_extra_chars_for_context,
)
from audio_transcriber.llm.summary import summarize_meeting


def _speakers() -> list[Speaker]:
    return [
        Speaker(id="SPEAKER_00", display_name="Спикер 1"),
        Speaker(id="SPEAKER_01", display_name="Спикер 2"),
    ]


def _short_entries() -> list[TranscriptEntry]:
    speakers = _speakers()
    return [
        TranscriptEntry(
            start=0.0,
            end=1.0,
            text="Обсудили сроки релиза. Решили выпустить в пятницу.",
            speaker=speakers[0],
        ),
        TranscriptEntry(start=1.0, end=2.0, text="Я подготовлю отчёт.", speaker=speakers[1]),
    ]


def _long_entries() -> list[TranscriptEntry]:
    speaker = _speakers()[0]
    return [
        TranscriptEntry(
            start=float(i), end=float(i) + 1, text="обсуждали задачу " + "слово " * 8, speaker=speaker
        )
        for i in range(8)
    ]


class _EchoClient:
    """LLM-заглушка: всегда возвращает заданный ответ и считает вызовы."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0
        self.messages: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.calls += 1
        self.messages.append(messages)
        return self.reply

    def close(self) -> None:
        pass


def test_summarize_meeting_short_transcript_single_call() -> None:
    llm = _EchoClient("Тема: релиз\nЧто сделать:\n- подготовить отчёт — Спикер 2")

    summary = summarize_meeting(_short_entries(), _speakers(), llm=llm)

    assert summary is not None
    assert summary.startswith("Тема: релиз")
    assert llm.calls == 1


def test_summarize_meeting_empty_marker_returns_none() -> None:
    llm = _EchoClient("нет данных")

    assert summarize_meeting(_short_entries(), _speakers(), llm=llm) is None


def test_human_name_extracts_real_names_and_skips_labels() -> None:
    assert human_name("Спикер 1 — Максим") == "Максим"
    assert human_name("Иван") == "Иван"
    assert human_name("Спикер 1") is None
    assert human_name("SPEAKER_00") is None
    assert human_name("?") is None
    assert human_name("   ") is None


def test_named_speaker_labels_uses_names_and_falls_back() -> None:
    speakers = [
        Speaker(id="SPEAKER_00", display_name="Спикер 1 — Максим"),
        Speaker(id="SPEAKER_01", display_name="Иван"),
        Speaker(id="SPEAKER_02", display_name="Спикер 3"),
    ]

    assert named_speaker_labels(speakers) == {
        "SPEAKER_00": "Максим",
        "SPEAKER_01": "Иван",
        "SPEAKER_02": "Спикер 3",
    }


def test_summarize_meeting_uses_speaker_display_names() -> None:
    """Имена говорящих (enrollment/переименование) попадают в промпт резюме."""
    speakers = [
        Speaker(id="SPEAKER_00", display_name="Спикер 1 — Максим"),
        Speaker(id="SPEAKER_01", display_name="Иван"),
    ]
    entries = [
        TranscriptEntry(start=0.0, end=1.0, text="Привет.", speaker=speakers[0]),
        TranscriptEntry(start=1.0, end=2.0, text="Здравствуйте.", speaker=speakers[1]),
    ]
    llm = _EchoClient("Участники: Максим, Иван")

    summary = summarize_meeting(entries, speakers, llm=llm)

    assert summary == "Участники: Максим, Иван"
    prompt = llm.messages[0][1]["content"]
    assert "Максим: Привет." in prompt
    assert "Иван: Здравствуйте." in prompt
    assert "Спикер 1:" not in prompt


def test_summarize_meeting_long_transcript_uses_map_reduce() -> None:
    llm = _EchoClient("Тема: тест")

    summary = summarize_meeting(_long_entries(), llm=llm, max_chunk_chars=60)

    # Несколько фрагментов (map) + один сводящий запрос (reduce).
    assert llm.calls > 2
    assert summary is not None


def test_summarize_meeting_uses_custom_system_prompt() -> None:
    """#97: пользовательский шаблон заменяет встроенный системный промпт."""
    llm = _EchoClient("Тема: X")

    summarize_meeting(
        _short_entries(), _speakers(), llm=llm, system_prompt="МОЙ СИСТЕМНЫЙ ПРОМПТ"
    )

    system_message = llm.messages[0][0]
    assert system_message["role"] == "system"
    assert system_message["content"] == "МОЙ СИСТЕМНЫЙ ПРОМПТ"


def test_summarize_meeting_custom_prompt_used_for_map_and_reduce() -> None:
    """Кастомный шаблон применяется и к фрагментам, и к сведению."""
    llm = _EchoClient("тезис")

    summarize_meeting(_long_entries(), llm=llm, max_chunk_chars=60, system_prompt="CUSTOM")

    assert llm.calls > 2
    for messages in llm.messages:
        assert messages[0]["content"] == "CUSTOM"


def test_summarize_meeting_returns_none_on_llm_failure() -> None:
    class _BrokenClient:
        def chat(self, messages: list[dict[str, str]]) -> str:
            raise RuntimeError("модель недоступна")

        def close(self) -> None:
            pass

    assert summarize_meeting(_short_entries(), _speakers(), llm=_BrokenClient()) is None


def test_load_extra_instructions_merges_text_and_file(tmp_path: Path) -> None:
    prompt_file = tmp_path / "extra.txt"
    prompt_file.write_text("Инструкция из файла", encoding="utf-8")

    assert load_extra_instructions("Текстовая инструкция", prompt_file) == (
        "Текстовая инструкция\nИнструкция из файла"
    )


def test_load_extra_instructions_empty_values_change_nothing(tmp_path: Path) -> None:
    assert load_extra_instructions(None, None) is None
    assert load_extra_instructions("   ", None) is None
    # Отсутствующий файл не роняет конвейер — только пустой результат.
    assert load_extra_instructions(None, tmp_path / "missing.txt") is None


def test_append_extra_instructions_truncates_to_limit() -> None:
    result = append_extra_instructions("SYSTEM", "а" * 50, max_chars=10)

    assert result.startswith("SYSTEM")
    assert "а" * 10 in result
    assert result.endswith("…")
    assert "а" * 11 not in result


def test_apply_extra_to_messages_empty_is_identity() -> None:
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]

    assert apply_extra_to_messages(messages, None, max_chars=100) is messages


def test_prompt_recording_client_applies_extra_and_records(tmp_path: Path) -> None:
    inner = _EchoClient("Тема: X")
    recorder = PromptRecorder(tmp_path / "prompt.txt")
    wrapped = PromptRecordingClient(
        inner,
        stage="резюме",
        extra_instructions="Отвечай максимально кратко",
        on_prompt=recorder.record,
    )

    summarize_meeting(_short_entries(), _speakers(), llm=wrapped)

    # Доп. инструкции попали в системный промпт фактического запроса.
    assert "Отвечай максимально кратко" in inner.messages[0][0]["content"]
    # Промпт этапа записан и сохранён в файл.
    assert recorder.entries[0][0] == "резюме"
    path = recorder.write()
    assert path is not None
    content = path.read_text(encoding="utf-8")
    assert "=== резюме ===" in content
    assert "Отвечай максимально кратко" in content


def test_prompt_recorder_without_prompts_writes_nothing(tmp_path: Path) -> None:
    recorder = PromptRecorder(tmp_path / "prompt.txt")

    assert recorder.write() is None
    assert not (tmp_path / "prompt.txt").exists()


def test_max_extra_chars_scales_and_is_bounded() -> None:
    assert max_extra_chars_for_context(4096) == 1024
    assert max_extra_chars_for_context(128) == 500
    assert max_extra_chars_for_context(10_000_000) == 4000
