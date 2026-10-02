"""Тесты LLM-постобработки, не требующие реальной LLM.

Проверяются детерминированные части: сборка стенограммы с метками, разбор
JSON-ответа, подстановка имён в ``display_name``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from audio_transcriber.config.settings import AppConfig
from audio_transcriber.domain.models import Speaker, TranscriptEntry
from audio_transcriber.llm import postprocess as postprocess_module
from audio_transcriber.llm.glossary import Glossary
from audio_transcriber.llm.postprocess import (
    _ordered_correction_keys,
    _verify_terms_with_llm,
    apply_participant_names,
    build_transcript_for_llm,
    chunk_chars_for_context,
    extract_participants,
    filter_participants_by_transcript,
    parse_participants_json,
    run_llm_postprocess,
)


def _speakers() -> list[Speaker]:
    return [
        Speaker(id="SPEAKER_00", display_name="Спикер 1"),
        Speaker(id="SPEAKER_01", display_name="Спикер 2"),
    ]


def _entries() -> list[TranscriptEntry]:
    speakers = _speakers()
    return [
        TranscriptEntry(start=0.0, end=1.0, text="Привет, я Максим.", speaker=speakers[0]),
        TranscriptEntry(start=1.0, end=2.0, text="Здравствуйте, я Ольга.", speaker=speakers[1]),
        TranscriptEntry(start=2.0, end=3.0, text="Рад знакомству.", speaker=speakers[0]),
    ]


def test_build_transcript_uses_speaker_labels() -> None:
    transcript = build_transcript_for_llm(_entries(), speakers=_speakers())

    assert "Спикер 1: Привет, я Максим." in transcript
    assert "Спикер 2: Здравствуйте, я Ольга." in transcript


def test_parse_participants_json_plain() -> None:
    labels = {"SPEAKER_00": "Спикер 1", "SPEAKER_01": "Спикер 2"}
    raw = '{"participants": [{"speaker": "Спикер 1", "name": "Максим"}]}'

    assert parse_participants_json(raw, labels) == {"SPEAKER_00": "Максим"}


def test_parse_participants_json_with_markdown_fence_and_noise() -> None:
    labels = {"SPEAKER_00": "Спикер 1"}
    raw = 'Вот результат:\n```json\n{"participants": [{"speaker": "Спикер 1", "name": "Максим"}]}\n```\nНадеюсь, помог.'

    assert parse_participants_json(raw, labels) == {"SPEAKER_00": "Максим"}


def test_parse_participants_json_returns_empty_on_garbage() -> None:
    labels = {"SPEAKER_00": "Спикер 1"}

    assert parse_participants_json("не JSON вовсе", labels) == {}


class _DuplicateNameClient:
    def chat(self, messages: list[dict[str, str]]) -> str:
        return (
            '{"participants": ['
            '{"speaker": "Спикер 1", "name": "Максим"}, '
            '{"speaker": "Спикер 2", "name": "Максим"}]}'
        )

    def close(self) -> None:
        pass


def test_extract_participants_deduplicates_names() -> None:
    names = extract_participants(_entries(), llm=_DuplicateNameClient(), speakers=_speakers())

    # одно имя не может принадлежать двум говорящим
    assert list(names.values()).count("Максим") == 1
    assert len(names) == 1


class _PhantomNameClient:
    """LLM выдумывает имя, которого нет в тексте стенограммы."""

    def chat(self, messages: list[dict[str, str]]) -> str:
        return (
            '{"participants": ['
            '{"speaker": "Спикер 1", "name": "Никита"}, '
            '{"speaker": "Спикер 2", "name": "Ольга"}]}'
        )

    def close(self) -> None:
        pass


def test_extract_participants_drops_hallucinated_names() -> None:
    names = extract_participants(_entries(), llm=_PhantomNameClient(), speakers=_speakers())

    # «Никита» в тексте нет — не применяем; «Ольга» есть — оставляем
    assert names == {"SPEAKER_01": "Ольга"}


def test_filter_participants_by_transcript_keeps_present_names() -> None:
    names = {"SPEAKER_00": "Максим", "SPEAKER_01": "Ольга"}

    assert filter_participants_by_transcript(names, _entries()) == names


def test_filter_participants_by_transcript_drops_absents() -> None:
    names = {"SPEAKER_00": "Максим", "SPEAKER_01": "Никита"}

    assert filter_participants_by_transcript(names, _entries()) == {"SPEAKER_00": "Максим"}


def test_filter_participants_by_transcript_drops_placeholder() -> None:
    names = {"SPEAKER_00": "Максим", "SPEAKER_01": "?"}

    assert filter_participants_by_transcript(names, _entries()) == {"SPEAKER_00": "Максим"}


def test_filter_participants_multiword_needs_one_part() -> None:
    entries = [TranscriptEntry(start=0.0, end=1.0, text="Меня зовут Максим Петров.")]
    # в тексте есть только имя — этого достаточно
    assert filter_participants_by_transcript({"SPEAKER_00": "Максим Петров"}, entries) == {
        "SPEAKER_00": "Максим Петров"
    }
    # фамилии в тексте нет, но имя присутствует — тоже оставляем
    assert filter_participants_by_transcript({"SPEAKER_00": "Максим Смирнов"}, entries) == {
        "SPEAKER_00": "Максим Смирнов"
    }
    # в тексте нет ни одного слова из имени — отбрасываем
    assert filter_participants_by_transcript({"SPEAKER_00": "Иван Смирнов"}, entries) == {}


def test_filter_participants_matches_declension() -> None:
    entries = [TranscriptEntry(start=0.0, end=1.0, text="Приветствую, Максима!")]
    assert filter_participants_by_transcript({"SPEAKER_00": "Максим"}, entries) == {
        "SPEAKER_00": "Максим"
    }


def test_parse_participants_json_skips_placeholder_names() -> None:
    labels = {"SPEAKER_00": "Спикер 1", "SPEAKER_01": "Спикер 2"}
    raw = (
        '{"participants": ['
        '{"speaker": "Спикер 1", "name": "?"}, '
        '{"speaker": "Спикер 2", "name": "неизвестно"}]}'
    )

    assert parse_participants_json(raw, labels) == {}


def test_parse_participants_json_skips_unknown_speakers() -> None:
    labels = {"SPEAKER_00": "Спикер 1", "SPEAKER_01": "Спикер 2"}
    raw = (
        '{"participants": ['
        '{"speaker": "Спикер 1", "name": "Максим"}, '
        '{"speaker": "?", "name": "Никита"}, '
        '{"speaker": "Спикер 9", "name": "Боб"}, '
        '{"speaker": "Спикер 2", "name": "Евгений"}]}'
    )

    # неизвестные метки («?», «Спикер 9») не создают фантомных участников
    assert parse_participants_json(raw, labels) == {
        "SPEAKER_00": "Максим",
        "SPEAKER_01": "Евгений",
    }


def test_apply_participant_names_updates_display_names() -> None:
    entries, speakers = apply_participant_names(_entries(), _speakers(), {"SPEAKER_00": "Максим"})

    names = {speaker.id: speaker.display_name for speaker in speakers}
    assert names["SPEAKER_00"] == "Спикер 1 — Максим"
    assert names["SPEAKER_01"] == "Спикер 2"
    # Реплики ссылаются на обновлённого говорящего.
    assert entries[0].speaker is not None
    assert entries[0].speaker.display_name == "Спикер 1 — Максим"


def test_apply_participant_names_updates_extra_speakers() -> None:
    speakers = _speakers()
    entry = TranscriptEntry(
        start=0.0,
        end=1.0,
        text="хором",
        speaker=speakers[0],
        overlap=True,
        extra_speakers=[speakers[1]],
    )

    entries, _ = apply_participant_names(
        [entry], speakers, {"SPEAKER_00": "Максим", "SPEAKER_01": "Ольга"}
    )

    # Имена обновляются и у основного, и у дополнительного говорящего.
    assert entries[0].speaker_label == "Спикер 1 — Максим + Спикер 2 — Ольга"


def test_apply_participant_names_without_names_is_identity() -> None:
    entries, speakers = apply_participant_names(_entries(), _speakers(), {})

    assert entries == _entries()
    assert speakers == _speakers()


def test_apply_participant_names_skips_empty_and_placeholder() -> None:
    _, speakers = apply_participant_names(
        _entries(), _speakers(), {"SPEAKER_00": "  ", "SPEAKER_01": "неизвестно"}
    )

    assert [speaker.display_name for speaker in speakers] == ["Спикер 1", "Спикер 2"]


class _NamesAndTermsClient:
    """Клиент, который возвращает и имя, и правку термина."""

    def chat(self, messages: list[dict[str, str]]) -> str:
        return (
            '{"participants": [{"speaker": "Спикер 1", "name": "Максим"}], '
            '"corrections": [{"before": "АИБ", "after": "ОИБ"}]}'
        )

    def close(self) -> None:
        pass


def test_run_llm_postprocess_extracts_names_when_enabled(
    tmp_path: Path, audio_file: Path
) -> None:
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        llm_enabled=True,
        llm_extract_names=True,
    )
    entries = _entries()

    new_entries, speakers, participants, _summary = run_llm_postprocess(
        config, entries, _speakers(), client=_NamesAndTermsClient()
    )

    assert participants is not None
    assert speakers[0].display_name == "Спикер 1 — Максим"
    assert new_entries[0].speaker.display_name == "Спикер 1 — Максим"
    # В шапку попадает только переименованный говорящий, а не «Спикер 2».
    assert participants == ["Спикер 1 — Максим"]


def test_run_llm_postprocess_participants_only_renamed(tmp_path: Path, audio_file: Path) -> None:
    """Участники — только говорящие с именем; «Спикер N» без имени не попадает."""
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        llm_enabled=True,
        llm_extract_names=True,
    )

    _, speakers, participants, _summary = run_llm_postprocess(
        config, _entries(), _speakers(), client=_NamesAndTermsClient()
    )

    assert speakers[1].display_name == "Спикер 2"
    assert participants == ["Спикер 1 — Максим"]


def test_run_llm_postprocess_participants_include_enrolled_names(
    tmp_path: Path, audio_file: Path
) -> None:
    """Имена из enrollment/образцов голоса попадают в участников без LLM."""
    speakers = [
        Speaker(id="SPEAKER_00", display_name="Иван"),
        Speaker(id="SPEAKER_01", display_name="Мария"),
        Speaker(id="SPEAKER_02", display_name="Спикер 3"),
    ]
    entries = [
        TranscriptEntry(start=0.0, end=1.0, text="Привет.", speaker=speakers[0]),
        TranscriptEntry(start=1.0, end=2.0, text="Здравствуйте.", speaker=speakers[1]),
        TranscriptEntry(start=2.0, end=3.0, text="И вам.", speaker=speakers[2]),
    ]
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        llm_enabled=True,
        llm_extract_names=True,
    )

    _entries_out, _speakers_out, participants, _summary = run_llm_postprocess(
        config, entries, speakers, client=_FakeClient()
    )

    # LLM ничего не нашла, но enrollment-имена уже есть у говорящих.
    assert participants == ["Иван", "Мария"]


def test_run_llm_postprocess_skips_names_when_disabled(tmp_path: Path, audio_file: Path) -> None:
    glossary_path = tmp_path / "glossary.txt"
    glossary_path.write_text("ОИБ\n", encoding="utf-8")
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        llm_enabled=True,
        llm_extract_names=False,
        glossary_path=glossary_path,
    )
    entries = [
        TranscriptEntry(
            start=0.0,
            end=1.0,
            text="Меня зовут Максим. Обсуждали АИБ.",
            speaker=_speakers()[0],
        )
    ]

    new_entries, speakers, participants, _summary = run_llm_postprocess(
        config, entries, _speakers(), client=_NamesAndTermsClient()
    )

    # имена не извлекались — говорящий остаётся под меткой
    assert participants is None
    assert speakers[0].display_name == "Спикер 1"
    # правка терминов по глоссарию продолжает работать
    assert new_entries[0].text.endswith("ОИБ.")


class _FakeClient:
    """Заглушка LLM-клиента: возвращает пустые ответы без обращений к модели."""

    def chat(self, messages: list[dict[str, str]]) -> str:
        return '{"participants": [], "corrections": []}'

    def close(self) -> None:
        pass


def test_run_llm_postprocess_writes_suggested_terms(tmp_path: Path, audio_file: Path) -> None:
    glossary_path = tmp_path / "glossary.txt"
    glossary_path.write_text("ОИБ\n", encoding="utf-8")
    speakers = _speakers()
    entries = [
        TranscriptEntry(
            start=0.0,
            end=1.0,
            text="Обсуждали ХТТП и снова ХТТП. ОИБ уже в списке.",
            speaker=speakers[0],
        )
    ]
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        llm_enabled=True,
        llm_suggest_terms=True,
        glossary_path=glossary_path,
    )

    run_llm_postprocess(config, entries, speakers, client=_FakeClient())

    suggested = glossary_path.with_name("glossary.suggested.txt")
    assert suggested.is_file()
    content = suggested.read_text(encoding="utf-8")
    assert "ХТТП\t2" in content
    assert "ОИБ" not in content


def test_run_llm_postprocess_suggestions_disabled_by_default(
    tmp_path: Path, audio_file: Path
) -> None:
    glossary_path = tmp_path / "glossary.txt"
    glossary_path.write_text("ОИБ\n", encoding="utf-8")
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        llm_enabled=True,
        glossary_path=glossary_path,
    )

    run_llm_postprocess(config, _entries(), _speakers(), client=_FakeClient())

    assert not glossary_path.with_name("glossary.suggested.txt").exists()


# --- Задача 1: жизненный цикл LLM-клиента ---------------------------------


class _EmptyChatClient:
    """Клиент, который ничего не находит и считает вызовы ``close``."""

    def __init__(self) -> None:
        self.closed = 0
        self.calls = 0

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.calls += 1
        return '{"participants": [], "corrections": []}'

    def close(self) -> None:
        self.closed += 1


def test_run_llm_postprocess_closes_internally_created_client(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Если клиент создан внутри — он обязан быть закрыт (иначе утечка VRAM)."""
    created: list[_EmptyChatClient] = []

    def fake_create(*_args, **_kwargs):
        client = _EmptyChatClient()
        created.append(client)
        return client

    monkeypatch.setattr(postprocess_module, "create_llm_client", fake_create)
    config = AppConfig(input_file=audio_file, output_dir=tmp_path / "out", llm_enabled=True)

    run_llm_postprocess(config, _entries(), _speakers())

    assert len(created) == 1
    assert created[0].closed == 1


def test_run_llm_postprocess_does_not_close_external_client(
    tmp_path: Path, audio_file: Path
) -> None:
    """Клиент, переданный снаружи, принадлежит вызывающему — не закрываем."""
    client = _EmptyChatClient()
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        llm_enabled=True,
        llm_extract_names=True,
    )

    run_llm_postprocess(config, _entries(), _speakers(), client=client)

    assert client.calls > 0
    assert client.closed == 0


def test_run_llm_postprocess_closes_internal_client_even_on_error(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Даже при сбоях LLM созданный клиент закрывается (try/finally)."""
    client = _EmptyChatClient()

    def fake_chat(messages: list[dict[str, str]]) -> str:
        raise RuntimeError("модель недоступна")

    client.chat = fake_chat  # type: ignore[method-assign]
    monkeypatch.setattr(postprocess_module, "create_llm_client", lambda *_a, **_kw: client)
    config = AppConfig(input_file=audio_file, output_dir=tmp_path / "out", llm_enabled=True)

    run_llm_postprocess(config, _entries(), _speakers())

    assert client.closed == 1


# --- Задача 4: правка терминов -------------------------------------------


class _TermCorrectionClient:
    """LLM-заглушка, возвращающая заданные пары ``before/after``."""

    def __init__(self, corrections: list[dict[str, str]]) -> None:
        self._corrections = corrections
        self.prompts: list[str] = []

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.prompts.append(messages[-1]["content"])
        return json.dumps({"corrections": self._corrections})

    def close(self) -> None:
        pass


def test_verify_terms_replaces_lowercase_and_mixed_case() -> None:
    """Ключ в исходном регистре должен матчить «аиб»/«АиБ» в тексте."""
    glossary = Glossary(terms=["ОИБ"])
    llm = _TermCorrectionClient([{"before": "АИБ", "after": "ОИБ"}])
    entries = [TranscriptEntry(start=0.0, end=1.0, text="обсуждали аиб, затем АиБ и АИБ.")]

    result = _verify_terms_with_llm(entries, glossary, llm=llm)

    assert result[0].text == "обсуждали ОИБ, затем ОИБ и ОИБ."


def test_verify_terms_replaces_when_llm_returns_lowercase_key() -> None:
    """Обратный случай: LLM вернул ключ в нижнем регистре."""
    glossary = Glossary(terms=["ОИБ"])
    llm = _TermCorrectionClient([{"before": "аиб", "after": "ОИБ"}])
    entries = [TranscriptEntry(start=0.0, end=1.0, text="Обсуждали АИБ.")]

    result = _verify_terms_with_llm(entries, glossary, llm=llm)

    assert result[0].text == "Обсуждали ОИБ."


def test_ordered_correction_keys_longest_first_and_deterministic() -> None:
    first = {"аиб": "ОИБ", "аибс": "ОИБС", "мк": "МК"}
    second = {"мк": "МК", "аибс": "ОИБС", "аиб": "ОИБ"}

    assert _ordered_correction_keys(first) == _ordered_correction_keys(second)
    # Длинные ключи раньше: иначе короткий «аиб» перехватил бы «аибс».
    assert _ordered_correction_keys(first) == ["аибс", "аиб", "мк"]


def test_ordered_correction_keys_tie_break_is_lexicographic() -> None:
    # При равной длине порядок задаёт сам ключ, а не порядок ответа LLM.
    assert _ordered_correction_keys({"б": "x", "а": "y"}) == ["б", "а"]
    assert _ordered_correction_keys({"а": "y", "б": "x"}) == ["б", "а"]


def test_verify_terms_result_is_independent_of_llm_order() -> None:
    """Перестановка правок в ответе LLM не меняет итоговую стенограмму."""
    glossary = Glossary(terms=["ОИБ", "АИБС"])
    entries = [TranscriptEntry(start=0.0, end=1.0, text="Обсуждали АИБС.")]
    corrections_a = [
        {"before": "АИБ", "after": "ОИБ"},
        {"before": "АИБС", "after": "АИБС"},
    ]
    corrections_b = list(reversed(corrections_a))

    first = _verify_terms_with_llm(
        entries, glossary, llm=_TermCorrectionClient(corrections_a)
    )
    second = _verify_terms_with_llm(
        entries, glossary, llm=_TermCorrectionClient(corrections_b)
    )

    # «АИБС» не распадается на «ОИБ» + «С»: длинный ключ имеет приоритет.
    assert first[0].text == second[0].text == "Обсуждали АИБС."


def test_verify_terms_chunks_long_transcript() -> None:
    """Длинная стенограмма уходит в LLM несколькими фрагментами (без HTTP 400)."""
    glossary = Glossary(terms=["ОИБ"])
    llm = _TermCorrectionClient([{"before": "АИБ", "after": "ОИБ"}])
    entries = [
        TranscriptEntry(
            start=float(i),
            end=float(i) + 1,
            text="обсуждали аиб " + "слово " * 6,
        )
        for i in range(10)
    ]

    result = _verify_terms_with_llm(entries, glossary, llm=llm, max_chunk_chars=40)

    assert len(llm.prompts) > 1
    assert all("ОИБ" in entry.text for entry in result)


def test_chunk_chars_for_context_scales_with_context() -> None:
    assert chunk_chars_for_context(4096) == 6000
    assert chunk_chars_for_context(8192) == 12000
    assert chunk_chars_for_context(2048) == 3000
    # Нижняя граница: слишком маленький контекст не даёт неработоспособный чанк.
    assert chunk_chars_for_context(128) == 1000


class _LengthCountingClient:
    """Считает суммарную длину сообщений каждого запроса (эмуляция контекста)."""

    def __init__(self) -> None:
        self.calls = 0
        self.max_request_chars = 0

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.calls += 1
        total = sum(len(message.get("content", "")) for message in messages)
        self.max_request_chars = max(self.max_request_chars, total)
        return '{"corrections": []}'

    def close(self) -> None:
        pass


def _long_entries(count: int = 60) -> list[TranscriptEntry]:
    return [
        TranscriptEntry(
            start=float(i),
            end=float(i) + 1,
            text="обсуждали системы и регламенты " * 3,
        )
        for i in range(count)
    ]


def test_verify_terms_fits_budget_with_large_glossary() -> None:
    """Большой глоссарий + длинный текст → каждый запрос влезает в бюджет."""
    terms = [f"Термин{i:03d}" for i in range(480)]
    glossary = Glossary(terms=terms)
    llm = _LengthCountingClient()

    _verify_terms_with_llm(_long_entries(), glossary, llm=llm, max_chunk_chars=6000)

    assert llm.calls > 1
    # Весь запрос (system + user + резерв ответа) укладывается в бюджет.
    assert llm.max_request_chars <= 6000


def test_verify_terms_limits_glossary_in_prompt() -> None:
    """Глоссарий не влезает целиком — в промпт попадает только его часть."""
    terms = [f"код{i:03d}" for i in range(480)]
    glossary = Glossary(terms=terms)
    llm = _TermCorrectionClient([])

    _verify_terms_with_llm(_long_entries(), glossary, llm=llm, max_chunk_chars=6000)

    assert llm.prompts
    assert all(prompt.count("код") < len(terms) for prompt in llm.prompts)


def test_verify_terms_rotates_glossary_across_chunks() -> None:
    """Разные фрагменты проверяют разные термины — суммарное покрытие растёт."""
    terms = [f"код{i:03d}" for i in range(480)]
    glossary = Glossary(terms=terms)
    llm = _TermCorrectionClient([])

    _verify_terms_with_llm(_long_entries(120), glossary, llm=llm, max_chunk_chars=6000)

    assert llm.prompts
    first_prompt_terms = sum(term in llm.prompts[0] for term in terms)
    union: set[str] = set()
    for prompt in llm.prompts:
        union.update(term for term in terms if term in prompt)
    assert len(union) > first_prompt_terms


def test_extract_participants_respects_chunk_limit() -> None:
    class _CountingClient:
        def __init__(self) -> None:
            self.calls = 0

        def chat(self, messages: list[dict[str, str]]) -> str:
            self.calls += 1
            return '{"participants": []}'

        def close(self) -> None:
            pass

    entries = [
        TranscriptEntry(start=float(i), end=float(i) + 1, text="Привет " * 10) for i in range(6)
    ]
    llm = _CountingClient()

    extract_participants(entries, llm=llm, speakers=_speakers(), max_chunk_chars=30)

    assert llm.calls > 1


def test_run_llm_postprocess_passes_context_chunk_limit(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Актуальный лимит чанка считается из ``llm_context_size`` и передаётся дальше."""
    recorded: dict[str, int | None] = {}

    def fake_extract(entries, *, llm, speakers=None, max_chunk_chars=None):
        recorded["extract"] = max_chunk_chars
        return {}

    def fake_correct(entries, glossary, *, llm=None, max_chunk_chars=None):
        recorded["correct"] = max_chunk_chars
        return entries

    monkeypatch.setattr(postprocess_module, "extract_participants", fake_extract)
    monkeypatch.setattr(postprocess_module, "correct_terms", fake_correct)
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        llm_enabled=True,
        llm_context_size=8192,
        llm_extract_names=True,
    )

    run_llm_postprocess(config, _entries(), _speakers(), client=_FakeClient())

    expected = chunk_chars_for_context(8192)
    assert recorded["extract"] == expected
    assert recorded["correct"] == expected


# --- Задача 1: резюме встречи и прозрачность промптов ---------------------


class _CapturingClient:
    """Клиент, возвращающий резюме-текст и запоминающий отправленные промпты."""

    def __init__(self, reply: str = "Тема: тест") -> None:
        self.reply = reply
        self.messages: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.messages.append(messages)
        return self.reply

    def close(self) -> None:
        pass


def test_run_llm_postprocess_returns_summary(tmp_path: Path, audio_file: Path) -> None:
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        llm_enabled=True,
    )
    config.ensure_output_dir()

    entries, _speakers_out, participants, summary = run_llm_postprocess(
        config, _entries(), _speakers(), client=_CapturingClient("Тема: релиз")
    )

    assert summary is not None
    assert "Тема: релиз" in summary
    assert participants is None
    assert entries


def test_run_llm_postprocess_summary_disabled_makes_no_calls(
    tmp_path: Path, audio_file: Path
) -> None:
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        llm_enabled=True,
        llm_summary=False,
    )
    client = _EmptyChatClient()

    _entries_out, _speakers_out, _participants, summary = run_llm_postprocess(
        config, _entries(), _speakers(), client=client
    )

    assert summary is None
    assert client.calls == 0


def test_run_llm_postprocess_summary_failure_is_soft(
    tmp_path: Path, audio_file: Path
) -> None:
    class _BrokenClient:
        def chat(self, messages: list[dict[str, str]]) -> str:
            raise RuntimeError("модель недоступна")

        def close(self) -> None:
            pass

    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        llm_enabled=True,
    )
    config.ensure_output_dir()

    entries, _speakers_out, _participants, summary = run_llm_postprocess(
        config, _entries(), _speakers(), client=_BrokenClient()
    )

    # Сбой LLM не роняет конвейер — просто нет резюме.
    assert summary is None
    assert entries


def test_run_llm_postprocess_saves_prompt_file(tmp_path: Path, audio_file: Path) -> None:
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        llm_enabled=True,
    )
    config.ensure_output_dir()

    run_llm_postprocess(config, _entries(), _speakers(), client=_CapturingClient())

    prompt_file = config.output_dir / f"{audio_file.stem}.llm_prompt.txt"
    assert prompt_file.is_file()
    content = prompt_file.read_text(encoding="utf-8")
    assert "=== резюме ===" in content
    assert "--- system ---" in content
    assert "--- user ---" in content


def test_run_llm_postprocess_applies_extra_instructions(
    tmp_path: Path, audio_file: Path
) -> None:
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        llm_enabled=True,
        llm_prompt_extra="ПИШИ МАКСИМАЛЬНО КРАТКО",
    )
    config.ensure_output_dir()
    client = _CapturingClient()

    run_llm_postprocess(config, _entries(), _speakers(), client=client)

    system_contents = [
        message["content"]
        for messages in client.messages
        for message in messages
        if message["role"] == "system"
    ]
    assert any("ПИШИ МАКСИМАЛЬНО КРАТКО" in content for content in system_contents)


def test_run_llm_postprocess_without_prompts_writes_no_file(
    tmp_path: Path, audio_file: Path
) -> None:
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        llm_enabled=True,
        llm_summary=False,
        llm_extract_names=False,
    )
    config.ensure_output_dir()

    run_llm_postprocess(config, _entries(), _speakers(), client=_EmptyChatClient())

    assert not (config.output_dir / f"{audio_file.stem}.llm_prompt.txt").exists()
