"""Тесты enrollment-диаризации: сопоставление говорящих с именами по голосу.

Реальная модель эмбеддингов не загружается — вместо неё внедряется фейковый
движок, а декодирование аудио подменяется (``load_waveform``). Отдельно
проверяются чистые функции над синтетическими эмбеддингами: косинусное
сходство, усреднение нескольких образцов и жадный мэтчинг с порогом.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from audio_transcriber.diarization import enrollment
from audio_transcriber.diarization.enrollment import (
    assign_speaker_names,
    average_embeddings,
    cosine_similarities,
    enroll_speakers,
    l2_normalize,
    match_speakers,
)
from audio_transcriber.domain.models import SpeakerSegment

SAMPLE_RATE = 16000


def _unit(*values: float) -> np.ndarray:
    return l2_normalize(np.asarray(values, dtype=np.float32))


# --- чистые функции на синтетических эмбеддингах ---------------------------


def test_cosine_similarities_known_values() -> None:
    similarities = cosine_similarities(
        {"A": _unit(1, 0), "B": _unit(0, 1)},
        {"X": _unit(1, 0), "Y": _unit(1, 1)},
    )

    assert similarities["A"]["X"] == pytest.approx(1.0)
    assert similarities["B"]["X"] == pytest.approx(0.0)
    assert similarities["A"]["Y"] == pytest.approx(1 / np.sqrt(2), abs=1e-6)


def test_match_speakers_picks_confident_pairs() -> None:
    similarities = {"A": {"X": 0.9, "Y": 0.1}, "B": {"X": 0.2, "Y": 0.8}}

    assert match_speakers(similarities, 0.6) == {"A": "X", "B": "Y"}


def test_match_speakers_threshold_leaves_unmatched() -> None:
    similarities = {"A": {"X": 0.55, "Y": 0.4}}

    assert match_speakers(similarities, 0.6) == {}


def test_match_speakers_is_one_to_one() -> None:
    # Два говорящих ближе всего к одному имени — имя достаётся только лучшему.
    similarities = {"A": {"X": 0.95}, "B": {"X": 0.9}}

    assert match_speakers(similarities, 0.6) == {"A": "X"}


def test_match_speakers_allow_shared_names_is_many_to_one() -> None:
    # При allow_shared_names одно имя может достаться нескольким кластерам —
    # один реальный говорящий, раздробленный на фрагменты, называется целиком.
    similarities = {"A": {"X": 0.95, "Y": 0.1}, "B": {"X": 0.9, "Y": 0.2}}

    assert match_speakers(similarities, 0.6, allow_shared_names=True) == {
        "A": "X",
        "B": "X",
    }


def test_match_speakers_allow_shared_names_respects_threshold() -> None:
    similarities = {"A": {"X": 0.95}, "B": {"X": 0.5}}

    assert match_speakers(similarities, 0.6, allow_shared_names=True) == {"A": "X"}


def test_average_embeddings_multiple_samples() -> None:
    averaged = average_embeddings([_unit(1, 0), _unit(0, 1)])

    assert averaged is not None
    assert averaged == pytest.approx(np.asarray([1, 1], dtype=np.float32) / np.sqrt(2), abs=1e-6)


def test_average_embeddings_empty_is_none() -> None:
    assert average_embeddings([]) is None


def test_l2_normalize_zero_vector_is_safe() -> None:
    assert np.allclose(l2_normalize(np.zeros(3, dtype=np.float32)), 0.0)


# --- вспомогательные фикстуры для end-to-end тестов ------------------------


class _FakeEngine:
    """Фейковый движок: вектор выбирается по «речевому» значению окна.

    Сигнал выбирается как значение ненулевых сэмплов (медиана): подготовка
    образца (#29) добавляет по краям немного тишины и нормализует уровень, но
    не меняет, какой именно речевой блок попал в окно. Так тест по-прежнему
    видит выбор окна (по энергии), игнорируя паддинг-тишину.
    """

    def __init__(self, vectors: dict[int, list[float]]) -> None:
        self.window_seconds = 5.0
        self._vectors = {key: np.asarray(value, dtype=np.float32) for key, value in vectors.items()}

    def embed(self, waveform: np.ndarray) -> np.ndarray:
        voiced = np.asarray(waveform, dtype=np.float32).reshape(-1)
        voiced = voiced[np.abs(voiced) > 1e-6]
        key = round(float(np.median(voiced)) * 100) if voiced.size else 0
        return self._vectors[key]


class _RaisingEngine:
    """Движок, всегда падающий на инференсе (мягкая деградация)."""

    window_seconds = 5.0

    def embed(self, waveform: np.ndarray) -> np.ndarray:
        raise RuntimeError("инференс недоступен")


@pytest.fixture
def install_loader(monkeypatch: pytest.MonkeyPatch):
    """Подменяет ``load_waveform``: образцы — константы, аудио — блоки сигнала.

    Остальная часть аудио — тишина. Блоки задают «речь» нужной громкости в
    нужные интервалы, чтобы энергетический выбор окна был проверяемым.
    """

    def install(
        reference_tags: dict[str, int],
        *,
        audio_seconds: float = 20.0,
        audio_blocks: dict[tuple[float, float], float] | None = None,
    ) -> None:
        def loader(path: Path, *, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
            tag = reference_tags.get(path.name)
            if tag is not None:
                return np.full(sample_rate, float(tag), dtype=np.float32)
            samples = np.zeros(int(audio_seconds * sample_rate), dtype=np.float32)
            for (start, end), value in (audio_blocks or {}).items():
                samples[round(start * sample_rate) : round(end * sample_rate)] = value
            return samples

        monkeypatch.setattr(enrollment, "load_waveform", loader)

    return install


def _segment(path: Path) -> Path:
    path.write_bytes(b"")
    return path


def test_assign_speaker_names_matches_by_voice(
    tmp_path: Path, install_loader
) -> None:
    install_loader(
        {"ivan.wav": 0.11, "maria.wav": 0.22},
        audio_blocks={(0.0, 3.0): 0.5, (10.0, 13.0): 0.7},
    )
    ivan = _segment(tmp_path / "ivan.wav")
    maria = _segment(tmp_path / "maria.wav")
    audio = tmp_path / "call.wav"
    audio.write_bytes(b"")
    engine = _FakeEngine(
        {50: [1, 0, 0], 70: [0, 1, 0], 11: [0.99, 0.01, 0], 22: [0.01, 0.99, 0]}
    )
    segments = [
        SpeakerSegment(start=0.0, end=3.0, speaker_id="SPEAKER_00"),
        SpeakerSegment(start=10.0, end=13.0, speaker_id="SPEAKER_01"),
    ]

    mapping = assign_speaker_names(
        speaker_segments=segments,
        references={"Иван": (ivan,), "Мария": (maria,)},
        audio_path=audio,
        min_similarity=0.6,
        engine=engine,
    )

    assert mapping == {"SPEAKER_00": "Иван", "SPEAKER_01": "Мария"}


def test_assign_speaker_names_shares_name_between_fragments(
    tmp_path: Path, install_loader
) -> None:
    """Один голос, раздробленный на два кластера, называется целиком.

    При мэтчинге «один-к-одному» имя досталось бы лишь одному фрагменту, а
    крупнейший кластер того же человека остался бы «Спикер N». По умолчанию
    имя могут разделить несколько кластеров (many-to-one) — это и поднимает
    охват имён без ложных срабатываний (порог всё равно соблюдается).
    """
    install_loader(
        {"ivan.wav": 0.11},
        audio_blocks={(0.0, 3.0): 0.5, (10.0, 13.0): 0.7},
    )
    ivan = _segment(tmp_path / "ivan.wav")
    audio = tmp_path / "call.wav"
    audio.write_bytes(b"")
    engine = _FakeEngine(
        {
            11: [0.99, 0.01, 0],
            50: [0.98, 0.02, 0],
            70: [0.96, 0.04, 0],
        }
    )
    segments = [
        SpeakerSegment(start=0.0, end=3.0, speaker_id="SPEAKER_00"),
        SpeakerSegment(start=10.0, end=13.0, speaker_id="SPEAKER_01"),
    ]

    mapping = assign_speaker_names(
        speaker_segments=segments,
        references={"Иван": (ivan,)},
        audio_path=audio,
        min_similarity=0.6,
        engine=engine,
    )

    assert mapping == {"SPEAKER_00": "Иван", "SPEAKER_01": "Иван"}


def test_assign_speaker_names_averages_multiple_samples_per_name(
    tmp_path: Path, install_loader
) -> None:
    install_loader({"a.wav": 0.11, "b.wav": 0.12}, audio_blocks={(0.0, 3.0): 0.5})
    first = _segment(tmp_path / "a.wav")
    second = _segment(tmp_path / "b.wav")
    audio = tmp_path / "call.wav"
    audio.write_bytes(b"")
    engine = _FakeEngine({50: [1, 0, 0], 11: [1, 0, 0], 12: [0.9, 0.1, 0]})

    mapping = assign_speaker_names(
        speaker_segments=[SpeakerSegment(start=0.0, end=3.0, speaker_id="SPEAKER_00")],
        references={"Иван": (first, second)},
        audio_path=audio,
        min_similarity=0.6,
        engine=engine,
    )

    assert mapping == {"SPEAKER_00": "Иван"}


def test_assign_speaker_names_respects_threshold(tmp_path: Path, install_loader) -> None:
    install_loader({"ivan.wav": 0.11}, audio_blocks={(0.0, 3.0): 0.5})
    ivan = _segment(tmp_path / "ivan.wav")
    audio = tmp_path / "call.wav"
    audio.write_bytes(b"")
    # косинус([1, 0], [0.5, sqrt(3)/2]) = 0.5 — ниже порога 0.6, выше 0.4
    engine = _FakeEngine({50: [1, 0], 11: [0.5, float(np.sqrt(0.75))]})
    kwargs = {
        "speaker_segments": [SpeakerSegment(start=0.0, end=3.0, speaker_id="SPEAKER_00")],
        "references": {"Иван": (ivan,)},
        "audio_path": audio,
        "engine": engine,
    }

    assert assign_speaker_names(min_similarity=0.6, **kwargs) == {}
    assert assign_speaker_names(min_similarity=0.4, **kwargs) == {"SPEAKER_00": "Иван"}


def test_assign_speaker_names_without_references_is_noop(tmp_path: Path) -> None:
    audio = tmp_path / "call.wav"
    audio.write_bytes(b"")

    mapping = assign_speaker_names(
        speaker_segments=[SpeakerSegment(start=0.0, end=3.0, speaker_id="SPEAKER_00")],
        references={},
        audio_path=audio,
        engine=_RaisingEngine(),
    )

    assert mapping == {}


def test_assign_speaker_names_without_speakers_is_noop(tmp_path: Path) -> None:
    audio = tmp_path / "call.wav"
    audio.write_bytes(b"")
    reference = _segment(tmp_path / "ivan.wav")

    mapping = assign_speaker_names(
        speaker_segments=[],
        references={"Иван": (reference,)},
        audio_path=audio,
        engine=_RaisingEngine(),
    )

    assert mapping == {}


def test_assign_speaker_names_degrades_when_model_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, install_loader
) -> None:
    install_loader({"ivan.wav": 0.11})
    reference = _segment(tmp_path / "ivan.wav")
    audio = tmp_path / "call.wav"
    audio.write_bytes(b"")

    def failing_factory(**kwargs: object) -> object:
        raise RuntimeError("локальная модель недоступна")

    monkeypatch.setattr(enrollment, "PyannoteEmbeddingEngine", failing_factory)

    mapping = assign_speaker_names(
        speaker_segments=[SpeakerSegment(start=0.0, end=3.0, speaker_id="SPEAKER_00")],
        references={"Иван": (reference,)},
        audio_path=audio,
    )

    assert mapping == {}


def test_assign_speaker_names_degrades_when_embedding_fails(
    tmp_path: Path, install_loader
) -> None:
    install_loader({"ivan.wav": 0.11}, audio_blocks={(0.0, 3.0): 0.5})
    reference = _segment(tmp_path / "ivan.wav")
    audio = tmp_path / "call.wav"
    audio.write_bytes(b"")

    mapping = assign_speaker_names(
        speaker_segments=[SpeakerSegment(start=0.0, end=3.0, speaker_id="SPEAKER_00")],
        references={"Иван": (reference,)},
        audio_path=audio,
        engine=_RaisingEngine(),
    )

    assert mapping == {}


def test_assign_speaker_names_ignores_broken_reference(
    tmp_path: Path, install_loader, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Битый образец пропускается, рабочий продолжает участвовать в матчинге."""
    install_loader({"good.wav": 0.11}, audio_blocks={(0.0, 3.0): 0.5})
    good = _segment(tmp_path / "good.wav")
    broken = _segment(tmp_path / "broken.wav")
    audio = tmp_path / "call.wav"
    audio.write_bytes(b"")
    real_loader = enrollment.load_waveform

    def loader(path: Path, *, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        if path.name == "broken.wav":
            raise RuntimeError("битый файл")
        return real_loader(path, sample_rate=sample_rate)

    monkeypatch.setattr(enrollment, "load_waveform", loader)
    engine = _FakeEngine({50: [1, 0, 0], 11: [0.99, 0.01, 0]})

    mapping = assign_speaker_names(
        speaker_segments=[SpeakerSegment(start=0.0, end=3.0, speaker_id="SPEAKER_00")],
        references={"Иван": (broken, good)},
        audio_path=audio,
        min_similarity=0.6,
        engine=engine,
    )

    assert mapping == {"SPEAKER_00": "Иван"}


# --- энергетический выбор окон и диагностика -------------------------------


def test_reference_uses_most_energetic_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Образец: 12 с тишины, речь только в конце (7–12 с).

    Старое поведение брало первые 5 секунд (тишину) — эмбеддинг не строился.
    """
    ref_wf = np.zeros(12 * SAMPLE_RATE, dtype=np.float32)
    ref_wf[7 * SAMPLE_RATE : 12 * SAMPLE_RATE] = 0.42
    audio_wf = np.zeros(20 * SAMPLE_RATE, dtype=np.float32)
    audio_wf[0 : 3 * SAMPLE_RATE] = 0.42
    reference = _segment(tmp_path / "ivan.wav")
    audio = tmp_path / "call.wav"
    audio.write_bytes(b"")

    def loader(path: Path, *, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        return ref_wf if path.name == "ivan.wav" else audio_wf

    monkeypatch.setattr(enrollment, "load_waveform", loader)
    engine = _FakeEngine({42: [1, 0, 0]})

    mapping = assign_speaker_names(
        speaker_segments=[SpeakerSegment(start=0.0, end=3.0, speaker_id="SPEAKER_00")],
        references={"Иван": (reference,)},
        audio_path=audio,
        min_similarity=0.6,
        engine=engine,
    )

    assert mapping == {"SPEAKER_00": "Иван"}


def test_speaker_uses_most_energetic_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Длинный сегмент 0–10 с, но речь только в конце (5–10 с).

    Старое поведение брало начало сегмента (тишину) — эмбеддинг не строился.
    """
    audio_wf = np.zeros(20 * SAMPLE_RATE, dtype=np.float32)
    audio_wf[5 * SAMPLE_RATE : 10 * SAMPLE_RATE] = 0.42
    ref_wf = np.full(SAMPLE_RATE, 0.42, dtype=np.float32)
    reference = _segment(tmp_path / "ivan.wav")
    audio = tmp_path / "call.wav"
    audio.write_bytes(b"")

    def loader(path: Path, *, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        return ref_wf if path.name == "ivan.wav" else audio_wf

    monkeypatch.setattr(enrollment, "load_waveform", loader)
    engine = _FakeEngine({42: [1, 0, 0]})

    mapping = assign_speaker_names(
        speaker_segments=[SpeakerSegment(start=0.0, end=10.0, speaker_id="SPEAKER_00")],
        references={"Иван": (reference,)},
        audio_path=audio,
        min_similarity=0.6,
        engine=engine,
    )

    assert mapping == {"SPEAKER_00": "Иван"}


def test_enroll_speakers_reports_best_candidate_when_below_threshold(
    tmp_path: Path, install_loader, caplog: pytest.LogCaptureFixture
) -> None:
    """При отсутствии совпадений в лог (INFO) идут матрица и лучший кандидат."""
    install_loader({"ivan.wav": 0.11}, audio_blocks={(0.0, 3.0): 0.5})
    reference = _segment(tmp_path / "ivan.wav")
    audio = tmp_path / "call.wav"
    audio.write_bytes(b"")
    # косинус = 0.5 — ниже порога 0.6.
    engine = _FakeEngine({50: [1, 0], 11: [0.5, float(np.sqrt(0.75))]})

    with caplog.at_level(logging.INFO, logger="audio_transcriber.diarization.enrollment"):
        outcome = enroll_speakers(
            speaker_segments=[SpeakerSegment(start=0.0, end=3.0, speaker_id="SPEAKER_00")],
            references={"Иван": (reference,)},
            audio_path=audio,
            min_similarity=0.6,
            engine=engine,
        )

    assert outcome.mapping == {}
    assert outcome.speaker_count == 1
    name, score = outcome.best_candidates["SPEAKER_00"]
    assert name == "Иван"
    assert score == pytest.approx(0.5, abs=1e-6)
    assert "SPEAKER_00 — лучший «Иван» 0.500 < 0.60" in caplog.text
    assert "Enrollment: сходство SPEAKER_00" in caplog.text


def test_enroll_speakers_outcome_for_matched(
    tmp_path: Path, install_loader
) -> None:
    install_loader({"ivan.wav": 0.11}, audio_blocks={(0.0, 3.0): 0.5})
    reference = _segment(tmp_path / "ivan.wav")
    audio = tmp_path / "call.wav"
    audio.write_bytes(b"")
    engine = _FakeEngine({50: [1, 0, 0], 11: [0.99, 0.01, 0]})

    outcome = enroll_speakers(
        speaker_segments=[SpeakerSegment(start=0.0, end=3.0, speaker_id="SPEAKER_00")],
        references={"Иван": (reference,)},
        audio_path=audio,
        min_similarity=0.6,
        engine=engine,
    )

    assert outcome.mapping == {"SPEAKER_00": "Иван"}
    assert outcome.best_candidates == {}
    assert outcome.speaker_count == 1


def test_assign_speaker_names_uses_waveform_without_decoding_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Переданный waveform (например, из денойза) отменяет повторный декод."""
    reference = _segment(tmp_path / "ivan.wav")
    audio = tmp_path / "call.wav"
    audio.write_bytes(b"")
    decoded: list[str] = []

    def loader(path: Path, *, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        decoded.append(path.name)
        return np.full(sample_rate, 0.11, dtype=np.float32)

    monkeypatch.setattr(enrollment, "load_waveform", loader)
    provided = np.zeros(20 * SAMPLE_RATE, dtype=np.float32)
    provided[0 : 3 * SAMPLE_RATE] = 0.5
    engine = _FakeEngine({50: [1, 0, 0], 11: [0.99, 0.01, 0]})

    mapping = assign_speaker_names(
        speaker_segments=[SpeakerSegment(start=0.0, end=3.0, speaker_id="SPEAKER_00")],
        references={"Иван": (reference,)},
        audio_path=audio,
        min_similarity=0.6,
        engine=engine,
        waveform=provided,
    )

    assert mapping == {"SPEAKER_00": "Иван"}
    # Декодирован только образец; аудио задачи взято из переданного waveform.
    assert decoded == ["ivan.wav"]


def test_assign_speaker_names_decodes_audio_without_waveform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Без waveform поведение прежнее: аудио задачи декодируется из файла."""
    reference = _segment(tmp_path / "ivan.wav")
    audio = tmp_path / "call.wav"
    audio.write_bytes(b"")
    decoded: list[str] = []

    def loader(path: Path, *, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        decoded.append(path.name)
        if path.name == "ivan.wav":
            return np.full(sample_rate, 0.11, dtype=np.float32)
        samples = np.zeros(20 * sample_rate, dtype=np.float32)
        samples[0 : 3 * sample_rate] = 0.5
        return samples

    monkeypatch.setattr(enrollment, "load_waveform", loader)
    engine = _FakeEngine({50: [1, 0, 0], 11: [0.99, 0.01, 0]})

    mapping = assign_speaker_names(
        speaker_segments=[SpeakerSegment(start=0.0, end=3.0, speaker_id="SPEAKER_00")],
        references={"Иван": (reference,)},
        audio_path=audio,
        min_similarity=0.6,
        engine=engine,
    )

    assert mapping == {"SPEAKER_00": "Иван"}
    assert decoded == ["ivan.wav", "call.wav"]


def test_reference_trims_pause_heavy_sample_to_speech(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Образец 8 с, речь только 5–7 с: окно 5 с содержит паузы → режется до речи."""
    ref_wf = np.zeros(8 * SAMPLE_RATE, dtype=np.float32)
    ref_wf[5 * SAMPLE_RATE : 7 * SAMPLE_RATE] = 0.42
    audio_wf = np.zeros(20 * SAMPLE_RATE, dtype=np.float32)
    audio_wf[0 : 3 * SAMPLE_RATE] = 0.42
    reference = _segment(tmp_path / "ivan.wav")
    audio = tmp_path / "call.wav"
    audio.write_bytes(b"")

    def loader(path: Path, *, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        return ref_wf if path.name == "ivan.wav" else audio_wf

    monkeypatch.setattr(enrollment, "load_waveform", loader)
    engine = _FakeEngine({42: [1, 0, 0]})

    mapping = assign_speaker_names(
        speaker_segments=[SpeakerSegment(start=0.0, end=3.0, speaker_id="SPEAKER_00")],
        references={"Иван": (reference,)},
        audio_path=audio,
        min_similarity=0.6,
        engine=engine,
    )

    assert mapping == {"SPEAKER_00": "Иван"}
