"""Тесты сборки конвейера (``run_pipeline``) с фиктивными компонентами."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from audio_transcriber.config.settings import AppConfig
from audio_transcriber.domain.enums import Device, ExportFormat
from audio_transcriber.domain.models import (
    Speaker,
    SpeakerSegment,
    TranscriptEntry,
    TranscriptionResult,
    TranscriptionSegment,
)
from audio_transcriber.merging.aligner import OverlapSegmentMerger
from audio_transcriber.pipeline import run_pipeline
from audio_transcriber.progress import ProgressEvent
from audio_transcriber.utils.exceptions import ProcessingCancelled


class FakeRecognizer:
    def transcribe(self, audio_path: Path, *, language: str | None = None):
        return (
            [TranscriptionSegment(start=0.0, end=1.0, text="привет")],
            "ru",
            1.0,
        )


class FakeDiarizer:
    def diarize(
        self,
        audio_path: Path,
        *,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        waveform: object = None,
    ):
        return [SpeakerSegment(start=0.0, end=1.0, speaker_id="SPEAKER_00")]


class FakeMerger:
    def merge(self, transcription_segments, speaker_segments, known_speakers=None):
        speaker = Speaker(id="SPEAKER_00", display_name="Иван")
        entries = [TranscriptEntry(start=0.0, end=1.0, text="привет", speaker=speaker)]
        return entries, [speaker]


class FakeCorrector:
    def correct(self, entries):
        return [
            TranscriptEntry(
                start=entry.start,
                end=entry.end,
                text=entry.text.upper(),
                speaker=entry.speaker,
            )
            for entry in entries
        ]


class MultiSegmentMerger:
    """Возвращает несколько коротких реплик одного говорящего."""

    def merge(self, transcription_segments, speaker_segments, known_speakers=None):
        speaker = Speaker(id="SPEAKER_00", display_name="Иван")
        entries = [
            TranscriptEntry(start=0.0, end=1.0, text="привет", speaker=speaker),
            TranscriptEntry(start=1.1, end=2.0, text="мир", speaker=speaker),
        ]
        return entries, [speaker]


class RecordingCorrector:
    """Корректор, запоминающий входные реплики, чтобы проверить порядок этапов."""

    def __init__(self) -> None:
        self.seen: list[TranscriptEntry] = []

    def correct(self, entries):
        self.seen = list(entries)
        return entries


class ArtifactMerger:
    """Возвращает реплику-артефакт и обычную реплику одного говорящего."""

    def merge(self, transcription_segments, speaker_segments, known_speakers=None):
        speaker = Speaker(id="SPEAKER_00", display_name="Иван")
        entries = [
            TranscriptEntry(start=0.0, end=1.0, text="[АПЛОДИСМЕНТЫ]", speaker=speaker),
            TranscriptEntry(start=1.1, end=2.0, text="привет", speaker=speaker),
        ]
        return entries, [speaker]


class RecordingCleaner:
    """Очистка, запоминающая входные реплики, чтобы проверить порядок этапов."""

    def __init__(self) -> None:
        self.seen: list[TranscriptEntry] | None = None

    def clean(self, entries):
        self.seen = list(entries)
        return entries


def test_run_pipeline_merges_sentences_before_correction(audio_file: Path, tmp_path: Path) -> None:
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        export_formats=(ExportFormat.TXT,),
        enable_correction=True,
    )
    corrector = RecordingCorrector()

    result = run_pipeline(
        config,
        device=Device.CPU,
        recognizer=FakeRecognizer(),
        diarizer=FakeDiarizer(),
        merger=MultiSegmentMerger(),
        corrector=corrector,
    )

    # Корректор получил уже склеенную реплику.
    assert len(corrector.seen) == 1
    assert corrector.seen[0].text == "привет мир"
    assert len(result.entries) == 1
    assert result.entries[0].text == "привет мир"


def test_run_pipeline_applies_text_corrector(audio_file: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    config = AppConfig(
        input_file=audio_file,
        output_dir=output_dir,
        export_formats=(ExportFormat.TXT,),
        enable_correction=True,
    )

    result = run_pipeline(
        config,
        device=Device.CPU,
        recognizer=FakeRecognizer(),
        diarizer=FakeDiarizer(),
        merger=FakeMerger(),
        corrector=FakeCorrector(),
    )

    assert result.entries[0].text == "ПРИВЕТ"


class ExplodingDiarizer:
    """Диаризатор, который обязан не вызываться при отключённой диаризации."""

    def diarize(self, *args, **kwargs):
        raise AssertionError("диаризация не должна вызываться, когда она отключена")


def test_run_pipeline_skips_diarization_when_disabled(
    audio_file: Path, tmp_path: Path
) -> None:
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        export_formats=(ExportFormat.TXT,),
        diarization_enabled=False,
    )

    result = run_pipeline(
        config,
        device=Device.CPU,
        recognizer=FakeRecognizer(),
        diarizer=ExplodingDiarizer(),
    )

    # Реплики без говорящего, список говорящих пуст.
    assert len(result.entries) == 1
    assert result.entries[0].speaker is None
    assert result.speakers == []


def test_run_pipeline_skips_correction_when_disabled(audio_file: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    config = AppConfig(
        input_file=audio_file,
        output_dir=output_dir,
        export_formats=(ExportFormat.TXT,),
        enable_correction=False,
    )

    result = run_pipeline(
        config,
        device=Device.CPU,
        recognizer=FakeRecognizer(),
        diarizer=FakeDiarizer(),
        merger=FakeMerger(),
    )

    assert result.entries[0].text == "привет"


def test_run_pipeline_removes_artifact_entries_by_default(
    audio_file: Path, tmp_path: Path
) -> None:
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        export_formats=(ExportFormat.TXT,),
        diarization_enabled=False,
    )

    result = run_pipeline(
        config,
        device=Device.CPU,
        recognizer=FakeRecognizer(),
        merger=ArtifactMerger(),
    )

    assert [entry.text for entry in result.entries] == ["привет"]


def test_run_pipeline_keeps_artifacts_when_cleaning_disabled(
    audio_file: Path, tmp_path: Path
) -> None:
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        export_formats=(ExportFormat.TXT,),
        diarization_enabled=False,
        clean_artifacts=False,
    )

    result = run_pipeline(
        config,
        device=Device.CPU,
        recognizer=FakeRecognizer(),
        merger=ArtifactMerger(),
    )

    # Очистка выключена: пометка остаётся в тексте (склейка предложений
    # объединяет обе реплики одного говорящего).
    assert [entry.text for entry in result.entries] == ["[АПЛОДИСМЕНТЫ] привет"]


def test_run_pipeline_cleans_before_sentence_merger(audio_file: Path, tmp_path: Path) -> None:
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        export_formats=(ExportFormat.TXT,),
        diarization_enabled=False,
    )
    cleaner = RecordingCleaner()

    result = run_pipeline(
        config,
        device=Device.CPU,
        recognizer=FakeRecognizer(),
        merger=MultiSegmentMerger(),
        artifact_cleaner=cleaner,
    )

    # Очистка получила отдельные реплики (до склейки предложений).
    assert cleaner.seen is not None
    assert [entry.text for entry in cleaner.seen] == ["привет", "мир"]
    assert len(result.entries) == 1
    assert result.entries[0].text == "привет мир"


def test_run_pipeline_emits_clean_progress_event(audio_file: Path, tmp_path: Path) -> None:
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        export_formats=(ExportFormat.TXT,),
        diarization_enabled=False,
    )
    events = []

    run_pipeline(
        config,
        device=Device.CPU,
        recognizer=FakeRecognizer(),
        merger=FakeMerger(),
        on_progress=events.append,
    )

    assert any(event.stage == "clean" for event in events)


class RecordingRecognizer:
    """Распознаватель, запоминающий путь к аудио, который ему передали."""

    def __init__(self) -> None:
        self.seen: list[Path] = []

    def transcribe(self, audio_path: Path, *, language: str | None = None):
        self.seen.append(audio_path)
        return ([TranscriptionSegment(start=0.0, end=1.0, text="привет")], "ru", 1.0)


class RecordingDiarizer:
    """Диаризатор, запоминающий путь к аудио, который ему передали."""

    def __init__(self) -> None:
        self.seen: list[Path] = []
        self.seen_waveform: object = None

    def diarize(
        self,
        audio_path: Path,
        *,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        waveform: object = None,
    ):
        self.seen.append(audio_path)
        self.seen_waveform = waveform
        return [SpeakerSegment(start=0.0, end=1.0, speaker_id="SPEAKER_00")]


class FakeDenoiser:
    """Заглушка денойзера: возвращает заданный путь и считает вызовы/закрытия."""

    def __init__(self, output: Path | None = None) -> None:
        self.output = output
        self.calls: list[Path] = []
        self.closed = 0

    def denoise(self, input_path: Path) -> Path:
        self.calls.append(input_path)
        return self.output if self.output is not None else input_path

    def close(self) -> None:
        self.closed += 1


def test_run_pipeline_feeds_denoised_audio_to_asr_and_diarization(
    audio_file: Path, tmp_path: Path
) -> None:
    denoised = tmp_path / "denoised.wav"
    denoised.write_bytes(b"")
    denoiser = FakeDenoiser(output=denoised)
    recognizer = RecordingRecognizer()
    diarizer = RecordingDiarizer()
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        export_formats=(ExportFormat.TXT,),
    )

    run_pipeline(
        config,
        device=Device.CPU,
        recognizer=recognizer,
        diarizer=diarizer,
        merger=FakeMerger(),
        denoiser=denoiser,
    )

    assert denoiser.calls == [audio_file]
    assert recognizer.seen == [denoised]
    assert diarizer.seen == [denoised]
    # Временный файл освобождается по завершении этапов ASR/диаризации.
    assert denoiser.closed == 1


class WaveformDenoiser:
    """Денойзер, отдающий уже декодированный waveform (как DeepFilterDenoiser)."""

    def __init__(self, output: Path, waveform: object) -> None:
        self.output = output
        self.last_waveform = waveform
        self.closed = 0

    def denoise(self, input_path: Path) -> Path:
        return self.output

    def close(self) -> None:
        self.closed += 1


def test_run_pipeline_reuses_denoised_waveform_for_diarization(
    audio_file: Path, tmp_path: Path
) -> None:
    import numpy as np

    denoised = tmp_path / "denoised.wav"
    denoised.write_bytes(b"")
    waveform = np.arange(4, dtype=np.float32)
    denoiser = WaveformDenoiser(denoised, waveform)
    diarizer = RecordingDiarizer()
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        export_formats=(ExportFormat.TXT,),
        # Без кэша — проверяем прямой проброс waveform от денойзера.
        use_cache=False,
    )

    run_pipeline(
        config,
        device=Device.CPU,
        recognizer=FakeRecognizer(),
        diarizer=diarizer,
        merger=FakeMerger(),
        denoiser=denoiser,
    )

    # Диаризация получила и путь, и уже декодированный массив — повторного
    # декодирования WAV не будет (внутри pyannote load_waveform не вызовется).
    assert diarizer.seen == [denoised]
    assert diarizer.seen_waveform is waveform
    assert denoiser.closed == 1


def test_run_pipeline_decodes_audio_once_for_all_consumers(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Денойз выключен: один декод обслуживает диаризацию, enrollment и образцы."""
    import numpy as np

    from audio_transcriber import pipeline as pipeline_module

    waveform = np.full(16000, 0.5, dtype=np.float32)
    decoded: list[Path] = []

    def counting_loader(path: Path, *, sample_rate: int = 16000) -> object:
        decoded.append(path)
        return waveform

    monkeypatch.setattr(pipeline_module, "load_waveform", counting_loader)

    diarizer = RecordingDiarizer()
    seen: dict[str, object] = {}

    def fake_assign(**kwargs: object) -> dict[str, str]:
        seen["enrollment"] = kwargs["waveform"]
        return {}

    def fake_samples(
        result: object,
        *,
        audio_path: Path,
        output_dir: Path,
        waveform: object = None,
    ) -> dict[str, Path]:
        seen["samples"] = waveform
        return {}

    monkeypatch.setattr("audio_transcriber.pipeline.assign_speaker_names", fake_assign)
    monkeypatch.setattr("audio_transcriber.pipeline.extract_speaker_samples", fake_samples)

    reference = _reference_file(tmp_path)
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        export_formats=(ExportFormat.TXT,),
        denoise=False,
        use_cache=False,
        speaker_references={"Иван": (reference,)},
        voices_dir=tmp_path / "no_voices",
        export_speaker_samples=True,
    )

    run_pipeline(
        config,
        device=Device.CPU,
        recognizer=FakeRecognizer(),
        diarizer=diarizer,
        merger=OverlapSegmentMerger(),
    )

    # Ровно один декод на весь конвейер — все потребители получили один массив.
    assert decoded == [audio_file]
    assert diarizer.seen_waveform is waveform
    assert seen["enrollment"] is waveform
    assert seen["samples"] is waveform


def test_run_pipeline_passes_hybrid_embedder_to_enrollment(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Гибрид отдаёт enrollment свой CAM++-эмбеддер (одно пространство)."""
    from audio_transcriber.diarization.hybrid_engine import HybridSpeakerDiarizer

    reference = _reference_file(tmp_path)
    embedder = object()
    diarizer = HybridSpeakerDiarizer("cpu", embedder=embedder)
    monkeypatch.setattr(
        diarizer,
        "diarize",
        lambda *_a, **_k: [SpeakerSegment(start=0.0, end=3.0, speaker_id="SPEAKER_00")],
    )
    captured: dict[str, object] = {}

    def fake_assign(**kwargs: object) -> dict[str, str]:
        captured.update(kwargs)
        return {}

    monkeypatch.setattr("audio_transcriber.pipeline.assign_speaker_names", fake_assign)
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        export_formats=(ExportFormat.TXT,),
        denoise=False,
        use_cache=False,
        speaker_references={"Иван": (reference,)},
        voices_dir=tmp_path / "no_voices",
    )

    run_pipeline(
        config,
        device=Device.CPU,
        recognizer=FakeRecognizer(),
        diarizer=diarizer,
        merger=OverlapSegmentMerger(),
    )

    # Enrollment получил именно эмбеддер гибрида, а не движок по умолчанию.
    assert captured["engine"] is embedder


def test_run_pipeline_skips_audio_decode_when_no_stage_needs_it(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Если аудио не нужно ни диаризации, ни образцам — декода нет."""
    from audio_transcriber import pipeline as pipeline_module

    decoded: list[Path] = []

    def counting_loader(path: Path, *, sample_rate: int = 16000) -> object:
        decoded.append(path)
        raise AssertionError("декодирование не должно вызываться")

    monkeypatch.setattr(pipeline_module, "load_waveform", counting_loader)
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        export_formats=(ExportFormat.TXT,),
        denoise=False,
        use_cache=False,
        diarization_enabled=False,
        export_speaker_samples=False,
    )

    run_pipeline(
        config,
        device=Device.CPU,
        recognizer=FakeRecognizer(),
        merger=FakeMerger(),
    )

    assert decoded == []


def test_run_pipeline_skips_denoise_when_disabled(audio_file: Path, tmp_path: Path) -> None:
    recognizer = RecordingRecognizer()
    diarizer = RecordingDiarizer()
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        export_formats=(ExportFormat.TXT,),
        denoise=False,
    )
    events = []

    run_pipeline(
        config,
        device=Device.CPU,
        recognizer=recognizer,
        diarizer=diarizer,
        merger=FakeMerger(),
        on_progress=events.append,
    )

    # Денойз выключен — ASR и диаризация получают исходный файл, события нет.
    assert recognizer.seen == [audio_file]
    assert diarizer.seen == [audio_file]
    assert not any(event.stage == "denoise" for event in events)


def test_run_pipeline_emits_denoise_progress_event(audio_file: Path, tmp_path: Path) -> None:
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        export_formats=(ExportFormat.TXT,),
        diarization_enabled=False,
    )
    events = []

    run_pipeline(
        config,
        device=Device.CPU,
        recognizer=FakeRecognizer(),
        merger=FakeMerger(),
        denoiser=FakeDenoiser(output=audio_file),
        on_progress=events.append,
    )

    assert any(event.stage == "denoise" for event in events)


class ProgressDenoiser:
    """Денойзер, эмитящий прогресс через прокинутый конвейером колбэк."""

    def __init__(self, output: Path) -> None:
        self.output = output
        self.on_progress = None
        self.closed = 0

    def denoise(self, input_path: Path) -> Path:
        if self.on_progress is not None:
            self.on_progress(
                ProgressEvent("denoise", "Шумоподавление", fraction=0.5)
            )
        return self.output

    def close(self) -> None:
        self.closed += 1


def test_run_pipeline_forwards_progress_callback_to_denoiser(
    audio_file: Path, tmp_path: Path
) -> None:
    """Колбэк прогресса доходит до денойзера (в т.ч. через ``CachingDenoiser``)."""
    denoiser = ProgressDenoiser(audio_file)
    events: list[ProgressEvent] = []
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        export_formats=(ExportFormat.TXT,),
        diarization_enabled=False,
        use_cache=True,
    )

    run_pipeline(
        config,
        device=Device.CPU,
        recognizer=FakeRecognizer(),
        merger=FakeMerger(),
        denoiser=denoiser,
        on_progress=events.append,
    )

    # Обёртка прокинула колбэк во внутренний денойзер, и его событие добралось
    # до приёмника конвейера.
    assert denoiser.on_progress is not None
    assert any(
        event.stage == "denoise" and event.fraction == 0.5 for event in events
    )


def test_run_pipeline_continues_when_denoiser_degrades_softly(
    audio_file: Path, tmp_path: Path
) -> None:
    # Денойзер вернул исходный путь (движок недоступен) — конвейер не падает.
    recognizer = RecordingRecognizer()
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        export_formats=(ExportFormat.TXT,),
    )

    result = run_pipeline(
        config,
        device=Device.CPU,
        recognizer=recognizer,
        diarizer=FakeDiarizer(),
        merger=FakeMerger(),
        denoiser=FakeDenoiser(output=None),
    )

    assert recognizer.seen == [audio_file]
    assert len(result.entries) == 1


def test_run_pipeline_closes_denoiser_on_error(audio_file: Path, tmp_path: Path) -> None:
    """Даже если распознавание падает, временный денойзенный файл удаляется."""

    class ExplodingRecognizer:
        def transcribe(self, audio_path: Path, *, language: str | None = None):
            raise RuntimeError("ASR взорвался")

    denoiser = FakeDenoiser(output=audio_file)
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        export_formats=(ExportFormat.TXT,),
        diarization_enabled=False,
    )

    with pytest.raises(RuntimeError):
        run_pipeline(
            config,
            device=Device.CPU,
            recognizer=ExplodingRecognizer(),
            merger=FakeMerger(),
            denoiser=denoiser,
        )

    assert denoiser.closed == 1



# --- Пакет 5 «enrollment-диаризация»: имена по образцам голоса --------------


def _reference_file(tmp_path: Path) -> Path:
    reference = tmp_path / "voice.wav"
    reference.write_bytes(b"")
    return reference


def test_run_pipeline_enrollment_names_override_speaker_names(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = _reference_file(tmp_path)
    captured: dict[str, object] = {}

    def fake_assign(**kwargs: object) -> dict[str, str]:
        captured.update(kwargs)
        return {"SPEAKER_00": "Иван"}

    monkeypatch.setattr("audio_transcriber.pipeline.assign_speaker_names", fake_assign)
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        export_formats=(ExportFormat.TXT,),
        speaker_names={"SPEAKER_00": "Пётр"},
        speaker_references={"Иван": (reference,)},
        enrollment_min_similarity=0.55,
        voices_dir=tmp_path / "no_voices",
    )

    result = run_pipeline(
        config,
        device=Device.CPU,
        recognizer=FakeRecognizer(),
        diarizer=FakeDiarizer(),
        merger=OverlapSegmentMerger(),
    )

    # Enrollment-имя приоритетнее переименования по индексу.
    assert result.entries[0].speaker is not None
    assert result.entries[0].speaker.display_name == "Иван"
    assert result.speakers[0].display_name == "Иван"
    # Параметры сопоставления переданы в движок enrollment.
    assert captured["min_similarity"] == 0.55
    assert captured["references"] == {"Иван": (reference,)}
    assert captured["audio_path"] == audio_file


def test_run_pipeline_falls_back_to_speaker_names_when_enrollment_misses(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = _reference_file(tmp_path)
    monkeypatch.setattr("audio_transcriber.pipeline.assign_speaker_names", lambda **_: {})
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        export_formats=(ExportFormat.TXT,),
        speaker_names={"SPEAKER_00": "Пётр"},
        speaker_references={"Иван": (reference,)},
        voices_dir=tmp_path / "no_voices",
    )

    result = run_pipeline(
        config,
        device=Device.CPU,
        recognizer=FakeRecognizer(),
        diarizer=FakeDiarizer(),
        merger=OverlapSegmentMerger(),
    )

    assert result.entries[0].speaker is not None
    assert result.entries[0].speaker.display_name == "Пётр"


def test_run_pipeline_skips_enrollment_without_references(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []

    def fake_assign(**kwargs: object) -> dict[str, str]:
        calls.append(kwargs)
        return {}

    monkeypatch.setattr("audio_transcriber.pipeline.assign_speaker_names", fake_assign)
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        export_formats=(ExportFormat.TXT,),
        voices_dir=tmp_path / "no_voices",
    )

    run_pipeline(
        config,
        device=Device.CPU,
        recognizer=FakeRecognizer(),
        diarizer=FakeDiarizer(),
        merger=OverlapSegmentMerger(),
    )

    assert calls == []


# --- Пакет «протокол по кнопке»: protocol_auto / generate_protocol ----------


class RecordingLlm:
    """Минимальный LLM-клиент: запоминает закрытие, chat не вызывается."""

    def __init__(self) -> None:
        self.closed = 0

    def chat(self, messages):  # pragma: no cover - не должен вызываться
        raise AssertionError("LLM не должна запрашиваться в этом сценарии")

    def close(self) -> None:
        self.closed += 1


def test_run_pipeline_without_protocol_skips_summary_and_export(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary_calls: list[int] = []
    monkeypatch.setattr(
        "audio_transcriber.llm.summary.summarize_meeting",
        lambda *_args, **_kwargs: summary_calls.append(1) or "РЕЗЮМЕ",
    )
    output_dir = tmp_path / "out"
    config = AppConfig(
        input_file=audio_file,
        output_dir=output_dir,
        export_formats=(ExportFormat.TXT, ExportFormat.DOCX),
        llm_enabled=True,
        llm_summary=True,
        protocol_auto=False,
        glossary_enabled=False,
        export_speaker_samples=False,
    )

    result = run_pipeline(
        config,
        device=Device.CPU,
        recognizer=FakeRecognizer(),
        diarizer=FakeDiarizer(),
        merger=FakeMerger(),
        llm_client=RecordingLlm(),
    )

    # Стенограмма построена, но резюме не считалось и файлы не писались.
    assert [entry.text for entry in result.entries] == ["привет"]
    assert result.summary is None
    assert summary_calls == []
    assert not (output_dir / f"{audio_file.stem}.txt").exists()
    assert not (output_dir / f"{audio_file.stem}.docx").exists()


def test_run_pipeline_with_protocol_auto_writes_files(
    audio_file: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "out"
    config = AppConfig(
        input_file=audio_file,
        output_dir=output_dir,
        export_formats=(ExportFormat.TXT,),
        protocol_auto=True,
        export_speaker_samples=False,
    )

    run_pipeline(
        config,
        device=Device.CPU,
        recognizer=FakeRecognizer(),
        diarizer=FakeDiarizer(),
        merger=FakeMerger(),
    )

    assert (output_dir / f"{audio_file.stem}.txt").is_file()


def _protocol_result(audio_file: Path) -> TranscriptionResult:
    speaker = Speaker(id="SPEAKER_00", display_name="Пётр")
    return TranscriptionResult(
        source_path=audio_file,
        language="ru",
        duration=1.0,
        entries=[TranscriptEntry(start=0.0, end=1.0, text="привет", speaker=speaker)],
        speakers=[speaker],
        low_confidence_threshold=-1.0,
    )


def test_generate_protocol_recomputes_summary_and_exports(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from audio_transcriber.protocol import generate_protocol

    seen_speakers: dict[str, list[str]] = {}

    def fake_summarize(entries, speakers, *, llm, max_chunk_chars=None):
        seen_speakers["names"] = [speaker.display_name for speaker in speakers]
        return "РЕЗЮМЕ"

    monkeypatch.setattr("audio_transcriber.protocol.summarize_meeting", fake_summarize)

    exported: dict[Path, object] = {}

    class FakeExporter:
        def export(self, result, output_path) -> None:
            exported[Path(output_path)] = result

    monkeypatch.setattr(
        "audio_transcriber.protocol.create_exporter", lambda _fmt: FakeExporter()
    )

    output_dir = tmp_path / "out"
    config = AppConfig(
        input_file=audio_file,
        output_dir=output_dir,
        export_formats=(ExportFormat.TXT, ExportFormat.DOCX),
        llm_enabled=True,
        llm_summary=True,
        protocol_auto=False,
        timeline=False,
    )
    client = RecordingLlm()

    artifacts = generate_protocol(config, _protocol_result(audio_file), llm_client=client)

    assert artifacts.summary == "РЕЗЮМЕ"
    assert {path.name for path in artifacts.paths} == {
        f"{audio_file.stem}.txt",
        f"{audio_file.stem}.docx",
    }
    assert all(result.summary == "РЕЗЮМЕ" for result in exported.values())
    # Экспорт получил текущие имена говорящих, а не исходные/пустые.
    first = next(iter(exported.values()))
    assert first.entries[0].speaker.display_name == "Пётр"  # type: ignore[union-attr]
    assert seen_speakers["names"] == ["Пётр"]
    # Клиент передан снаружи — вызывающий владеет им и сам закрывает.
    assert client.closed == 0


def test_generate_protocol_without_llm_exports_without_summary(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from audio_transcriber.protocol import generate_protocol

    class FakeExporter:
        def export(self, result, output_path) -> None:
            Path(output_path).write_text("ok", encoding="utf-8")

    monkeypatch.setattr(
        "audio_transcriber.protocol.create_exporter", lambda _fmt: FakeExporter()
    )
    output_dir = tmp_path / "out"
    config = AppConfig(
        input_file=audio_file,
        output_dir=output_dir,
        export_formats=(ExportFormat.TXT,),
        llm_enabled=False,
        protocol_auto=False,
        timeline=False,
    )

    artifacts = generate_protocol(config, _protocol_result(audio_file))

    assert artifacts.summary is None
    assert artifacts.paths == (output_dir / f"{audio_file.stem}.txt",)
    assert artifacts.paths[0].is_file()


# --- Отмена конвейера (#22) -----------------------------------------------


class ToggleRecognizer:
    """Распознаватель, который при желании взводит флаг отмены во время стадии."""

    def __init__(self, cancel_event: threading.Event | None = None) -> None:
        self.cancel_event = cancel_event
        self.calls = 0

    def transcribe(self, audio_path: Path, *, language: str | None = None):
        self.calls += 1
        if self.cancel_event is not None:
            self.cancel_event.set()
        return ([TranscriptionSegment(0.0, 1.0, "привет")], "ru", 1.0)


def test_run_pipeline_cancel_before_start_stops_without_error(
    audio_file: Path, tmp_path: Path
) -> None:
    """Взведённый до старта флаг прекращает конвейер до первой стадии."""
    cancel_event = threading.Event()
    cancel_event.set()
    recognizer = ToggleRecognizer(cancel_event)
    config = AppConfig(input_file=audio_file, output_dir=tmp_path / "out")

    with pytest.raises(ProcessingCancelled):
        run_pipeline(
            config,
            device=Device.CPU,
            recognizer=recognizer,
            diarizer=FakeDiarizer(),
            merger=FakeMerger(),
            cancel_event=cancel_event,
        )

    # Ни одна стадия не запускалась — отмена проверяется до работы.
    assert recognizer.calls == 0


def test_run_pipeline_cancel_during_asr_keeps_completed_cache(
    audio_file: Path, tmp_path: Path
) -> None:
    """Отмена после ASR: исключение ProcessingCancelled, кэш стадии сохранён.

    Повторный запуск без отмены переиспользует кэш ASR — распознавание не
    выполняется заново, что и лежит в основе возобновления.
    """
    output_dir = tmp_path / "out"
    cancel_event = threading.Event()
    config = AppConfig(
        input_file=audio_file,
        output_dir=output_dir,
        use_cache=True,
    )
    first = ToggleRecognizer(cancel_event)

    with pytest.raises(ProcessingCancelled):
        run_pipeline(
            config,
            device=Device.CPU,
            recognizer=first,
            diarizer=FakeDiarizer(),
            merger=FakeMerger(),
            cancel_event=cancel_event,
        )

    assert first.calls == 1
    # ASR успел закэшироваться до срабатывания контрольной точки.
    assert list((output_dir / ".cache").glob("asr-*.json"))

    # Возобновление: свежий флаг, тот же тип движка — кэш ASR переиспользован.
    resumed = ToggleRecognizer()
    result = run_pipeline(
        config,
        device=Device.CPU,
        recognizer=resumed,
        diarizer=FakeDiarizer(),
        merger=FakeMerger(),
    )

    assert resumed.calls == 0
    assert result.entries



# --- EEND-движки (nemo-speech, hybrid): enrollment всё равно выполняется ---


class NoEmbeddingDiarizer(FakeDiarizer):
    """Заглушка EEND-движка (Sortformer): per-speaker эмбеддингов у него нет.

    Раньше конвейер пропускал enrollment для таких движков по флагу
    ``supports_enrollment``. Теперь enrollment использует собственный
    embedding-движок и выполняется независимо от движка диаризации.
    """


@pytest.mark.parametrize("engine", ["nemo-speech", "hybrid"])
def test_run_pipeline_runs_enrollment_for_eend_engines(
    engine: str,
    audio_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enrollment не гейтится типом движка: имена по образцам присваиваются."""
    reference = _reference_file(tmp_path)
    calls: list[dict[str, object]] = []

    def fake_assign(**kwargs: object) -> dict[str, str]:
        calls.append(kwargs)
        return {"SPEAKER_00": "Иван"}

    monkeypatch.setattr("audio_transcriber.pipeline.assign_speaker_names", fake_assign)
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        export_formats=(ExportFormat.TXT,),
        speaker_names={"SPEAKER_00": "Пётр"},
        speaker_references={"Иван": (reference,)},
        voices_dir=tmp_path / "no_voices",
        diarization_engine=engine,
    )

    result = run_pipeline(
        config,
        device=Device.CPU,
        recognizer=FakeRecognizer(),
        diarizer=NoEmbeddingDiarizer(),
        merger=OverlapSegmentMerger(),
    )

    # Enrollment вызван, несмотря на EEND-движок; имя образца приоритетнее.
    assert len(calls) == 1
    assert calls[0]["references"] == {"Иван": (reference,)}
    assert result.entries[0].speaker is not None
    assert result.entries[0].speaker.display_name == "Иван"
