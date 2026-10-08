"""Поиск дубликатов в библиотеке образцов голоса (``voices/``, issue #39).

В каталоге копится «мусор»: один и тот же образец под разными именами
(``Иван.wav`` и ``Иван Клон.wav``) или с суффиксом ``(N)`` (#19). Модуль ищет
такие записи в три слоя, но группировку «одной личности» намеренно ограничивает,
чтобы не склеивать разных людей (issue #116):

1. **Точные дубликаты** — совпадает sha256 содержимого файла. Это единственный
   слой, который может свести разные имена: одинаковые байты — буквально один
   и тот же файл, личность тут ни при чём.
2. **Почти одинаковые по аудио** — спектральный отпечаток сравнивается только
   для образцов **одного имени** (``Иван.wav`` ↔ ``Иван (2).wav``): это один
   человек по определению, и отпечаток ловит перекодированные/нормализованные
   копии. Для разных имён отпечаток **не применяется**: он отражает форму
   спектра/тембр/канал, а не личность, и разные голоса в одних акустических
   условиях дают высокую корреляцию (ложные срабатывания #116).
3. **Почти одинаковые по эмбеддингам** — единственный безопасный способ
   связать **разные имена** (один человек под разными именами): L2-нормированные
   speaker-эмбеддинги (CAM++) сравниваются по косинусу с высоким порогом
   (``embedding_threshold``, по умолчанию 0.95). Если движок эмбеддингов
   недоступен, кросс-именные группы не предлагаются вовсе.

Дедупликация ничего не удаляет молча: модуль лишь возвращает группы
дубликатов с оценкой и подсказкой «кого оставить». Удаление/объединение
выполняет вызывающий код (веб-API) по явному действию пользователя.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from audio_transcriber.diarization.embeddings import l2_normalize
from audio_transcriber.diarization.voices import (
    base_sample_name,
    collect_voice_library,
    sample_index,
)
from audio_transcriber.utils.audio import SAMPLE_RATE, load_waveform
from audio_transcriber.utils.playback import read_duration

logger = logging.getLogger(__name__)

#: Порог косинусной близости аудио-отпечатков для «почти одинаковых».
#: Применяется только внутри одного имени: для разных имён форма спектра не
#: различает личность (issue #116).
DEFAULT_NEAR_THRESHOLD = 0.9

#: Порог косинусной близости speaker-эмбеддингов для «почти одинаковых».
#: Единственный слой, связывающий разные имена, поэтому порог высокий:
#: у CAM++ косинус разных людей заметно ниже (issue #116).
DEFAULT_EMBEDDING_THRESHOLD = 0.95

#: Размер чанка чтения файла при подсчёте хеша (байты).
_CHUNK_SIZE = 1 << 20

#: Число частотных полос спектрального отпечатка.
_FINGERPRINT_BANDS = 24

#: Параметры STFT для спектрального отпечатка.
_FINGERPRINT_FFT = 1024
_FINGERPRINT_HOP = 512

#: Минимальная частота отпечатка (Гц) — ниже почти нет полезной речи.
_FINGERPRINT_FMIN = 50.0


def file_digest(path: Path) -> str:
    """sha256 содержимого файла (потоково, без чтения целиком в память)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _band_matrix(
    fft_size: int, sample_rate: int, bands: int
) -> np.ndarray:
    """Матрица ``(bands, fft_size//2+1)`` суммирования спектра по лог-полосам."""
    freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
    edges = np.geomspace(_FINGERPRINT_FMIN, sample_rate / 2.0, bands + 1)
    indices = np.clip(np.searchsorted(edges, freqs, side="right") - 1, 0, bands - 1)
    matrix = np.zeros((bands, freqs.size), dtype=np.float32)
    matrix[indices, np.arange(freqs.size)] = 1.0
    return matrix


def audio_fingerprint(
    waveform: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
    bands: int = _FINGERPRINT_BANDS,
) -> np.ndarray | None:
    """Спектральный отпечаток сигнала: L2-нормированный вектор формы спектра.

    Сигнал режется на окна, для каждого считается спектр мощности, энергии
    суммируются в лог-равномерные полосы и усредняются по времени; результат
    центрируется (вычитается среднее) и L2-нормируется. Центрирование делает
    сходство корреляцией формы спектра и убирает «общий фон» — разные голоса
    дают низкую корреляцию, один голос при иной громкости — высокую.
    Пустой сигнал/чистая тишина дают ``None`` (сравнивать нечего).
    """
    samples = np.asarray(waveform, dtype=np.float64).reshape(-1)
    if samples.size == 0:
        return None
    if samples.size < _FINGERPRINT_FFT:
        samples = np.pad(samples, (0, _FINGERPRINT_FFT - samples.size))
    window = np.hanning(_FINGERPRINT_FFT)
    frames = np.lib.stride_tricks.sliding_window_view(samples, _FINGERPRINT_FFT)[
        ::_FINGERPRINT_HOP
    ]
    spectra = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    matrix = _band_matrix(_FINGERPRINT_FFT, sample_rate, bands)
    energy = np.log1p(spectra @ matrix.T).mean(axis=0)
    centred = energy - float(energy.mean())
    norm = float(np.linalg.norm(centred))
    if norm <= 1e-12:
        return None
    return (centred / norm).astype(np.float32)


def cosine_similarity(left: np.ndarray | None, right: np.ndarray | None) -> float:
    """Косинусная близость двух L2-нормированных векторов (0 при пустом)."""
    if left is None or right is None:
        return 0.0
    return float(np.dot(left, right))


@dataclass(frozen=True, slots=True)
class SampleSignature:
    """Признаки одного образца: имя, путь, размер, длительность, хеш, векторы."""

    name: str
    path: Path
    size: int
    duration: float
    digest: str
    fingerprint: np.ndarray | None
    embedding: np.ndarray | None

    @property
    def filename(self) -> str:
        """Имя файла образца (уникальный идентификатор в API)."""
        return self.path.name


def _read_signature(
    name: str,
    path: Path,
    *,
    embedder: object | None,
    compute_fingerprint: bool,
) -> SampleSignature | None:
    """Собирает признаки образца; ``None`` при ошибке чтения файла."""
    try:
        size = path.stat().st_size
    except OSError as exc:
        logger.warning("Дедуп голосов: файл недоступен %s: %s", path, exc)
        return None
    try:
        digest = file_digest(path)
    except OSError as exc:
        logger.warning("Дедуп голосов: не удалось прочитать %s: %s", path, exc)
        return None

    fingerprint: np.ndarray | None = None
    embedding: np.ndarray | None = None
    waveform: np.ndarray | None = None
    if compute_fingerprint or embedder is not None:
        try:
            waveform = load_waveform(path)
        except Exception as exc:  # noqa: BLE001 — битый файл: хеша достаточно
            logger.warning("Дедуп голосов: не удалось декодировать %s: %s", path, exc)
    if compute_fingerprint and waveform is not None:
        fingerprint = audio_fingerprint(waveform)
    if embedder is not None and waveform is not None and waveform.size:
        try:
            embedding = l2_normalize(embedder.embed(waveform))  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 — эмбеддер не критичен для аудита
            logger.warning("Дедуп голосов: эмбеддинг недоступен для %s: %s", path, exc)
            embedding = None
    try:
        duration = round(read_duration(path), 2)
    except Exception:  # noqa: BLE001 — длительность только для показа
        duration = 0.0
    return SampleSignature(
        name=name,
        path=path,
        size=size,
        duration=duration,
        digest=digest,
        fingerprint=fingerprint,
        embedding=embedding,
    )


def load_signatures(
    directory: Path | None,
    *,
    embedder: object | None = None,
    compute_fingerprints: bool = True,
) -> list[SampleSignature]:
    """Признаки всех образцов библиотеки (по группам имён)."""
    signatures: list[SampleSignature] = []
    for name, paths in collect_voice_library(directory).items():
        for path in paths:
            signature = _read_signature(
                name,
                path,
                embedder=embedder,
                compute_fingerprint=compute_fingerprints,
            )
            if signature is not None:
                signatures.append(signature)
    return signatures


def _keeper_index(indices: Sequence[int], signatures: Sequence[SampleSignature]) -> int:
    """Индекс «канонического» образца для предложения «оставить этот».

    Приоритет: имя с наибольшим числом образцов в группе (более «основная»
    личность), затем основной файл без суффикса ``(N)``, затем более длинное
    аудио (больше данных для эмбеддинга), затем имя файла — для стабильности.
    """
    name_counts: dict[str, int] = {}
    for index in indices:
        name = signatures[index].name
        name_counts[name] = name_counts.get(name, 0) + 1

    def key(index: int) -> tuple[int, int, float, str]:
        signature = signatures[index]
        return (
            -name_counts[signature.name],
            sample_index(signature.path.stem),
            -signature.duration,
            signature.filename.casefold(),
        )

    return min(indices, key=key)


@dataclass(frozen=True, slots=True)
class DuplicateMember:
    """Участник группы дубликатов (плоское представление для API)."""

    name: str
    path: Path
    duration: float
    size: int

    @property
    def filename(self) -> str:
        """Имя файла образца."""
        return self.path.name

    def as_dict(self) -> dict[str, object]:
        """Плоское представление участника (имя, файл, длительность, размер)."""
        return {
            "name": self.name,
            "filename": self.filename,
            "duration": self.duration,
            "size": self.size,
        }


@dataclass(frozen=True, slots=True)
class DuplicateGroup:
    """Группа дубликатов: тип совпадения, оценка, участники и «кого оставить».

    ``kind`` — ``exact`` (побайтово одинаковые), ``audio`` (похожи по
    спектральному отпечатку) или ``embedding`` (похожи по speaker-эмбеддингу).
    ``names`` — все имена людей, затронутые группой (её длина > 1 означает
    «один человек под разными именами» — стоит объединить).
    """

    kind: str
    score: float
    members: tuple[DuplicateMember, ...]
    keep: DuplicateMember

    @property
    def names(self) -> tuple[str, ...]:
        """Уникальные имена людей группы (в порядке появления)."""
        seen: list[str] = []
        for member in self.members:
            if member.name not in seen:
                seen.append(member.name)
        return tuple(seen)

    def as_dict(self) -> dict[str, object]:
        """Плоское представление группы для API."""
        return {
            "kind": self.kind,
            "score": round(self.score, 4),
            "names": list(self.names),
            "keep": self.keep.filename,
            "members": [member.as_dict() for member in self.members],
        }


class _UnionFind:
    """Простой DSU для склейки кластеров в группы."""

    def __init__(self, size: int) -> None:
        self._parent = list(range(size))
        self._rank = [0] * size

    def find(self, item: int) -> int:
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self._rank[left_root] < self._rank[right_root]:
            left_root, right_root = right_root, left_root
        self._parent[right_root] = left_root
        if self._rank[left_root] == self._rank[right_root]:
            self._rank[left_root] += 1


def find_duplicate_groups(
    directory: Path | None,
    *,
    near_threshold: float = DEFAULT_NEAR_THRESHOLD,
    embedding_threshold: float = DEFAULT_EMBEDDING_THRESHOLD,
    embedder: object | None = None,
    compute_fingerprints: bool = True,
) -> list[DuplicateGroup]:
    """Ищет группы дубликатов в библиотеке ``directory``.

    Сначала образцы склеиваются по sha256 в точные кластеры. Затем кластеры
    сравниваются попарно (по представителю):

    * совпадение **эмбеддингов** (если движок доступен) выше
      ``embedding_threshold`` связывает кластеры — в том числе под разными
      именами;
    * совпадение **аудио-отпечатка** выше ``near_threshold`` связывает кластеры
      только если у них есть общее имя (один человек под одним именем), иначе
      отпечаток игнорируется — он не различает личности (issue #116).

    Связанные кластеры объединяются в одну группу (union-find): так ловятся и
    внутригрупповые дубликаты (#19), и «один человек под разными именами» — но
    лишь по эмбеддингам. Если эмбеддинги недоступны, кросс-именные группы не
    предлагаются вовсе. Возвращается отсортированный список: сначала точные,
    затем по убыванию оценки. Ничего не удаляется — только предложение.
    """
    signatures = load_signatures(
        directory, embedder=embedder, compute_fingerprints=compute_fingerprints
    )
    if len(signatures) < 2:
        return []

    by_digest: dict[str, list[int]] = {}
    for index, signature in enumerate(signatures):
        by_digest.setdefault(signature.digest, []).append(index)
    clusters = list(by_digest.values())
    reps = [_keeper_index(indices, signatures) for indices in clusters]
    cluster_names = [
        {signatures[index].name for index in indices} for indices in clusters
    ]

    union = _UnionFind(len(clusters))
    edge_kind: dict[tuple[int, int], str] = {}
    edge_score: dict[tuple[int, int], float] = {}
    for left in range(len(clusters)):
        for right in range(left + 1, len(clusters)):
            a, b = signatures[reps[left]], signatures[reps[right]]
            embedding_score = cosine_similarity(a.embedding, b.embedding)
            embeddings_ready = a.embedding is not None and b.embedding is not None
            if embeddings_ready and embedding_score >= embedding_threshold:
                kind, score = "embedding", embedding_score
            elif (
                bool(cluster_names[left] & cluster_names[right])
                and compute_fingerprints
            ):
                audio_score = cosine_similarity(a.fingerprint, b.fingerprint)
                if audio_score < near_threshold:
                    continue
                kind, score = "audio", audio_score
            else:
                continue
            union.union(left, right)
            edge_kind[(left, right)] = kind
            edge_score[(left, right)] = score

    components: dict[int, list[int]] = {}
    for index in range(len(clusters)):
        components.setdefault(union.find(index), []).append(index)

    groups: list[DuplicateGroup] = []
    for members in components.values():
        if len(members) == 1:
            indices = clusters[members[0]]
            if len(indices) < 2:
                continue
            group = _build_group(
                signatures, indices, kind="exact", score=1.0
            )
        else:
            member_indices = [index for item in members for index in clusters[item]]
            kinds = {
                edge_kind[(left, right)]
                for left in members
                for right in members
                if (left, right) in edge_kind
            }
            scores = [
                edge_score[(left, right)]
                for left in members
                for right in members
                if (left, right) in edge_score
            ]
            group = _build_group(
                signatures,
                member_indices,
                kind="embedding" if "embedding" in kinds else "audio",
                score=min(scores) if scores else 0.0,
            )
        groups.append(group)

    priority = {"exact": 0, "embedding": 1, "audio": 2}
    groups.sort(
        key=lambda group: (
            priority.get(group.kind, 3),
            -group.score,
            group.names,
        )
    )
    return groups


def _build_group(
    signatures: Sequence[SampleSignature],
    indices: Iterable[int],
    *,
    kind: str,
    score: float,
) -> DuplicateGroup:
    """Собирает :class:`DuplicateGroup` из индексов образцов."""
    ordered = sorted(
        set(indices),
        key=lambda index: (
            signatures[index].name.casefold(),
            sample_index(signatures[index].path.stem),
            signatures[index].filename.casefold(),
        ),
    )
    members = tuple(
        DuplicateMember(
            name=signatures[index].name,
            path=signatures[index].path,
            duration=signatures[index].duration,
            size=signatures[index].size,
        )
        for index in ordered
    )
    keeper_index = _keeper_index(ordered, signatures)
    keep = DuplicateMember(
        name=signatures[keeper_index].name,
        path=signatures[keeper_index].path,
        duration=signatures[keeper_index].duration,
        size=signatures[keeper_index].size,
    )
    return DuplicateGroup(kind=kind, score=score, members=members, keep=keep)


@dataclass(frozen=True, slots=True)
class SimilarSample:
    """Похожий образец, найденный при добавлении нового (issue #39)."""

    name: str
    filename: str
    kind: str
    score: float

    def as_dict(self) -> dict[str, object]:
        """Плоское представление для API."""
        return {
            "name": self.name,
            "filename": self.filename,
            "kind": self.kind,
            "score": round(self.score, 4),
        }


def find_similar_samples(
    target: Path,
    directory: Path | None,
    *,
    near_threshold: float = DEFAULT_NEAR_THRESHOLD,
    embedding_threshold: float = DEFAULT_EMBEDDING_THRESHOLD,
    embedder: object | None = None,
) -> list[SimilarSample]:
    """Похожие на ``target`` образцы библиотеки (для предупреждения при загрузке).

    Точное совпадение (тот же sha256) имеет оценку 1.0 и тип ``exact``.
    Аудио-отпечаток учитывается только при совпадении имени (тот же человек),
    эмбеддинг — для любых имён; так предупреждение не срабатывает на разных
    людей с похожим тембром (issue #116). Возвращает совпадения по убыванию
    оценки. Ошибки чтения файла не роняют вызывающий код — просто нет совпадений.
    """
    target_path = Path(target)
    candidates = load_signatures(
        directory, embedder=embedder, compute_fingerprints=True
    )
    target_signature = _read_signature(
        base_sample_name(target_path.stem),
        target_path,
        embedder=embedder,
        compute_fingerprint=True,
    )
    if target_signature is None:
        return []
    try:
        resolved_target = target_path.resolve()
    except OSError:
        resolved_target = target_path

    matches: list[SimilarSample] = []
    for candidate in candidates:
        try:
            same_file = candidate.path.resolve() == resolved_target
        except OSError:
            same_file = candidate.path == target_path
        if same_file:
            continue
        if candidate.digest == target_signature.digest:
            matches.append(
                SimilarSample(
                    name=candidate.name,
                    filename=candidate.filename,
                    kind="exact",
                    score=1.0,
                )
            )
            continue
        embedding_score = cosine_similarity(
            target_signature.embedding, candidate.embedding
        )
        audio_score = cosine_similarity(
            target_signature.fingerprint, candidate.fingerprint
        )
        embeddings_ready = (
            target_signature.embedding is not None and candidate.embedding is not None
        )
        same_name = candidate.name == target_signature.name
        if embeddings_ready and embedding_score >= embedding_threshold:
            matches.append(
                SimilarSample(
                    name=candidate.name,
                    filename=candidate.filename,
                    kind="embedding",
                    score=embedding_score,
                )
            )
        elif same_name and audio_score >= near_threshold:
            matches.append(
                SimilarSample(
                    name=candidate.name,
                    filename=candidate.filename,
                    kind="audio",
                    score=audio_score,
                )
            )
    matches.sort(key=lambda item: (-item.score, item.name.casefold(), item.filename))
    return matches
