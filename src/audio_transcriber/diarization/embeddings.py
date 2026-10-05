"""Общие speaker-эмбеддинги на sherpa-onnx (3D-Speaker CAM++).

Модуль вынесен из оценщика числа говорящих (#64) и переиспользуется
гибридной диаризацией (#64, часть 2): одна и та же ONNX-модель CAM++ даёт
L2-нормированные эмбеддинги голоса, по которым агломеративная кластеризация
по косинусу отделяет говорящих.

Здесь собрано всё, что не зависит от конкретного потребителя:

* проверка доступности ``sherpa-onnx`` (без импорта — ``find_spec``);
* кэш моделей (``~/.cache/audio-transcriber/sherpa``), атомарная загрузка с
  релизов ``k2-fsa/sherpa-onnx`` и разрешение имени/пути модели;
* :class:`SpeakerEmbedder` — обёртка над ``SpeakerEmbeddingExtractor`` с
  ленивой загрузкой модели (одна модель — много окон);
* :func:`cluster_embeddings` — агломеративная кластеризация по косинусному
  расстоянию (``complete``-linkage по умолчанию — против chaining/перемержа
  кластеров, см. docstring функции), возвращающая метки кластеров.

Любая ошибка не «роняет» вызывающий код: потребитель сам решает, как
деградировать (оценщик возвращает ``None``, гибрид — понятную ошибку).
"""

from __future__ import annotations

import importlib.util
import logging
import os
import tempfile
import urllib.request
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from audio_transcriber.config.defaults import DEFAULT_DIARIZATION_ESTIMATE_MODEL
from audio_transcriber.progress import ProgressCallback, ProgressEvent
from audio_transcriber.utils.audio import SAMPLE_RATE

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "EMBEDDING_MODEL_URL",
    "SpeakerEmbedder",
    "cluster_embeddings",
    "compute_embeddings",
    "count_clusters",
    "default_model_cache_dir",
    "download_file",
    "embedder_available",
    "l2_normalize",
    "resolve_embedding_model",
    "resolve_named_model",
    "sherpa_available",
]

#: Имя ONNX-модели эмбеддингов по умолчанию (та же, что у оценщика говорящих).
DEFAULT_EMBEDDING_MODEL = DEFAULT_DIARIZATION_ESTIMATE_MODEL

#: URL модели эмбеддингов по умолчанию (release ``speaker-recongition-models``
#: — да, с опечаткой в имени релиза у k2-fsa, это исторически так).
EMBEDDING_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    f"speaker-recongition-models/{DEFAULT_EMBEDDING_MODEL}"
)

#: Таймаут загрузки одной модели (секунды).
_DOWNLOAD_TIMEOUT = 120.0


def sherpa_available() -> bool:
    """Установлен ли ``sherpa-onnx`` (без импорта — ``find_spec``)."""
    try:
        return importlib.util.find_spec("sherpa_onnx") is not None
    except (ImportError, ValueError):
        return False


def default_model_cache_dir() -> Path:
    """Каталог кэша моделей (``XDG_CACHE_HOME`` или ``~/.cache``)."""
    raw = os.environ.get("XDG_CACHE_HOME", "").strip()
    base = Path(raw).expanduser() if raw else Path.home() / ".cache"
    return base / "audio-transcriber" / "sherpa"


def is_valid_file(path: Path) -> bool:
    """Непустой существующий файл (мягко: ошибки доступа → ``False``)."""
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _emit(on_progress: ProgressCallback | None, message: str, fraction: float | None) -> None:
    if on_progress is None:
        return
    on_progress(ProgressEvent("diarization", message=message, fraction=fraction))


def download_file(
    url: str,
    target: Path,
    *,
    on_progress: ProgressCallback | None = None,
    label: str = "модели эмбеддингов",
) -> Path | None:
    """Скачивает ``url`` в ``target`` атомарно; ``None`` при любой ошибке.

    Пишет во временный файл рядом с целевым и делает ``replace`` — недокачанный
    файл никогда не выглядит как готовая модель.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=".download-", suffix=".part", dir=target.parent)
    os.close(fd)
    tmp_path = Path(raw_tmp)
    _emit(on_progress, f"Загрузка {label}: {target.name}", 0.0)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "audio-transcriber"})
        with urllib.request.urlopen(request, timeout=_DOWNLOAD_TIMEOUT) as response:
            total_header = response.headers.get("Content-Length")
            total = int(total_header) if total_header and total_header.isdigit() else 0
            received = 0
            with open(tmp_path, "wb") as handle:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    handle.write(chunk)
                    received += len(chunk)
                    if total > 0:
                        _emit(
                            on_progress,
                            f"Загрузка {label}: {target.name}",
                            min(1.0, received / total),
                        )
        if received == 0:
            raise OSError("получен пустой ответ")
        tmp_path.replace(target)
    except Exception as exc:  # noqa: BLE001 — мягкая деградация: модель не критична
        logger.warning(
            "Не удалось скачать %s %s: %s — модель недоступна",
            label,
            url,
            exc,
        )
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            logger.debug("Не удалось удалить временный файл %s", tmp_path)
        return None
    _emit(on_progress, f"Модель готова: {target.name}", 1.0)
    return target


def resolve_named_model(
    model: str,
    cache_dir: Path,
    *,
    default_url: str,
    on_progress: ProgressCallback | None = None,
    download: bool = True,
    label: str = "модели эмбеддингов",
) -> Path | None:
    """Находит модель ``model``: существующий путь или файл в каталоге кэша.

    Для модели по умолчанию недостающий файл скачивается (если ``download``).
    Неизвестное имя без файла на диске не скачивается — вернём ``None``.
    """
    candidate = Path(model).expanduser()
    if is_valid_file(candidate):
        return candidate

    in_cache = cache_dir / model
    if is_valid_file(in_cache):
        return in_cache

    if (
        download
        and not candidate.is_absolute()
        and model == DEFAULT_DIARIZATION_ESTIMATE_MODEL
    ):
        return download_file(default_url, in_cache, on_progress=on_progress, label=label)

    logger.info(
        "Модель эмбеддингов %r не найдена и не будет скачана (ожидается путь "
        "к .onnx или имя модели по умолчанию)",
        model,
    )
    return None


def resolve_embedding_model(
    model: str = DEFAULT_EMBEDDING_MODEL,
    model_dir: Path | None = None,
    *,
    on_progress: ProgressCallback | None = None,
    download: bool = True,
) -> Path | None:
    """Разрешает путь к ONNX-модели эмбеддингов (загрузка из релиза по умолчанию)."""
    cache_dir = model_dir or default_model_cache_dir()
    return resolve_named_model(
        model,
        cache_dir,
        default_url=EMBEDDING_MODEL_URL,
        on_progress=on_progress,
        download=download,
        label="модели эмбеддингов",
    )


def embedder_available(model: str = DEFAULT_EMBEDDING_MODEL, *, model_dir: Path | None = None) -> bool:
    """Доступен ли эмбеддер: есть ``sherpa-onnx`` и модель уже на диске.

    Модель **не** скачивается — проверка доступности не должна иметь побочных
    эффектов. Оценщик числа говорящих при работе сам скачает модель, если надо.
    """
    if not sherpa_available():
        return False
    return resolve_embedding_model(model, model_dir, download=False) is not None


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """L2-нормирует вектор; нулевой вектор возвращается без изменений."""
    result = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(result))
    if norm > 0.0:
        result = result / norm
    return result


class SpeakerEmbedder:
    """Извлекает speaker-эмбеддинги через ``sherpa-onnx`` (модель CAM++).

    Модель загружается лениво и переиспользуется для всех окон: загрузка ONNX
    не бесплатна, а гибридная диаризация вызывает эмбеддер десятки раз.
    """

    def __init__(self, model: Path, *, num_threads: int | None = None) -> None:
        self._model = Path(model)
        self._num_threads = num_threads
        self._extractor: object | None = None

    def _load(self) -> object:
        if self._extractor is not None:
            return self._extractor

        import sherpa_onnx

        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig()
        config.model = str(self._model)
        config.provider = "cpu"
        config.num_threads = (
            self._num_threads
            if self._num_threads is not None
            else max(1, min(4, os.cpu_count() or 1))
        )
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        return self._extractor

    def embed(self, waveform: np.ndarray) -> np.ndarray:
        """L2-нормированный эмбеддинг моно waveform float32."""
        extractor = self._load()
        samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
        stream = extractor.create_stream()  # type: ignore[attr-defined]
        stream.accept_waveform(SAMPLE_RATE, samples)
        stream.input_finished()
        vector = extractor.compute(stream)  # type: ignore[attr-defined]
        return l2_normalize(np.asarray(vector, dtype=np.float32))

    def embed_many(self, windows: Sequence[np.ndarray]) -> np.ndarray:
        """L2-нормированные эмбеддинги для последовательности окон."""
        return np.stack([self.embed(window) for window in windows])


def compute_embeddings(windows: Sequence[np.ndarray], model: Path) -> np.ndarray:
    """L2-нормированные эмбеддинги говорящего для каждого окна.

    Одноразовая обёртка над :class:`SpeakerEmbedder` — совместима с прежним
    поведением оценщика (загрузка модели на каждый вызов функции).
    """
    return SpeakerEmbedder(model).embed_many(windows)


def cluster_embeddings(
    embeddings: np.ndarray,
    *,
    threshold: float,
    n_clusters: int | None = None,
    linkage: str = "complete",
    metric: str = "cosine",
) -> np.ndarray:
    """Агломеративная кластеризация эмбеддингов.

    :param embeddings: матрица ``(N, D)`` (обычно L2-нормированных) векторов.
    :param threshold: порог расстояния (используется, если ``n_clusters``
        не задан).
    :param n_clusters: точное число кластеров — «ориентир»; если задано и
        допустимо (``1..N``), используется вместо порога.
    :param linkage: метод связи. По умолчанию ``"complete"``: он сравнивает
        **максимальное** расстояние между кластерами и потому не выстраивает
        «цепочки» (chaining) из близких соседей — именно chaining
        ``average``-linkage перемерживал кластеры говорящих (кластер, растущий
        через мостики-выбросы, вбирал соседей). Для форсированного
        ``n_clusters`` (явное число говорящих) вызывающий код задаёт
        ``"ward"`` — на реальных эмбеддингах CAM++ он даёт лучшее распределение
        при фиксированном ``k``.
    :param metric: метрика расстояния: ``"cosine"`` (по умолчанию; на
        L2-нормированных векторах расстояние = ``1 - cos``) либо
        ``"euclidean"`` (в паре с ``linkage="ward"``, который определён только
        для евклидовой метрики).
    :return: массив меток кластеров длины ``N``.
    """
    from sklearn.cluster import AgglomerativeClustering

    matrix = np.asarray(embeddings, dtype=np.float32)
    count = int(matrix.shape[0])
    if count <= 1:
        return np.zeros(count, dtype=int)

    # L2-нормировка: для косинуса 1 - <unit_i, unit_j>, а ward/euclidean тоже
    # считается на единичных векторах (так же, как в diag_hybrid и боевом коде).
    unit = matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)
    params: dict[str, object]
    if metric == "cosine":
        data = np.clip(1.0 - unit @ unit.T, 0.0, 2.0)
        np.fill_diagonal(data, 0.0)
        params = {"metric": "precomputed", "linkage": linkage}
    else:  # euclidean (для ward)
        data = unit
        params = {"metric": "euclidean", "linkage": linkage}

    if n_clusters is not None and 1 <= int(n_clusters) < count:
        model = AgglomerativeClustering(n_clusters=int(n_clusters), **params)
    else:
        model = AgglomerativeClustering(
            n_clusters=None, distance_threshold=float(threshold), **params
        )
    return np.asarray(model.fit_predict(data), dtype=int)


def count_clusters(embeddings: np.ndarray, *, threshold: float) -> int:
    """Число кластеров эмбеддингов (агломеративная кластеризация, complete/cosine)."""
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.shape[0] <= 1:
        return max(int(matrix.shape[0]), 1)
    labels = cluster_embeddings(matrix, threshold=threshold)
    return max(1, len(set(labels.tolist())))
