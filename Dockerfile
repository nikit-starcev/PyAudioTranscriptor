# syntax=docker/dockerfile:1

# =============================================================================
# PyAudioTranscriptor — самодостаточный Linux x86_64 образ с веб-UI и
# нативными движками. GPU-ускорение — через Vulkan (whisper.cpp/llama.cpp),
# поэтому CUDA/ROCm для работы не нужны: достаточно пробросить /dev/dri.
#
# Сборка:
#   docker build -t py-audio-transcriber .
#
# Запуск (данные и образцы голоса — на томах):
#   docker run --rm -p 127.0.0.1:8790:8790 \
#     -v "$PWD/web-data:/data" -v "$PWD/voices:/data/voices" \
#     py-audio-transcriber
#
# Подробности — в README, раздел «Docker».
# =============================================================================

# -----------------------------------------------------------------------------
# Стадия-источник whisper.cpp: официальный Vulkan-образ ggml-org. Из него
# копируются только бинарник и разделяемые библиотеки. Образ зафиксирован по
# digest (тег `main-vulkan` — плавающий); обновить так:
#   docker pull ghcr.io/ggml-org/whisper.cpp:main-vulkan
#   docker inspect --format '{{index .RepoDigests 0}}' \
#     ghcr.io/ggml-org/whisper.cpp:main-vulkan
# -----------------------------------------------------------------------------
FROM ghcr.io/ggml-org/whisper.cpp@sha256:f3090b56b4c6018d8eb7eb8abdea5e5a3d0341a8cb8852b2f2acb96436c44205 AS whisper-cpp

# -----------------------------------------------------------------------------
# llama.cpp (Vulkan) собирается из исходников. Готовый образ
# ghcr.io/ggml-org/llama.cpp:server-vulkan собран на Ubuntu 26.04 и требует
# glibc 2.43 — на Debian trixie (glibc 2.41) его библиотеки не грузятся.
# Сборка в builder'е на том же базовом образе даёт совместимые библиотеки и
# RPATH $ORIGIN (бэкенды грузятся из каталога бинарника).
# -----------------------------------------------------------------------------
FROM python:3.13-slim AS llama-cpp

# Ревизия зафиксирована: это тот же коммит, что и у проверенного образа
# ggml-org/llama.cpp:server-vulkan (0.5.0-dev, build 11223).
ARG LLAMA_CPP_COMMIT=4da6337767f973e2b4d0797e5b323d77d8565e4a

# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         git ca-certificates cmake ninja-build build-essential \
         libvulkan-dev glslc spirv-headers spirv-tools patchelf \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git init -q \
    && git remote add origin https://github.com/ggml-org/llama.cpp.git \
    && git fetch -q --depth 1 origin "${LLAMA_CPP_COMMIT}" \
    && git checkout -q FETCH_HEAD

RUN cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DGGML_VULKAN=ON \
        -DLLAMA_CURL=OFF \
        -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
    && cmake --build build -j"$(nproc)"

# Гарантируем, что и бинарник, и все бэкенды ищут свои .so рядом с собой
# ($ORIGIN), а не по абсолютным путям сборочного каталога.
# hadolint ignore=SC2016
RUN patchelf --set-rpath '$ORIGIN' build/bin/llama-server \
    && find build/bin -maxdepth 1 -name '*.so*' -type f \
         -exec patchelf --set-rpath '$ORIGIN' {} +

# -----------------------------------------------------------------------------
# Builder: разрешение зависимостей и установка проекта в переносимый .venv
# -----------------------------------------------------------------------------
FROM python:3.13-slim AS builder

# uv — официальный статический бинарник (пин версии для воспроизводимости).
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /uvx /bin/

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# git нужен hatch-vcs, чтобы вычислить версию проекта из git-тега при сборке
# (в образ копируется каталог .git; в финальную стадию он не попадает).
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Зависимости. torch/torchaudio переводим на CPU-индекс: на Linux GPU
#    работает через Vulkan (whisper.cpp/llama.cpp), а torch нужен только для
#    pyannote-диаризации. В репозитории те же пакеты привязаны к индексу cu126
#    (для NVIDIA-хостов), поэтому здесь правим URL индекса на cpu и
#    пересобираем lock — иначе `--frozen` притянет CUDA-колёса (несколько ГБ).
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    sed -i 's#https://download.pytorch.org/whl/cu126#https://download.pytorch.org/whl/cpu#' pyproject.toml \
    && SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv lock \
    && uv sync --frozen --no-dev --no-install-project \
         --extra web --extra gigaam --extra sherpa

# 2) Проект. .git — для hatch-vcs; APP_VERSION позволяет переопределить версию
#    при сборке из контекста без git.
COPY src/ ./src/
COPY .git ./.git
ARG APP_VERSION=""
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -n "${APP_VERSION}" ]; then export SETUPTOOLS_SCM_PRETEND_VERSION="${APP_VERSION}"; fi; \
    uv sync --frozen --no-dev --no-editable \
      --extra web --extra gigaam --extra sherpa

# -----------------------------------------------------------------------------
# Runtime: минимальный образ с нативными движками и непривилегированным юзером
# -----------------------------------------------------------------------------
FROM python:3.13-slim AS runtime

LABEL org.opencontainers.image.title="PyAudioTranscriptor" \
      org.opencontainers.image.description="Локальный транскрибер с веб-UI, диаризацией и нативными Vulkan-движками" \
      org.opencontainers.image.source="https://github.com/reconurge/PyAudioTranscriptor" \
      org.opencontainers.image.licenses="MIT"

# Системные зависимости:
#   ffmpeg             — декодирование аудио (и общие libav* для torchcodec)
#   libgomp1           — OpenMP (whisper.cpp, onnxruntime)
#   libsndfile1        — чтение WAV/FLAC рядом с аудиостеком
#   libvulkan1 + mesa-vulkan-drivers — Vulkan-загрузчик и ICD
#                        (RADV для AMD, ANV для Intel, lavapipe — софт-фолбэк)
#   curl               — healthcheck
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         ffmpeg \
         libgomp1 \
         libsndfile1 \
         libvulkan1 \
         mesa-vulkan-drivers \
         curl \
    && rm -rf /var/lib/apt/lists/*

# whisper.cpp (Vulkan): бинарник whisper-cli + разделяемые библиотеки в
# /usr/local/lib (этот каталог уже в поиске динамического загрузчика, поэтому
# достаточно ldconfig ниже).
COPY --from=whisper-cpp /usr/local/bin/whisper-cli /usr/local/bin/whisper-cli
COPY --from=whisper-cpp /usr/local/lib/libggml*.so* /usr/local/lib/
COPY --from=whisper-cpp /usr/local/lib/libwhisper*.so* /usr/local/lib/

# llama.cpp (Vulkan) из собственной сборки: бинарник и ВСЕ бэкенды (.so,
# включая libggml-vulkan.so) лежат вместе в /opt/llama. RPATH=$ORIGIN выставлен
# при сборке, поэтому библиотеки грузятся из каталога бинарника без
# LD_LIBRARY_PATH (это важно: libggml-*.so из whisper.cpp и llama.cpp имеют
# одинаковые SONAME, и глобальный путь заставил бы один движок загрузить
# библиотеку другого). В PATH — имя по умолчанию `llama-server`.
COPY --from=llama-cpp /src/build/bin/llama-server /opt/llama/llama-server
COPY --from=llama-cpp /src/build/bin/*.so* /opt/llama/
RUN ln -s /opt/llama/llama-server /usr/local/bin/llama-server \
    && ldconfig

# deep-filter (DeepFilterNet v0.5.6+, статическая musl-сборка; модель встроена
# в бинарник — веса не скачиваются). Контрольная сумма фиксирует артефакт.
ADD --chmod=755 \
    --checksum=sha256:70775e251eee44c0f2451a1e833326cf8bcbbe304d3e7cd12851e6fce72ef7da \
    https://github.com/Rikorose/DeepFilterNet/releases/download/v0.5.6/deep-filter-0.5.6-x86_64-unknown-linux-musl \
    /usr/local/bin/deep-filter

# Python-окружение из builder (torch CPU + extras web/gigaam/sherpa).
COPY --from=builder /app/.venv /app/.venv

# Готовая конфигурация для контейнера: пример-конфиг без абсолютных путей хоста
# (бинарники берутся из PATH, модели скачиваются в /data/models). Приложение
# читает config.env из рабочего каталога /app. Свой config.env можно смонтировать
# поверх /app/config.env — тогда пути в нём должны быть валидны внутри контейнера.
COPY config.example.env /app/config.env

# Непривилегированный пользователь. Все пользовательские данные — в /data (том):
# загруженные файлы, jobs.db, результаты, кэш, модели. Образцы голоса лежат в
# /data/voices; симлинки из /app сохраняют поведение путей по умолчанию
# (`voices/`, `glossary.db`) даже при монтировании своего config.env.
ARG UID=1000
ARG GID=1000
RUN groupadd -g "${GID}" app \
    && groupadd -g 107 render \
    && useradd -l -u "${UID}" -g "${GID}" -G video,render -m -s /usr/sbin/nologin app \
    && mkdir -p /data/voices \
    && chown -R app:app /data \
    && ln -s /data/voices /app/voices \
    && ln -s /data/glossary.db /app/glossary.db

# Entrypoint стартует как root лишь для того, чтобы выровнять владельца тома
# /data (Docker Desktop отдаёт bind-mount как root:root), после чего сбрасывает
# привилегии до `app` через setpriv. Сам веб-сервер и все движки работают как app.
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod 0755 /usr/local/bin/entrypoint.sh

ENV PATH="/app/.venv/bin:${PATH}" \
    AUDIO_TRANSCRIBER_WEB_DATA="/data" \
    XDG_CACHE_HOME="/data/.cache" \
    HF_HOME="/data/.cache/huggingface" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
EXPOSE 8790
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["curl", "-fsS", "http://127.0.0.1:8790/api/health"]

# Веб-UI слушает 0.0.0.0 внутри контейнера; публикация на хост — через compose.
# ENTRYPOINT запускает app-drop, а CMD можно переопределить (`docker run ... --port`).
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["audio-transcriber", "web", "--no-browser", "--host", "0.0.0.0", "--port", "8790"]
