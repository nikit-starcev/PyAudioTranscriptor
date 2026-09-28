"""LLM-постобработка стенограммы: имена участников и правка терминов.

Связывает :mod:`audio_transcriber.llm.client` (локальная LLM) и
:mod:`audio_transcriber.llm.glossary` (детерминированный матчер) в два
прикладных шага:

- :func:`extract_participants` — просит LLM прочитать стенограмму с метками
  спикеров и вернуть маппинг «Спикер N → Имя»;
- :func:`correct_terms` — правит термины по глоссарию (детерминированно) и,
  опционально, проверяет спорные случаи через LLM.

Оба шага устроены так, чтобы при сбое LLM или отсутствии модели мягко
пропускаться (возврат исходных данных + предупреждение), а не ронять
весь конвейер.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from dataclasses import replace

from audio_transcriber.config.settings import AppConfig
from audio_transcriber.domain.models import Speaker, TranscriptEntry
from audio_transcriber.llm.base import LlmClient
from audio_transcriber.llm.client import DEFAULT_CONTEXT_SIZE, create_llm_client
from audio_transcriber.llm.glossary import Glossary, load_glossary, write_suggested_terms
from audio_transcriber.progress import ProgressCallback, ProgressEvent
from audio_transcriber.utils.text import WORD_PATTERN

logger = logging.getLogger(__name__)

_PARTICIPANTS_SYSTEM_PROMPT = (
    "Ты — ассистент, который разбирает стенограммы телефонных разговоров. "
    "Тебе дана стенограмма с репликами, помеченными говорящими "
    "(«Спикер 1», «Спикер 2», …). Определи имена участников по содержанию "
    "реплик. Присваивай имя говорящему ТОЛЬКО при явном основании в тексте: "
    "прямое представление («я <Имя>», «меня зовут <Имя>», «это <Имя>») или "
    "прямое обращение к говорящему по имени («<Имя>, …»). "
    "Если явного основания нет — НЕ указывай имя для этого говорящего: "
    "лучше не назвать участника, чем придумать имя. "
    "ВАЖНО: каждое имя может принадлежать только ОДНОМУ говорящему — не "
    "присваивай одно и то же имя разным «Спикерам». Имя, которое лишь "
    "упоминается в тексте (например, о нём говорят в третьем лице), но не "
    "принадлежит говорящему, не указывай. Не выдумывай имена, которых нет в "
    "тексте стенограммы. "
    "Верни ТОЛЬКО строгий JSON без пояснений и без текста до/после него."
)

_TERM_CHECK_SYSTEM_PROMPT = (
    "Ты — ассистент, который исправляет опечатки распознавания речи в "
    "стенограмме. Дан список канонических терминов. Найди в тексте слова, "
    "которые являются ASR-опечатками этих терминов (например, «АИБ» вместо "
    "«ОИБ»), и верни ТОЛЬКО строгий JSON. Не меняй ничего, кроме таких "
    "опечаток терминов, и не переписывай смысл."
)


def _unique_speakers(entries: list[TranscriptEntry]) -> list[Speaker]:
    """Возвращает уникальных говорящих в порядке первого появления."""
    seen: dict[str, Speaker] = {}
    for entry in entries:
        if entry.speaker is not None and entry.speaker.id not in seen:
            seen[entry.speaker.id] = entry.speaker
    return list(seen.values())


def _speaker_labels(speakers: list[Speaker]) -> dict[str, str]:
    """Метки говорящих для LLM: ``speaker_id -> «Спикер N»`` (1-indexed)."""
    return {
        speaker.id: f"Спикер {index}" for index, speaker in enumerate(speakers, start=1)
    }


def _transcript_lines(
    entries: list[TranscriptEntry], labels: dict[str, str]
) -> Iterator[str]:
    """Строки стенограммы с метками «Спикер N: реплика» (пустые пропускаются)."""
    for entry in entries:
        if not entry.text.strip():
            continue
        label = labels.get(entry.speaker.id, "?") if entry.speaker else "?"
        yield f"{label}: {entry.text.strip()}"


# Имена-заглушки, которые LLM иногда возвращает вместо реального имени.
# Такие значения не применяются и не попадают в список «Участники».
_PLACEHOLDER_NAMES = frozenset(
    {
        "",
        "?",
        "??",
        "???",
        "-",
        "—",
        "n/a",
        "na",
        "none",
        "null",
        "неизвестно",
        "неизвестный",
        "не определён",
        "не определено",
        "не установлен",
        "не назван",
        "нет имени",
    }
)


def _normalize_name(raw: object) -> str | None:
    """Приводит имя к чистой строке; пустые значения и заглушки → ``None``."""
    name = " ".join(str(raw).split())
    if not name or name.casefold() in _PLACEHOLDER_NAMES:
        return None
    return name


def _transcript_tokens(entries: list[TranscriptEntry]) -> set[str]:
    """Токены (слова) текста стенограммы в нижнем регистре."""
    tokens: set[str] = set()
    for entry in entries:
        for match in WORD_PATTERN.finditer(entry.text):
            tokens.add(match.group(0).casefold())
    return tokens


def _name_part_in_tokens(part: str, tokens: set[str]) -> bool:
    """Есть ли часть имени в тексте: точное совпадение или форма склонения."""
    if part in tokens:
        return True
    # Короткие части не расширяем: иначе «Мак» совпадёт с «максимально».
    if len(part) < 4:
        return False
    return any(token.startswith(part) and len(token) - len(part) <= 3 for token in tokens)


def _name_in_transcript(name: str, tokens: set[str]) -> bool:
    """Встречается ли имя в тексте стенограммы (без учёта регистра).

    Для имени из нескольких слов достаточно, чтобы в тексте присутствовало
    хотя бы одно значимое слово (имя или фамилия).
    """
    parts = [match.group(0).casefold() for match in WORD_PATTERN.finditer(name)]
    if not parts:
        return False
    significant = [part for part in parts if len(part) >= 3] or parts
    return any(_name_part_in_tokens(part, tokens) for part in significant)


def filter_participants_by_transcript(
    names: dict[str, str], entries: list[TranscriptEntry]
) -> dict[str, str]:
    """Оставляет только имена, реально встречающиеся в тексте стенограммы.

    Защита от галлюцинаций LLM: имя, которого нет в тексте, отбрасывается —
    говорящий остаётся под меткой «Спикер N».
    """
    tokens = _transcript_tokens(entries)
    filtered: dict[str, str] = {}
    for speaker_id, name in names.items():
        normalized = _normalize_name(name)
        if normalized is None:
            continue
        if _name_in_transcript(normalized, tokens):
            filtered[speaker_id] = normalized
        else:
            logger.info("Имя «%s» не найдено в тексте стенограммы — не применяю", name)
    return filtered


def build_transcript_for_llm(
    entries: list[TranscriptEntry],
    *,
    speakers: list[Speaker] | None = None,
    limit: int | None = None,
) -> str:
    """Собирает текст стенограммы с метками «Спикер N: реплика».

    Метки назначаются детерминированно по порядку списка ``speakers``
    (1-indexed), поэтому LLM всегда видит одни и те же имена независимо от
    того, как говорящие названы в данных.
    """
    speakers = speakers or _unique_speakers(entries)
    labels = _speaker_labels(speakers)

    transcript = "\n".join(_transcript_lines(entries, labels))
    if limit is not None and len(transcript) > limit:
        transcript = transcript[:limit] + "\n… (стенограмма обрезана)"
    return transcript


def _extract_json_object(raw: str) -> dict:
    """Извлекает первый JSON-объект из ответа LLM (устойчиво к лишнему тексту)."""
    text = raw.strip()
    # Снимаем ```json ... ``` обёртку, если LLM её добавила.
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("В ответе LLM нет JSON-объекта")

    return json.loads(text[start : end + 1])


def parse_participants_json(raw: str, labels_by_id: dict[str, str]) -> dict[str, str]:
    """Разбирает JSON с участниками и возвращает ``speaker_id -> имя``.

    ``labels_by_id`` — отображение реального ``speaker_id`` (например,
    ``SPEAKER_00``) на метку, которую видел LLM (``Спикер 1``). Ответ вида
    ``{"participants": [{"speaker": "Спикер 1", "name": "Максим"}]}``
    приводится к ключам ``speaker_id``. При неудаче возвращает пустой словарь.
    """
    try:
        payload = _extract_json_object(raw)
    except ValueError:
        return {}

    participants = payload.get("participants", [])
    if not isinstance(participants, list):
        return {}

    id_by_label = {label: speaker_id for speaker_id, label in labels_by_id.items()}
    result: dict[str, str] = {}
    for item in participants:
        if not isinstance(item, dict):
            continue
        label = item.get("speaker") or item.get("label") or item.get("id")
        raw_name = item.get("name") or item.get("display_name")
        if not label or not raw_name:
            continue
        name = _normalize_name(raw_name)
        if name is None:
            # Пустое имя или заглушка («?», «неизвестно») — не создаём фантома.
            logger.debug("Имя: заглушка или пустое значение — пропускаю")
            continue
        speaker_id = id_by_label.get(str(label))
        if speaker_id is None:
            # LLM вернул говорящего, которого нет в разметке (например, имя
            # из текста или «?») — пропускаем, чтобы не создавать фантомов.
            logger.debug("Имена: неизвестный говорящий '%s' — пропускаю", label)
            continue
        result[speaker_id] = name
    return result


def _iter_transcript_chunks(
    entries: list[TranscriptEntry], labels: dict[str, str], *, max_chars: int
) -> list[str]:
    """Режет стенограмму на фрагменты по ~``max_chars`` символов.

    Нужно, потому что длинная стенограмма превышает контекст LLM — тогда
    имена извлекаются по фрагментам и объединяются.
    """
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in _transcript_lines(entries, labels):
        if current and size + len(line) > max_chars:
            chunks.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


# Запас под контекст LLM: на 4096 токенов безопасно ~6000 символов русского
# текста вместе с системным промптом, инструкцией и ответом модели. Коэффициент
# привязывает размер чанка к фактическому ``llm_context_size``, а не к
# магическому числу «под 4096».
_CHUNK_CHARS_PER_CONTEXT_TOKEN = 6000 / 4096
# Нижняя граница: крошечный контекст не должен давать неработоспособный чанк.
_MIN_CHUNK_CHARS = 1000


def chunk_chars_for_context(context_size: int) -> int:
    """Размер фрагмента стенограммы под контекст LLM в токенах.

    Пропорционален ``context_size`` с запасом на промпт и ответ модели.
    """
    return max(_MIN_CHUNK_CHARS, int(context_size * _CHUNK_CHARS_PER_CONTEXT_TOKEN))


# Значение по умолчанию — под стандартный контекст 4096 токенов.
_PARTICIPANT_CHUNK_CHARS = chunk_chars_for_context(DEFAULT_CONTEXT_SIZE)


def extract_participants(
    entries: list[TranscriptEntry],
    *,
    llm: LlmClient,
    speakers: list[Speaker] | None = None,
    max_chunk_chars: int | None = None,
) -> dict[str, str]:
    """Извлекает имена участников через LLM.

    Длинная стенограмма режется на фрагменты (иначе промпт превышает контекст
    модели и запрос падает с HTTP 400). Размер фрагмента задаётся
    ``max_chunk_chars`` — вызывающий передаёт лимит под фактический контекст
    LLM. Возвращает отображение ``speaker_id -> имя``. При сбое LLM или разбора
    возвращает пустой словарь и пишет предупреждение — этап не должен
    останавливать конвейер.
    """
    speakers = speakers or _unique_speakers(entries)
    if not speakers:
        return {}

    max_chars = max_chunk_chars if max_chunk_chars is not None else _PARTICIPANT_CHUNK_CHARS
    labels = _speaker_labels(speakers)
    chunks = _iter_transcript_chunks(entries, labels, max_chars=max_chars)

    names: dict[str, str] = {}
    for chunk in chunks:
        user_prompt = (
            "Стенограмма (фрагмент):\n"
            f"{chunk}\n\n"
            "Определи имена участников и верни строгий JSON:\n"
            '{"participants": [{"speaker": "Спикер 1", "name": "Максим"}, ...]}\n'
            "Указывай имя только при явном основании в тексте: прямое "
            "представление («я <Имя>», «меня зовут <Имя>») или обращение к "
            "говорящему по имени. Не выдумывай имена, которых нет в тексте. "
            "Каждое имя может принадлежать только одному говорящему. "
            "Если имя определить не удалось, не включай такого участника в список."
        )
        try:
            raw = llm.chat(
                [
                    {"role": "system", "content": _PARTICIPANTS_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            )
        except Exception as exc:  # noqa: BLE001 — мягкий пропуск фрагмента
            logger.warning("Извлечение имён (фрагмент) не удалось: %s", exc)
            continue

        try:
            found = parse_participants_json(raw, labels)
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("Не удалось разобрать ответ LLM с именами: %s", exc)
            continue

        for speaker_id, name in found.items():
            if speaker_id in names:
                continue
            if name in names.values():
                # одно имя не может принадлежать нескольким говорящим
                logger.debug("Имя «%s» уже присвоено — пропускаю", name)
                continue
            names[speaker_id] = name

    # Пост-фильтр от галлюцинаций: имя должно реально присутствовать в тексте.
    names = filter_participants_by_transcript(names, entries)

    if names:
        logger.info("LLM: определены имена участников — %s", names)
    return names


def apply_participant_names(
    entries: list[TranscriptEntry],
    speakers: list[Speaker],
    names: dict[str, str],
) -> tuple[list[TranscriptEntry], list[Speaker]]:
    """Подставляет имена в ``display_name`` говорящих («Спикер 1 — Максим»).

    Возвращает копии списков с обновлёнными говорящими; реплики ссылаются на
    те же объекты ``Speaker``.
    """
    if not names:
        return entries, speakers

    renamed: dict[str, Speaker] = {}
    for speaker in speakers:
        name = _normalize_name(names.get(speaker.id, ""))
        if name is None or name in speaker.display_name:
            # Пустое/неопределённое имя не добавляем — говорящий остаётся
            # под своей меткой («Спикер N»), фантомных имён не создаём.
            renamed[speaker.id] = speaker
            continue
        renamed[speaker.id] = replace(speaker, display_name=f"{speaker.display_name} — {name}")

    new_speakers = [renamed[speaker.id] for speaker in speakers]
    new_entries = [
        replace(entry, speaker=renamed[entry.speaker.id])
        if entry.speaker is not None and entry.speaker.id in renamed
        else entry
        for entry in entries
    ]
    return new_entries, new_speakers


def correct_terms(
    entries: list[TranscriptEntry],
    glossary: Glossary | None,
    *,
    llm: LlmClient | None = None,
    max_chunk_chars: int | None = None,
) -> list[TranscriptEntry]:
    """Правит термины стенограммы по глоссарию.

    Сначала применяется детерминированный матчер (всегда, если есть
    глоссарий). Если передан ``llm``, дополнительно запускается проверка
    спорных случаев: LLM находит опечатки терминов, но применить разрешено
    только замену, которая сама является термином глоссария (защита от
    переписывания смысла). ``max_chunk_chars`` ограничивает размер фрагмента,
    отправляемого в LLM.
    """
    if glossary is None and llm is None:
        return entries

    corrected: list[TranscriptEntry] = []
    if glossary is not None:
        for entry in entries:
            new_text, replacements = glossary.correct_text(entry.text)
            for old, new in replacements:
                logger.info("Глоссарий: «%s» → «%s»", old, new)
            corrected.append(replace(entry, text=new_text) if replacements else entry)
    else:
        corrected = list(entries)

    if llm is not None and glossary is not None and glossary.terms:
        corrected = _verify_terms_with_llm(
            corrected, glossary, llm=llm, max_chunk_chars=max_chunk_chars
        )

    return corrected


def _verify_terms_with_llm(
    entries: list[TranscriptEntry],
    glossary: Glossary,
    *,
    llm: LlmClient,
    max_chunk_chars: int | None = None,
) -> list[TranscriptEntry]:
    """Опциональная LLM-проверка спорных опечаток терминов (с защитой).

    Стенограмма режется на фрагменты тем же механизмом, что и извлечение имён:
    длинный текст иначе превышает контекст модели и запрос падает с HTTP 400.
    Ответы всех фрагментов объединяются, затем применяются разрешённые замены.
    """
    max_chars = max_chunk_chars if max_chunk_chars is not None else _PARTICIPANT_CHUNK_CHARS
    speakers = _unique_speakers(entries)
    labels = _speaker_labels(speakers)
    chunks = _iter_transcript_chunks(entries, labels, max_chars=max_chars)
    terms = "\n".join(glossary.terms)

    corrections: list[object] = []
    for chunk in chunks:
        user_prompt = (
            f"Термины:\n{terms}\n\n"
            f"Стенограмма (фрагмент):\n{chunk}\n\n"
            "Найди только опечатки перечисленных терминов и верни строгий JSON:\n"
            '{"corrections": [{"before": "АИБ", "after": "ОИБ"}]}\n'
            'Если опечаток нет, верни {"corrections": []}.'
        )
        try:
            raw = llm.chat(
                [
                    {"role": "system", "content": _TERM_CHECK_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            )
            payload = _extract_json_object(raw)
        except Exception as exc:  # noqa: BLE001 — мягко пропускаем фрагмент
            logger.warning("LLM-проверка терминов (фрагмент) не удалась: %s", exc)
            continue
        chunk_corrections = payload.get("corrections", [])
        if not isinstance(chunk_corrections, list):
            continue
        corrections.extend(chunk_corrections)

    if not corrections:
        return entries

    valid_terms = {term.casefold() for term in glossary.terms}
    # Ключи — в нижнем регистре (casefold): поиск в тексте регистронезависимый,
    # поэтому опечатка «аиб» из ответа LLM должна заменить «АИБ» в стенограмме.
    mapping: dict[str, str] = {}
    for item in corrections:
        if not isinstance(item, dict):
            continue
        before = str(item.get("before", "")).strip()
        after = str(item.get("after", "")).strip()
        if not before or not after:
            continue
        if after.casefold() not in valid_terms:
            continue  # защита: заменяем только на канонический термин
        mapping[before.casefold()] = after

    if not mapping:
        return entries

    def _apply(match: re.Match[str]) -> str:
        return mapping.get(match.group(0).casefold(), match.group(0))

    result: list[TranscriptEntry] = []
    # Длинные ключи раньше коротких — иначе «АИБ» перехватит «АИБС».
    pattern = re.compile(
        "|".join(re.escape(before) for before in sorted(mapping, key=len, reverse=True)),
        flags=re.IGNORECASE,
    )
    for entry in entries:
        new_text = pattern.sub(_apply, entry.text)
        if new_text != entry.text:
            logger.info("LLM-правка термина: %s", mapping)
            result.append(replace(entry, text=new_text))
        else:
            result.append(entry)
    return result


def run_llm_postprocess(
    config: AppConfig,
    entries: list[TranscriptEntry],
    speakers: list[Speaker],
    *,
    client: LlmClient | None = None,
    on_progress: ProgressCallback | None = None,
) -> tuple[list[TranscriptEntry], list[Speaker], list[str] | None]:
    """Полный этап LLM-постобработки для конвейера.

    Создаёт LLM-клиент из конфигурации (если он не передан), извлекает имена,
    правит термины по глоссарию и подставляет имена в вывод. Возвращает
    ``(entries, speakers, participants)``, где ``participants`` — список
    участников для шапки документа (``None``, если имена не определены).

    Если модель/бинарник не заданы или недоступны — возвращает исходные
    данные и пишет предупреждение.
    """
    emit = on_progress or (lambda _event: None)
    glossary = load_glossary(config.glossary_path)

    def _maybe_write_suggestions(current: list[TranscriptEntry]) -> None:
        """Собирает кандидатов в термины по финальному тексту (без LLM)."""
        if not (config.llm_suggest_terms and glossary is not None):
            return
        try:
            text = "\n".join(entry.text for entry in current if entry.text.strip())
            path = write_suggested_terms(glossary, text, source=config.input_file.name)
        except Exception as exc:  # noqa: BLE001 — не роняем конвейер
            logger.warning("Сбор предложений терминов не удался: %s", exc)
            return
        if path is not None:
            logger.info("Предложенные термины записаны в файл: %s", path)
            emit(ProgressEvent("llm", f"Предложения новых терминов: {path}", fraction=None))

    if client is None:
        client = create_llm_client(
            model_path=config.llm_model,
            binary=config.llm_binary,
            library_path=config.llm_lib_path,
            gpu=config.llm_gpu,
            context_size=config.llm_context_size,
        )
        # Клиент создан здесь — этот вызов владеет им и обязан закрыть,
        # иначе llama-server останется висеть и держать VRAM.
        owns_client = True
    else:
        # Клиент передан снаружи — владение у вызывающего, не закрываем.
        owns_client = False

    if client is None:
        logger.warning("LLM-постобработка пропущена: не указана модель (LLM_MODEL/--llm-model)")
        _maybe_write_suggestions(entries)
        return entries, speakers, None

    chunk_chars = chunk_chars_for_context(config.llm_context_size)
    try:
        emit(ProgressEvent("llm", "LLM-постобработка", fraction=None))

        names: dict[str, str] = {}
        if config.llm_extract_names:
            try:
                names = extract_participants(
                    entries,
                    llm=client,
                    speakers=speakers,
                    max_chunk_chars=chunk_chars,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Извлечение имён пропущено: %s", exc)
        else:
            logger.info("LLM: извлечение имён отключено (LLM_EXTRACT_NAMES=false)")

        try:
            entries = correct_terms(entries, glossary, llm=client, max_chunk_chars=chunk_chars)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Правка терминов пропущена: %s", exc)

        if names:
            entries, speakers = apply_participant_names(entries, speakers, names)

        _maybe_write_suggestions(entries)

        # В шапку попадают только реально переименованные говорящие: без имени
        # остаётся метка «Спикер N», ей в списке участников не место.
        participants = [speaker.display_name for speaker in speakers if speaker.id in names]
        return entries, speakers, participants or None
    finally:
        if owns_client:
            # Гарантированно снимаем llama-server в том числе на ветках ошибок.
            client.close()
