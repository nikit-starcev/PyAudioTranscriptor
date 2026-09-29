"""Сборка глоссария для матчера из локальной SQLite-БД и текстовых файлов.

Единая точка входа для конвейера: :func:`build_glossary` собирает
:class:`~audio_transcriber.llm.glossary.Glossary` из аккаунтов БД, а при
первом запуске переносит туда старые текстовые глоссарии (совместимость с
``GLOSSARY_PATH``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from audio_transcriber.llm.glossary import Glossary, load_glossary
from audio_transcriber.storage.glossary_db import GlossaryDB

if TYPE_CHECKING:
    from audio_transcriber.config.settings import AppConfig

logger = logging.getLogger(__name__)


def build_glossary(config: AppConfig) -> Glossary | None:
    """Собирает глоссарий для конвейера по конфигурации.

    Порядок:

    1. Если :attr:`AppConfig.glossary_enabled` выключен — ``None``.
    2. Если БД пуста и заданы текстовые ``glossary_path`` — старые файлы
       импортируются в БД (миграция, идемпотентно).
    3. Термины берутся из включённых источников и записей БД
       (:meth:`GlossaryDB.terms_for_matcher`).

    Если БД не дала ни одного термина (например, файлы пусты), для
    совместимости возвращается глоссарий из текстовых путей.
    """
    if not config.glossary_enabled:
        return None

    db_path = config.resolved_glossary_db()
    with GlossaryDB(db_path) as db:
        if config.glossary_path and db.count() == 0:
            reports = db.migrate_from_paths(list(config.glossary_path))
            for report in reports:
                logger.info(
                    "Глоссарий: импортирован источник «%s» (%d записей, пропущено %d)",
                    report.source,
                    report.added,
                    report.skipped,
                )
        terms = db.terms_for_matcher()

    if not terms and config.glossary_path:
        return load_glossary(config.glossary_path)

    if terms:
        logger.info("Глоссарий: загружено терминов — %d (БД %s)", len(terms), db_path)
    return Glossary(terms=terms)
