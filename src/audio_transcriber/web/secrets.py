"""Секреты веб-интерфейса (``web-data/secrets.json``).

Секреты — токен Hugging Face (нужен для gated-модели pyannote) и API-ключ
внешней LLM (``LLM_API_KEY``). Они хранятся отдельно от ``settings.json``:
файл принадлежит только владельцу (права ``0600``) и никогда не отдаётся API —
наружу выводится лишь флаг наличия и маска. Каталог ``web-data/`` целиком
исключён из git.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path

logger = logging.getLogger(__name__)

#: Имя поля с токеном в файле секретов.
HF_TOKEN_FIELD = "hf_token"

#: Имя поля с API-ключом внешней LLM в файле секретов.
LLM_API_KEY_FIELD = "llm_api_key"

#: Права файла секретов: чтение/запись только владельцу.
SECRETS_FILE_MODE = 0o600


class SecretsError(ValueError):
    """Не удалось сохранить/прочитать секреты (для ответа 400)."""


def mask_secret(value: str | None) -> str | None:
    """Маска секрета для показа: ``abc…WXYZ`` (или ``…`` для короткого)."""
    if not value:
        return None
    if len(value) <= 8:
        return "…"
    return f"{value[:3]}…{value[-4:]}"


def mask_hf_token(token: str | None) -> str | None:
    """Маска токена для показа: ``hf_…XXXX`` (или ``…`` для короткого)."""
    return mask_secret(token)


class SecretsStore:
    """Чтение/запись ``web-data/secrets.json`` с закрытыми правами доступа."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        """Путь к JSON-файлу секретов."""
        return self._path

    def load(self) -> dict[str, object]:
        """Содержимое файла или пустой словарь, если файла нет/он испорчен."""
        try:
            if not self._path.is_file():
                return {}
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("Не удалось прочитать секреты %s — считаются пустыми", self._path)
            return {}
        if not isinstance(payload, dict):
            return {}
        return {str(key): value for key, value in payload.items()}

    def get_hf_token(self) -> str | None:
        """Сохранённый токен Hugging Face или ``None``."""
        raw = self.load().get(HF_TOKEN_FIELD)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        return None

    def hf_token_set(self) -> bool:
        """Задан ли токен (значение не раскрывается)."""
        return self.get_hf_token() is not None

    def masked_hf_token(self) -> str | None:
        """Маска сохранённого токена для отображения в UI."""
        return mask_hf_token(self.get_hf_token())

    def get_llm_api_key(self) -> str | None:
        """Сохранённый API-ключ внешней LLM или ``None``."""
        raw = self.load().get(LLM_API_KEY_FIELD)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        return None

    def llm_api_key_set(self) -> bool:
        """Задан ли API-ключ внешней LLM (значение не раскрывается)."""
        return self.get_llm_api_key() is not None

    def masked_llm_api_key(self) -> str | None:
        """Маска сохранённого API-ключа для отображения в UI."""
        return mask_secret(self.get_llm_api_key())

    def set_hf_token(self, token: str | None) -> None:
        """Сохраняет токен (пустая строка/``None`` — удаляет) с правами ``0600``.

        Пустой токен удаляет поле из файла. Файл создаётся (или ему
        возвращаются права) строго ``0600``; сам токен не логируется.
        """
        self._set_secret(HF_TOKEN_FIELD, token)

    def set_llm_api_key(self, api_key: str | None) -> None:
        """Сохраняет API-ключ внешней LLM (пустая строка/``None`` — удаляет).

        Ключ хранится отдельно от ``settings.json`` в файле с правами ``0600``
        и никогда не логируется.
        """
        self._set_secret(LLM_API_KEY_FIELD, api_key)

    def _set_secret(self, field: str, value: str | None) -> None:
        payload = self.load()
        clean = value.strip() if isinstance(value, str) else ""
        if clean:
            payload[field] = clean
        else:
            payload.pop(field, None)
        self._write(payload)

    def _write(self, payload: Mapping[str, object]) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = json.dumps(dict(payload), ensure_ascii=False, indent=2)
            # Открываем с режимом сразу, затем подтверждаем chmod: так права
            # 0600 действуют и для уже существующего файла.
            descriptor = os.open(
                self._path,
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                SECRETS_FILE_MODE,
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(data)
            finally:
                os.chmod(self._path, SECRETS_FILE_MODE)
        except OSError as exc:
            raise SecretsError(f"Не удалось сохранить секреты: {exc}") from exc


def effective_hf_token(
    store: SecretsStore, defaults: Mapping[str, str] | None = None
) -> str | None:
    """Токен из секретов, иначе из окружения/``config.env`` (``None``, если нет)."""
    token = store.get_hf_token()
    if token:
        return token
    source = defaults or {}
    raw = source.get("HF_TOKEN") or source.get("HUGGING_FACE_HUB_TOKEN") or ""
    return raw.strip() or None


def effective_llm_api_key(
    store: SecretsStore, defaults: Mapping[str, str] | None = None
) -> str | None:
    """API-ключ внешней LLM из секретов, иначе из окружения/``config.env``."""
    api_key = store.get_llm_api_key()
    if api_key:
        return api_key
    source = defaults or {}
    raw = source.get("LLM_API_KEY") or ""
    return raw.strip() or None
