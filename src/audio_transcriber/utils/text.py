"""Общие текстовые хелперы для правки терминов и опечаток ASR.

Здесь собраны примитивы, которые нужны сразу нескольким подсистемам
(морфологический корректор и глоссарий): выделение слов, перенос регистра,
метрики сходства строк. Единое место исключает расхождение поведения между
ними.
"""

from __future__ import annotations

import re
from dataclasses import replace
from difflib import SequenceMatcher

from audio_transcriber.domain.models import WordTimestamp

# Выделяет «слова» без цифр и знаков препинания (Unicode-буквы).
WORD_PATTERN = re.compile(r"[^\W\d_]+", re.UNICODE)

# Буквы и цифры (без подчёркивания) — ключ сопоставления токенов при
# синхронизации текста и пословных меток: пунктуация в ключ не входит.
_WORD_KEY_RE = re.compile(r"[^\W_]+", re.UNICODE)

#: Символы, недопустимые в именах файлов (Windows/POSIX), и управляющие коды.
_FILENAME_FORBIDDEN = frozenset('/\\:*?"<>|')


def sanitize_filename(name: str, *, fallback: str = "speaker") -> str:
    """Преобразует произвольный текст в безопасное имя файла (без расширения).

    Запрещённые символы (``/ \\ : * ? " < > |``) и управляющие символы
    заменяются на ``_``; пробелы и точки по краям срезаются (иначе Windows не
    создаст файл). Пустой результат заменяется на ``fallback``. Кириллица и
    прочие буквы сохраняются — это имя говорящего, а не технический
    идентификатор.
    """
    cleaned = "".join(
        "_" if char in _FILENAME_FORBIDDEN or ord(char) < 32 else char for char in name
    )
    cleaned = cleaned.strip().strip(". ")
    return cleaned or fallback


def _token_key(token: str) -> str:
    """Ключ токена для сопоставления: буквы/цифры в нижнем регистре.

    Пунктуация и регистр игнорируются, поэтому нормализация
    («привет!!!» → «привет!») не рвёт соответствие слов. Токен без букв/цифр
    (чистая пунктуация) сопоставляется буквально — иначе все знаки препинания
    считались бы «равными» друг другу.
    """
    key = "".join(_WORD_KEY_RE.findall(token)).casefold()
    return key or token.casefold()


def sync_words_to_text(
    old_text: str, words: list[WordTimestamp], new_text: str
) -> list[WordTimestamp]:
    """Согласует пословные метки с изменённым текстом реплики (#84).

    Единая политика: либо точно пересобрать ``words`` под новый текст, либо
    **инвалидировать** их (пустой список), если соответствие установить нельзя.
    Пустые ``words`` (стадия выключена/движок не поддержал) остаются пустыми.

    Текст разбивается на токены по пробелам. Пословные метки сопоставляются
    токенам один-к-одному, если их число совпадает; иначе соответствие не
    установить — возвращается ``[]``. Сопоставление old↔new выполняется по
    ключам без пунктуации (:func:`_token_key`), поэтому удаление маркера
    («привет [СМЕХ] мир» → «привет мир») выбрасывает только слова удалённого
    фрагмента, а нормализация пунктуации слова сохраняет. Если хотя бы один
    токен нового текста не нашёл пары в старом (например, слово переписано
    LLM/корректором), точное соответствие невозможно — ``words``
    инвалидируются, чтобы экспорт не показывал токены удалённого текста.
    """
    if not words:
        return []
    if new_text == old_text:
        return list(words)

    old_tokens = old_text.split()
    if len(words) != len(old_tokens):
        return []

    new_tokens = new_text.split()
    matcher = SequenceMatcher(
        None,
        [_token_key(token) for token in old_tokens],
        [_token_key(token) for token in new_tokens],
        autojunk=False,
    )
    kept: list[WordTimestamp] = []
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            # Текст слова берём из нового текста: пунктуация могла измениться.
            kept.append(
                replace(words[block.a + offset], text=new_tokens[block.b + offset])
            )

    if len(kept) != len(new_tokens):
        # Не все токены нового текста сопоставлены — полагаться на метки нельзя.
        return []
    return kept


def match_case(original: str, replacement: str, *, acronym: bool = False) -> str:
    """Сохраняет регистр исходного слова у замены.

    Для канонических аббревиатур (``acronym=True``) замена всегда пишется
    ПРОПИСНЫМИ, если исходное слово начиналось с заглавной буквы: падежные
    формы вроде «АРМе» или «СУБДе» — это всё та же аббревиатура, а не
    отдельное слово, поэтому «Арм»/«Субд» были бы ошибкой. Полностью
    строчные формы («арме») сохраняют нижний регистр, как и обычные слова.
    """
    if acronym and original[:1].isupper():
        return replacement.upper()
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def ratio(left: str, right: str) -> float:
    """Доля сходства двух строк без учёта регистра (0..1).

    Основано на :class:`difflib.SequenceMatcher` и не зависит от порядка
    общих фрагментов.
    """
    return SequenceMatcher(None, left.casefold(), right.casefold()).ratio()


def levenshtein(left: str, right: str) -> int:
    """Расстояние Левенштейна между двумя строками."""
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def levenshtein_ratio(left: str, right: str) -> float:
    """Сходство двух строк по расстоянию Левенштейна (0..1).

    Регистр не меняется: вызывающий сам решает, нормализовать ли строки
    (в глоссарии обе строки уже приведены к нижнему регистру).
    """
    if not left and not right:
        return 1.0
    max_len = max(len(left), len(right))
    return 1.0 - levenshtein(left, right) / max_len
