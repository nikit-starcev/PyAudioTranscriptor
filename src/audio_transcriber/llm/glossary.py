"""Пользовательский глоссарий терминов и детерминированный матчер.

Глоссарий — простой текстовый файл: по одному термину в строке, строки,
начинающиеся с ``#``, и пустые строки игнорируются. Термин может быть как
одним словом (``ОИБ``, ``транскрибер``), так и фразой из нескольких слов.

Кроме списка терминов, файл поддерживает **явные пары** вида
``АИБ = ОИБ``: слева — ошибочная форма, которую слышит ASR, справа — канон.
Такие пары применяются всегда и приоритетнее нечёткого сопоставления — это
единственный безопасный способ гарантировать нужную замену.

Можно загрузить несколько файлов глоссария сразу (см. :func:`load_glossary`);
термины и пары объединяются.

Матчер решает две задачи без обращения к LLM:

1. **Правка терминов к канону** — сначала применяет явные пары, затем ищет в
   тексте термины из глоссария с учётом регистра и склонений и исправляет
   ASR-опечатки терминов (например, «АИБ» → «ОИБ»). Точное совпадение с
   термином никогда не меняется, а короткие аббревиатуры правятся только
   при единственном кандидате — иначе слово не трогается.
2. **Авто-предложение новых терминов** — находит потенциальные термины
   (аббревиатуры ПРОПИСНЫМИ и незнакомые существительные), которых нет в
   глоссарии, и возвращает их список для ручного добавления.

Всё детерминировано и не зависит от внешних сервисов, поэтому легко
покрывается юнит-тестами.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from audio_transcriber.utils.glossary_paths import (
    normalize_glossary_paths,
)
from audio_transcriber.utils.glossary_paths import (
    split_glossary_paths as split_glossary_paths,
)
from audio_transcriber.utils.text import (
    WORD_PATTERN,
    levenshtein,
    levenshtein_ratio,
    match_case,
)

logger = logging.getLogger(__name__)

# Имя файла с предложенными терминами (рядом с глоссарием):
# «glossary.txt» -> «glossary.suggested.txt».
SUGGESTED_TERMS_SUFFIX = ".suggested.txt"

# Длина слова, с которой считается, что у термина есть «корень», позволяющий
# отличать склонение («транскрибера» от «транскрибер») от опечатки.
_DECLENSION_MIN_STEM = 5
_DECLENSION_MAX_SUFFIX = 3

# Аббревиатуры короче этого не правим автоматически (слишком рискованно).
_ACRONYM_MIN_LEN = 3

# Короткие аббревиатуры (ОИБ, АИС, АРМ …) правим только при единственном
# кандидате в глоссарии: при неоднозначности слово не трогаем, иначе, например,
# «ОИБ» тянется к «ОИВ», а «АИБ» — к «АИС».
_SHORT_ACRONYM_MAX_LEN = 5

# Разделитель явной пары «как слышит ASR = канон» внутри строки глоссария.
_PAIR_SEPARATOR = "="

DEFAULT_MIN_WORD_LENGTH = 2
DEFAULT_MAX_EDIT_DISTANCE = 2
DEFAULT_SIMILARITY_THRESHOLD = 0.8


def _is_acronym(word: str) -> bool:
    """Считает слово аббревиатурой: целиком ПРОПИСНЫМИ и короткое."""
    return word.isupper() and 2 <= len(word) <= 8 and word.isalpha()


def _normalize_spaces(value: str) -> str:
    """Схлопывает любые пробельные последовательности к одному пробелу."""
    return " ".join(value.split())


class Glossary:
    """Глоссарий терминов с детерминированным матчером.

    :param terms: начальный список терминов (строки), либо ``None``. Строка
        может быть явной парой ``"АИБ = ОИБ"``.
    :param path: путь к одному файлу глоссария (одна строка — один термин
        либо пара).
    :param paths: последовательность путей к файлам глоссария; термины и пары
        объединяются. Можно указывать вместе с ``path``.
    """

    def __init__(
        self,
        terms: list[str] | None = None,
        *,
        path: Path | str | None = None,
        paths: Iterable[Path | str] | None = None,
        min_word_length: int = DEFAULT_MIN_WORD_LENGTH,
        max_edit_distance: int = DEFAULT_MAX_EDIT_DISTANCE,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> None:
        self._min_word_length = min_word_length
        self._max_edit_distance = max_edit_distance
        self._similarity_threshold = similarity_threshold

        self._single_terms: list[str] = []  # термины из одного слова (lower)
        self._phrase_terms: list[str] = []  # термины-фразы (lower)
        self._acronym_terms: set[str] = set()  # аббревиатуры (lower)
        self._terms_lower: set[str] = set()
        self._aliases: dict[str, str] = {}  # ошибочная форма (lower) → канон
        self._morph: Any = None  # ленивый импорт pymorphy3
        self._known_cache: dict[str, bool] = {}  # кэш «слово есть в словаре»
        self._paths: list[Path] = []

        if path is not None:
            self._load_file(Path(path))
        if paths is not None:
            for item in paths:
                self._load_file(Path(item))
        if terms:
            self.add(terms)

        self._source_path: Path | None = self._paths[0] if self._paths else None

    # ------------------------------------------------------------------
    # Загрузка
    # ------------------------------------------------------------------
    def _load_file(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Файл глоссария не найден: {path}")
        loaded: list[str] = []
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            loaded.append(line)
        self.add(loaded)
        self._paths.append(path)
        logger.info("Глоссарий: загружено строк — %d (из %s)", len(loaded), path)

    def add(self, terms: list[str]) -> None:
        """Добавляет термины и явные пары в глоссарий, дедуплицируя по регистру."""
        for raw in terms:
            term = raw.strip()
            if not term:
                continue
            canonical = self._register_pair(term)
            if canonical is None:
                continue
            self._register_term(canonical)

    def _register_pair(self, entry: str) -> str | None:
        """Регистрирует явную пару ``ошибочная форма = канон``.

        Возвращает канон, который нужно добавить как обычный термин, либо сам
        ``entry``, если пара не задана. При некорректной паре пишет
        предупреждение и возвращает ``None``.
        """
        if _PAIR_SEPARATOR not in entry:
            return entry
        wrong, _, canonical = entry.partition(_PAIR_SEPARATOR)
        wrong, canonical = wrong.strip(), canonical.strip()
        if not wrong or not canonical:
            logger.warning("Глоссарий: некорректная пара, строка пропущена: %r", entry)
            return None
        # Ключ-опечатку нормализуем: регистр не важен, а фразы — к одному
        # пробелу между словами (иначе фразовая пара не найдётся в тексте).
        self._aliases[_normalize_spaces(wrong.casefold())] = canonical
        return canonical

    def _register_term(self, term: str) -> None:
        # Регистр не важен; внутренние пробелы фразы схлопываем к одному,
        # чтобы «информационная  безопасность» и «информационная безопасность»
        # считались одним термином.
        folded = _normalize_spaces(term.casefold())
        if folded in self._terms_lower:
            return
        self._terms_lower.add(folded)
        if " " in folded:
            self._phrase_terms.append(folded)
        else:
            self._single_terms.append(folded)
            if _is_acronym(term):
                self._acronym_terms.add(folded)

    @property
    def terms(self) -> list[str]:
        """Все термины глоссария (в нижнем регистре, порядок добавления)."""
        return self._single_terms + self._phrase_terms

    @property
    def source_path(self) -> Path | None:
        """Путь к файлу, из которого загружен глоссарий (если известен)."""
        return self._source_path

    @property
    def suggested_path(self) -> Path | None:
        """Путь к файлу предложений новых терминов рядом с глоссарием."""
        if self._source_path is None:
            return None
        return self._source_path.with_name(self._source_path.stem + SUGGESTED_TERMS_SUFFIX)

    def __len__(self) -> int:
        return len(self._terms_lower)

    def __contains__(self, term: str) -> bool:
        return term.casefold() in self._terms_lower

    # ------------------------------------------------------------------
    # Правка
    # ------------------------------------------------------------------
    def _best_term_match(self, word: str) -> str | None:
        """Возвращает канонический термин для опечатки ``word``, либо ``None``.

        Правила:

        * точное совпадение (без учёта регистра) с термином не меняется;
        * аббревиатуры — не более одной правки символа;
        * короткие аббревиатуры правятся только при единственном лучшем
          кандидате: при ничьей слово не трогается (``DEBUG`` в лог);
        * длинные слова/термины — прежняя fuzzy-логика (лучшее сходство).
        """
        folded = word.casefold()
        if folded in self._terms_lower:
            return None  # слово уже канонично — правка не нужна

        # Чисто латинские токены (DNS, BGP) не правим нечётко: одна буква
        # отличает разные аббревиатуры (DNS ≠ DRS), а транслит-опечатки ASR
        # всё равно приходят кириллицей и обрабатываются ниже.
        if folded.isascii():
            return None

        candidates: list[tuple[str, int, float]] = []  # (термин, дистанция, сходство)

        for term in self._single_terms:
            if len(term) < self._min_word_length:
                continue

            # Склонение: слово длиннее термина на короткий «падежный» суффикс.
            if (
                term not in self._acronym_terms
                and len(term) >= _DECLENSION_MIN_STEM
                and folded.startswith(term)
                and 0 < len(folded) - len(term) <= _DECLENSION_MAX_SUFFIX
            ):
                return None  # это форма того же термина, а не опечатка

            distance = levenshtein(folded, term)

            if term in self._acronym_terms and len(term) >= _ACRONYM_MIN_LEN:
                # Аббревиатуры: достаточно одной правки символа (АИБ → ОИБ),
                # но только если слово не является обычным словом русского
                # языка — иначе «что»/«про»/«он» тянулись бы к «ЧТЗ»/«ППО»/«АОН».
                if distance != 1:
                    continue
                if not self._is_safe_acronym_candidate(folded, term):
                    continue
            else:
                if distance > self._max_edit_distance:
                    continue
                if levenshtein_ratio(folded, term) < self._similarity_threshold:
                    continue
                # Не правим обычные слова русского языка: «разработки» — это
                # не опечатка «разработчик». Мягкая правка только для слов,
                # которых нет в морфологическом словаре (настоящие ASR-опечатки).
                if self._is_known_word(folded):
                    continue

            candidates.append((term, distance, levenshtein_ratio(folded, term)))

        if not candidates:
            return None

        best = max(candidates, key=lambda item: (item[2], -item[1]))
        ties = [item for item in candidates if item[2] == best[2] and item[1] == best[1]]

        if len(ties) > 1 and self._is_ambiguous_acronym(word, ties):
            logger.debug(
                "Глоссарий: «%s» не исправлено — неоднозначные кандидаты: %s",
                word,
                ", ".join(item[0] for item in ties),
            )
            return None

        return best[0]

    def _is_safe_acronym_candidate(self, folded: str, term: str) -> bool:
        """Можно ли мягко исправить слово на аббревиатуру ``term``.

        Правка аббревиатуры безопасна, если слово — падежная форма самого
        термина («АРМе» → «АРМ») либо не является обычным словом русского
        языка (вероятная ASR-опечатка вроде «аиб»). Обычные слова («что»,
        «про», «он») не трогаем, даже если они на одну правку ближе к
        какой-нибудь аббревиатуре глоссария.
        """
        if folded.startswith(term) and 0 < len(folded) - len(term) <= _DECLENSION_MAX_SUFFIX:
            return True
        return not self._is_known_word(folded)

    def _is_ambiguous_acronym(self, word: str, ties: Sequence[tuple[str, int, float]]) -> bool:
        """Нужно ли трактовать ничью кандидатов как неоднозначную аббревиатуру.

        Защищаем только короткие аббревиатуры (само слово ПРОПИСНЫМИ и
        короткое, либо все кандидаты — короткие аббревиатуры): именно они
        чаще всего «тянутся» к чужому термину. Для длинных слов сохраняется
        прежнее поведение матчера.
        """
        if word.isupper() and 2 <= len(word) <= _SHORT_ACRONYM_MAX_LEN:
            return True
        return all(
            term in self._acronym_terms and len(term) <= _SHORT_ACRONYM_MAX_LEN
            for term, _, _ in ties
        )

    # ------------------------------------------------------------------
    # Правка фразовых терминов
    # ------------------------------------------------------------------
    def _best_phrase_match(
        self, folded_words: Sequence[str], candidates: Sequence[str]
    ) -> tuple[str, bool] | None:
        """Лучший фразовый термин для окна из ``folded_words``.

        Возвращает ``(канон, изменился)`` либо ``None``. Канон — термин в
        нижнем регистре (как хранится в глоссарии); ``изменился=False``
        означает точное совпадение — фраза уже канонична, менять не нужно.

        «Лёгкая опечатка» допускается, только если число слов совпадает,
        каждое отличающееся слово близко к канону (сходство не ниже порога),
        а суммарное расстояние Левенштейна по всем словам не превышает
        ``max_edit_distance``. Так фраза из нескольких слов исправляется
        точечно и не задевает другие формулировки.
        """
        window = " ".join(folded_words)
        best: tuple[str, bool] | None = None
        best_score = -1.0

        for phrase in candidates:
            phrase_words = phrase.split()
            if len(phrase_words) != len(folded_words):
                continue
            if window == phrase:
                return phrase, False  # точное совпадение — не меняем

            total_edits = 0
            matched = True
            for word, canonical in zip(folded_words, phrase_words, strict=True):
                if word == canonical:
                    continue
                distance = levenshtein(word, canonical)
                if (
                    distance > self._max_edit_distance
                    or levenshtein_ratio(word, canonical) < self._similarity_threshold
                ):
                    matched = False
                    break
                total_edits += distance

            if not matched or total_edits == 0 or total_edits > self._max_edit_distance:
                continue

            score = levenshtein_ratio(window, phrase)
            if score > best_score:
                best_score = score
                best = (phrase, True)

        return best

    @staticmethod
    def _apply_phrase_case(original_words: Sequence[str], canonical_words: Sequence[str]) -> str:
        """Переносит регистр исходной фразы на канон по словам.

        Сохранение регистра как у одиночных терминов: «АИБ безопасность» →
        «ОИБ безопасность», а «безопасность» остаётся строчной.
        """
        if len(original_words) != len(canonical_words):
            return " ".join(canonical_words)
        return " ".join(
            match_case(original, canonical)
            for original, canonical in zip(original_words, canonical_words, strict=True)
        )

    @staticmethod
    def _is_whitespace_connected(
        text: str, tokens: Sequence[re.Match[str]], index: int, length: int
    ) -> bool:
        """Слова окна разделены только пробелами (без знаков препинания).

        Иначе фраза «информационная, безопасность» совпала бы с термином, а
        замена выкинула бы запятую (или оставила парную скобку). Такие случаи
        не трогаем — это не слитная фраза термина.
        """
        return all(
            not text[tokens[index + offset].end() : tokens[index + offset + 1].start()].strip()
            for offset in range(length - 1)
        )

    def _correct_phrases(self, text: str) -> tuple[str, list[tuple[str, str]]]:
        """Применяет фразовые термины и пары-фразы к ``text``.

        Фразы ищутся как последовательности слов без учёта регистра. Точное
        совпадение не меняется, «лёгкая опечатка» правится к канону (см.
        :meth:`_best_phrase_match`). Явные пары-фразы
        (``ошибочная фраза = канон``) применяются всегда и приоритетнее
        нечёткого сопоставления.
        """
        if not self._phrase_terms and not any(" " in alias for alias in self._aliases):
            return text, []

        phrases_by_length: dict[int, list[str]] = {}
        for phrase in self._phrase_terms:
            phrases_by_length.setdefault(len(phrase.split()), []).append(phrase)

        alias_lengths = [len(alias.split()) for alias in self._aliases if " " in alias]
        lengths = sorted(set(phrases_by_length) | set(alias_lengths), reverse=True)

        tokens = list(WORD_PATTERN.finditer(text))
        replacements: list[tuple[str, str]] = []
        edits: list[tuple[int, int, str]] = []

        index = 0
        while index < len(tokens):
            consumed = 0
            for length in lengths:
                if index + length > len(tokens):
                    continue
                if length > 1 and not self._is_whitespace_connected(text, tokens, index, length):
                    continue
                words = [tokens[index + offset].group(0) for offset in range(length)]
                folded_words = [word.casefold() for word in words]
                window = " ".join(folded_words)

                alias = self._aliases.get(window)
                if alias is not None:
                    canonical = alias
                else:
                    found = self._best_phrase_match(folded_words, phrases_by_length.get(length, []))
                    if found is None:
                        continue
                    canonical, changed = found
                    if not changed:
                        consumed = length
                        break

                start = tokens[index].start()
                end = tokens[index + length - 1].end()
                original = text[start:end]
                new_text = self._apply_phrase_case(words, canonical.split())
                if new_text != original:
                    replacements.append((original, new_text))
                    edits.append((start, end, new_text))
                consumed = length
                break
            index += consumed or 1

        if not edits:
            return text, []

        result = text
        for start, end, new_text in reversed(edits):
            result = result[:start] + new_text + result[end:]
        return result, replacements

    def correct_text(self, text: str) -> tuple[str, list[tuple[str, str]]]:
        """Исправляет опечатки терминов в ``text`` к канону глоссария.

        Сначала применяются явные пары (``АИБ = ОИБ``) и фразовые термины —
        они приоритетнее посимвольного нечёткого сопоставления: так фраза из
        нескольких слов исправляется целиком, пока одиночный матчер не тронул
        её отдельные слова. Затем работает fuzzy-матчер одиночных терминов.
        Возвращает кортеж ``(исправленный текст, замены)``, где ``замены`` —
        список пар ``(как было, как стало)``.
        """
        if not self._single_terms and not self._phrase_terms and not self._aliases:
            return text, []

        # Фразы — первыми: они «поглощают» слова, которые одиночный матчер
        # иначе мог бы исправить по отдельности и разрушить совпадение фразы.
        text, phrase_replacements = self._correct_phrases(text)
        replacements: list[tuple[str, str]] = list(phrase_replacements)

        def replace_word(match: re.Match[str]) -> str:
            original = match.group(0)

            # Явная пара применяется всегда — в т.ч. к коротким формам.
            alias = self._aliases.get(original.casefold())
            if alias is not None:
                if alias != original:
                    replacements.append((original, alias))
                return alias

            if len(original) < self._min_word_length:
                return original
            canonical = self._best_term_match(original)
            if canonical is None:
                return original
            replacement = match_case(original, canonical, acronym=canonical in self._acronym_terms)
            if replacement != original:
                replacements.append((original, replacement))
            return replacement

        corrected = WORD_PATTERN.sub(replace_word, text)
        return corrected, replacements

    # ------------------------------------------------------------------
    # Авто-предложение новых терминов
    # ------------------------------------------------------------------
    def _ensure_morph(self) -> Any:
        if self._morph is None:
            import pymorphy3

            self._morph = pymorphy3.MorphAnalyzer()
        return self._morph

    def _is_known_word(self, word: str) -> bool:
        cached = self._known_cache.get(word)
        if cached is not None:
            return cached
        try:
            morph = self._ensure_morph()
            known = any(parse.is_known for parse in morph.parse(word))
        except Exception:  # noqa: BLE001 — словарь не должен ронять предложение
            known = True
        self._known_cache[word] = known
        return known

    def suggest_new_terms(self, text: str) -> list[str]:
        """Возвращает кандидатов в термины, отсутствующих в глоссарии.

        Кандидатами считаются: (1) аббревиатуры ПРОПИСНЫМИ и (2) слова с
        заглавной буквы, незнакомые морфологическому словарю (имена
        собственные, технические термины). Уже имеющиеся в глоссарии термины
        и обычные слова не предлагаются.
        """
        candidates: set[str] = set()
        for match in WORD_PATTERN.finditer(text):
            word = match.group(0)
            if len(word) < self._min_word_length:
                continue
            folded = word.casefold()
            if folded in self._terms_lower:
                continue
            if folded in self._aliases:
                continue  # известная ошибочная форма — не предлагаем

            if _is_acronym(word):
                candidates.add(word)
                continue

            if word[:1].isupper() and not word.isupper() and not self._is_known_word(word):
                candidates.add(word)

        return sorted(candidates, key=lambda item: item.casefold())


def load_glossary(
    paths: Path | str | Iterable[Path | str] | None,
) -> Glossary | None:
    """Загружает один или несколько файлов глоссария.

    Принимает одиночный путь (:class:`~pathlib.Path` или ``str``), строку со
    списком путей через запятую/``os.pathsep`` либо последовательность путей.
    Термины и явные пары из всех файлов объединяются. Возвращает ``None``,
    если пути не заданы.
    """
    resolved = normalize_glossary_paths(paths)
    if not resolved:
        return None
    return Glossary(paths=resolved)


def count_term_occurrences(text: str, term: str) -> int:
    """Считает, сколько раз ``term`` встречается в ``text`` как целое слово."""
    folded = term.casefold()
    return sum(1 for match in WORD_PATTERN.finditer(text) if match.group(0).casefold() == folded)


def collect_term_suggestions(glossary: Glossary, text: str) -> list[tuple[str, int]]:
    """Возвращает кандидаты в термины с частотой: ``[(термин, частота), …]``.

    Использует детерминированный :meth:`Glossary.suggest_new_terms`, поэтому
    не обращается к LLM. Результат отсортирован по убыванию частоты, затем
    по алфавиту — так чаще встречающиеся термины видны первыми.
    """
    counts = [
        (term, count_term_occurrences(text, term)) for term in glossary.suggest_new_terms(text)
    ]
    counts.sort(key=lambda item: (-item[1], item[0].casefold()))
    return counts


def _suggested_entry_term(line: str) -> str:
    """Возвращает нормализованный термин из строки файла предложений."""
    return line.split("\t", 1)[0].strip().casefold()


def _read_suggested_entries(path: Path) -> list[str]:
    """Читает ранее записанные строки предложений (без заголовка и комментариев)."""
    if not path.is_file():
        return []
    entries: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if _suggested_entry_term(line):
            entries.append(line)
    return entries


def write_suggested_terms(
    glossary: Glossary | None,
    text: str,
    *,
    source: str,
    suggested_path: Path | str | None = None,
) -> Path | None:
    """Дописывает предложенные термины в файл рядом с глоссарием.

    Кандидаты берутся из финального текста стенограммы (уже без опечаток
    глоссария). В файл попадают только термины, которых нет ни в глоссарии,
    ни в предыдущих предложениях, с указанием частоты и источника. Ранее
    записанные строки сохраняются, а не затираются. Возвращает путь к файлу
    или ``None``, если писать некуда (``glossary`` без пути).

    Файл создаётся даже при отсутствии новых кандидатов — с заголовком, чтобы
    путь к нему всегда можно было показать пользователю.
    """
    if glossary is None:
        return None

    path = Path(suggested_path) if suggested_path is not None else glossary.suggested_path
    if path is None:
        return None

    entries = _read_suggested_entries(path)
    already = {_suggested_entry_term(line) for line in entries}
    fresh = [
        (term, count)
        for term, count in collect_term_suggestions(glossary, text)
        if term.casefold() not in already
    ]

    header = [
        "# Предложения новых терминов (сформировано автоматически).",
        "# Формат: <термин>\\t<частота>\\t<источник>.",
        "# Проверьте список и вручную перенесите нужные строки в глоссарий.",
    ]
    body = entries + [f"{term}\t{count}\t{source}" for term, count in fresh]
    if not body:
        header.append("# Новых терминов не найдено.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(header + body) + "\n", encoding="utf-8")
    logger.info("Предложено новых терминов: %d (файл %s)", len(fresh), path)
    return path
