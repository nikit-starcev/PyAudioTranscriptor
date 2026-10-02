"""Редакторские правки стенограммы: частые ошибки и орфография (#51).

Модуль не переписывает смысл, а собирает **предложения** правок, которые UI
показывает списком «принять/отклонить»:

* правила «частых ошибок» — детерминированные и консервативные: пробелы,
  пунктуация, дефис вместо тире, пропущенный пробел после знака. Правила не
  трогают слова;
* орфография — переиспользуется консервативный морфологический корректор
  (:class:`~audio_transcriber.correction.morph_corrector.MorphTextCorrector`),
  который правит только неизвестные словоформы ближайшим словарным словом.

Каждое предложение несёт позиции ``start``/``end`` в исходном тексте реплики,
поэтому выбранные правки применяются к исходному тексту точно, а не поиском
подстроки. Операция детерминирована и не обращается к сети.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from audio_transcriber.correction.morph_corrector import MorphTextCorrector

#: Вид правки: правило частых ошибок.
KIND_COMMON = "common"
#: Вид правки: орфография (морфология).
KIND_SPELLING = "spelling"


@dataclass(frozen=True, slots=True)
class Suggestion:
    """Одно предложение правки текста реплики.

    ``index`` — индекс реплики в списке результата; ``start``/``end`` — позиции
    фрагмента в исходном тексте реплики; ``before``/``after`` — что меняется.
    ``id`` стабилен для одного и того же текста и служит для выбора правок.
    """

    index: int
    start: int
    end: int
    before: str
    after: str
    kind: str
    reason: str

    @property
    def id(self) -> str:
        return f"{self.index}:{self.start}:{self.end}:{self.kind}"

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "before": self.before,
            "after": self.after,
            "kind": self.kind,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class _Rule:
    """Правило частых ошибок: регулярное выражение и замена к нему."""

    pattern: re.Pattern[str]
    replacement: str
    reason: str


#: Консервативные правила «частых ошибок». Все они работают с «оболочкой»
#: текста (пробелы, знаки, тире) и никогда не меняют буквы слов.
_COMMON_RULES: tuple[_Rule, ...] = (
    _Rule(re.compile(r"^[ \t]+"), "", "лишние пробелы в начале"),
    _Rule(re.compile(r"[ \t]+$"), "", "лишние пробелы в конце"),
    _Rule(re.compile(r"[ \t]{2,}"), " ", "двойной пробел"),
    # Запятая/точка с запятой/двоеточие: убираем пробел перед знаком и
    # гарантируем пробел после, если дальше идёт буква. Одно правило вместо
    # двух — иначе правки перекрывались и пробел после знака терялся.
    _Rule(
        re.compile(r"[ \t]*([,;:])([ \t]*)(?=[А-Яа-яЁёA-Za-z])"),
        r"\1 ",
        "пробел вокруг знака",
    ),
    _Rule(re.compile(r"[ \t]+([,.;:!?…)\]»])"), r"\1", "пробел перед знаком препинания"),
    _Rule(re.compile(r"([(\[«])[ \t]+"), r"\1", "пробел после открывающей скобки"),
    _Rule(re.compile(r"(?<=\d)[ \t]+(%)"), r"\1", "пробел перед знаком «%»"),
    _Rule(re.compile(r"([!?;:,])\1+"), r"\1", "повтор пунктуации"),
    _Rule(
        re.compile(r"(?<=[А-Яа-яЁёA-Za-z])[ \t]+-[ \t]+(?=[А-Яа-яЁёA-Za-z])"),
        " — ",
        "дефис вместо тире",
    ),
    _Rule(re.compile(r"[ \t]+—[ \t]+"), " — ", "лишние пробелы вокруг тире"),
    _Rule(
        re.compile(r"(?<=[а-яё])([.!?])(?=[А-ЯЁ])"),
        r"\1 ",
        "нет пробела после точки",
    ),
)


def _drop_overlaps(items: Sequence[Suggestion]) -> list[Suggestion]:
    """Убирает пересекающиеся предложения, сохраняя порядок по позиции.

    При одинаковой левой границе предпочтение отдаётся более длинному
    фрагменту — так правка «съедает» больше проблемного текста за раз.
    """
    ordered = sorted(items, key=lambda item: (item.start, -(item.end - item.start)))
    kept: list[Suggestion] = []
    last_end = -1
    for item in ordered:
        if item.start < last_end:
            continue
        kept.append(item)
        last_end = item.end
    return kept


def suggest_common(text: str, *, entry_index: int) -> list[Suggestion]:
    """Предложения по правилам частых ошибок (без орфографии)."""
    found: list[Suggestion] = []
    for rule in _COMMON_RULES:
        for match in rule.pattern.finditer(text):
            before = match.group(0)
            after = match.expand(rule.replacement)
            if before == after:
                continue
            found.append(
                Suggestion(
                    index=entry_index,
                    start=match.start(),
                    end=match.end(),
                    before=before,
                    after=after,
                    kind=KIND_COMMON,
                    reason=rule.reason,
                )
            )
    return _drop_overlaps(found)


def suggest_spelling(
    text: str,
    *,
    entry_index: int,
    corrector: MorphTextCorrector,
) -> list[Suggestion]:
    """Предложения по орфографии через морфологический корректор."""
    return [
        Suggestion(
            index=entry_index,
            start=start,
            end=end,
            before=before,
            after=after,
            kind=KIND_SPELLING,
            reason="опечатка (морфология)",
        )
        for start, end, before, after in corrector.suggest_spans(text)
    ]


def build_suggestions(
    text: str,
    *,
    entry_index: int,
    fix_common: bool = True,
    check_spelling: bool = True,
    corrector: MorphTextCorrector | None = None,
) -> list[Suggestion]:
    """Собирает предложения правок для одной реплики.

    ``corrector`` переиспользуется для всех реплик (в нём ленивый словарь
    pymorphy3), поэтому передавайте один экземпляр на весь результат.
    """
    items: list[Suggestion] = []
    if fix_common:
        items.extend(suggest_common(text, entry_index=entry_index))
    if check_spelling and corrector is not None:
        items.extend(suggest_spelling(text, entry_index=entry_index, corrector=corrector))
    return _drop_overlaps(items)


def apply_suggestions(
    text: str, suggestions: Sequence[Suggestion]
) -> tuple[str, list[Suggestion]]:
    """Применяет выбранные предложения к исходному тексту.

    Позиции ``start``/``end`` указывают на исходный текст, поэтому правки
    накладываются от конца к началу устойчиво к сдвигу смещений.
    Пересекающиеся предложения пропускаются (их «поглотил» более ранний
    фрагмент). Возвращает новый текст и список фактически применённых правок.
    """
    ordered = sorted(suggestions, key=lambda item: item.start)
    parts: list[str] = []
    applied: list[Suggestion] = []
    cursor = 0
    for item in ordered:
        if item.start < cursor:
            continue
        parts.append(text[cursor:item.start])
        parts.append(item.after)
        cursor = item.end
        applied.append(item)
    parts.append(text[cursor:])
    return "".join(parts), applied
