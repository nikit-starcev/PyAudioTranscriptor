import type { ReactNode } from 'react'

import { cn } from './ui'

/** Диапазон подсветки изменяемого фрагмента в тексте реплики (#120). */
export type HighlightSpan = {
  id: string
  start: number
  end: number
  /** Акцент: правка выбрана в списке или наведена курсором. */
  accent?: boolean
  /** Правка отклонена в списке — фрагмент помечается как неприменяемый. */
  rejected?: boolean
}

export type HighlightPart = { text: string; span: HighlightSpan | null }

/** Режет текст на части по диапазонам правок, обрезая их по границам текста. */
export function splitBySpans(text: string, spans: HighlightSpan[]): HighlightPart[] {
  const ordered = spans
    .filter((span) => span.end > span.start && span.start < text.length)
    .map((span) => ({
      ...span,
      start: Math.max(0, span.start),
      end: Math.min(text.length, span.end),
    }))
    .sort((a, b) => a.start - b.start)

  const parts: HighlightPart[] = []
  let cursor = 0
  for (const span of ordered) {
    if (span.start < cursor) continue
    if (span.start > cursor) parts.push({ text: text.slice(cursor, span.start), span: null })
    parts.push({ text: text.slice(span.start, span.end), span })
    cursor = span.end
  }
  if (cursor < text.length) parts.push({ text: text.slice(cursor), span: null })
  return parts
}

/**
 * Рендерит текст с подсветкой изменяемых фрагментов (#120): базовая —
 * красным с подчёркиванием, акцентная — при выборе/наведении в списке правок.
 */
export function renderHighlightedText(
  text: string,
  spans: HighlightSpan[],
  keyPrefix: string,
): ReactNode[] {
  return splitBySpans(text, spans).map((part, index) => {
    if (!part.span) return part.text
    return (
      <mark
        key={`${keyPrefix}-${part.span.id}-${index}`}
        className={cn(
          'rounded-sm bg-danger-soft text-danger-soft-fg underline decoration-danger decoration-2 underline-offset-2',
          part.span.rejected && 'line-through decoration-danger-soft-fg opacity-60',
          part.span.accent && 'bg-warn-soft text-warn-soft-fg ring-1 ring-warn',
        )}
      >
        {part.text}
      </mark>
    )
  })
}
