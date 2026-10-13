import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import {
  api,
  errorMessage,
  type CorrectTextResponse,
  type Entry,
  type TextSuggestion,
  type TranscriptResult,
} from '../api'
import { Alert, Button, Card, CardHeader, Checkbox, Spinner, cn } from './ui'
import { renderHighlightedText } from './highlight'

type Props = {
  jobId: string
  /** Заменить текущий результат после применения выбранных правок. */
  onResult: (result: TranscriptResult) => void
  /** Отдать текущие предложения наверх — для подсветки в стенограмме (#120). */
  onSuggestions?: (items: TextSuggestion[]) => void
  /** Выбрать правку: переход к реплике в стенограмме (#120). */
  onSelect?: (item: TextSuggestion) => void
  /** Наведение на правку: акцент подсветки в стенограмме (#120). */
  onHover?: (id: string | null) => void
  /** Идентификатор выбранной правки (#120). */
  activeSuggestionId?: string | null
  /** Отдать список отклонённых правок — для приглушённой подсветки (#120). */
  onRejectedChange?: (ids: string[]) => void
}

//: Тело автозагрузки сохранённых семантических предложений (#75).
const LOAD_BODY = {
  dry_run: true,
  check_semantic: true,
  fix_common: false,
  check_spelling: false,
  respect_edited: true,
} as const

function suggestionBody(selection: string[]) {
  return {
    selection,
    fix_common: false,
    check_spelling: false,
    check_semantic: true,
    respect_edited: true,
  }
}

/**
 * Отдельный блок семантических правок LLM (#75).
 *
 * Предложения подтягиваются автоматически при открытии задачи и после
 * применения — без кнопки «Проверить текст». Показываются тем же списком
 * «принять/отклонить», что и редакторские правки; ничего не применяется само.
 */
function SemanticSuggestionsPanel({
  jobId,
  onResult,
  onSuggestions,
  onSelect,
  onHover,
  activeSuggestionId,
  onRejectedChange,
}: Props) {
  const [loading, setLoading] = useState(false)
  const [applyBusy, setApplyBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [suggestions, setSuggestions] = useState<TextSuggestion[]>([])
  const [rejected, setRejected] = useState<Set<string>>(new Set())
  // Реплики текущего результата — контекст правок в сводке (#120).
  const [contextEntries, setContextEntries] = useState<Entry[]>([])
  const requestRef = useRef(0)

  const load = useCallback(async () => {
    const requestId = ++requestRef.current
    setLoading(true)
    setError(null)
    try {
      const response = await api<CorrectTextResponse>(
        `/api/jobs/${jobId}/correct-text`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(LOAD_BODY),
        },
      )
      if (requestRef.current !== requestId) return
      setSuggestions(response.suggestions)
      setContextEntries(response.result.entries)
      setRejected(new Set())
    } catch (cause) {
      if (requestRef.current !== requestId) return
      setSuggestions([])
      setContextEntries([])
      setRejected(new Set())
      setError(errorMessage(cause))
    } finally {
      if (requestRef.current === requestId) setLoading(false)
    }
  }, [jobId])

  // Автозагрузка при открытии задачи и смене jobId.
  useEffect(() => {
    setSuggestions([])
    setContextEntries([])
    setRejected(new Set())
    setNotice(null)
    setError(null)
    void load()
    return () => {
      requestRef.current += 1
    }
  }, [jobId, load])

  // Текущие предложения отдаём наверх: стенограмма подсвечивает их в тексте (#120).
  useEffect(() => {
    onSuggestions?.(suggestions)
  }, [suggestions, onSuggestions])

  // Отклонённые правки отдаём наверх: подсветка в стенограмме приглушается (#120).
  useEffect(() => {
    onRejectedChange?.(Array.from(rejected))
  }, [rejected, onRejectedChange])

  const selected = useMemo(
    () => suggestions.filter((item) => !rejected.has(item.id)),
    [suggestions, rejected],
  )

  const apply = useCallback(async () => {
    setApplyBusy(true)
    setError(null)
    setNotice(null)
    try {
      const response = await api<CorrectTextResponse>(
        `/api/jobs/${jobId}/correct-text`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(suggestionBody(selected.map((item) => item.id))),
        },
      )
      onResult(response.result)
      const parts = [`Применено правок: ${response.applied_count}`]
      if (response.skipped_edited > 0) {
        parts.push(`пропущено реплик с ручными правками: ${response.skipped_edited}`)
      }
      setNotice(parts.join(' · '))
      await load()
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setApplyBusy(false)
    }
  }, [jobId, onResult, selected, load])

  const toggleSuggestion = useCallback((id: string) => {
    setRejected((current) => {
      const next = new Set(current)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }, [])

  const showEmpty = !loading && !error && suggestions.length === 0

  return (
    <Card>
      <CardHeader
        title="Предложения семантической правки (LLM)"
        description="Найдены на этапе обработки; применяются только выбранные"
        actions={loading ? <Spinner label="Загрузка семантических предложений" /> : undefined}
      />
      <div className="space-y-3 p-4">
        {error && (
          <Alert tone="danger" live>
            {error}
          </Alert>
        )}
        {notice && (
          <Alert tone="success" live>
            {notice}
          </Alert>
        )}

        {loading && suggestions.length === 0 && !error && (
          <p className="text-sm text-muted">Загрузка предложений…</p>
        )}

        {showEmpty && <p className="text-sm text-muted">Семантических предложений нет.</p>}

        {suggestions.length > 0 && (
          <div className="space-y-2">
            <div className="flex flex-wrap items-center gap-3 text-xs text-muted">
              <span>
                Предложений: {suggestions.length} · выбрано: {selected.length}
              </span>
              <Button variant="secondary" size="sm" onClick={() => setRejected(new Set())}>
                Принять все
              </Button>
              <Button
                variant="secondary"
                size="sm"
                onClick={() => setRejected(new Set(suggestions.map((item) => item.id)))}
              >
                Отклонить все
              </Button>
              <Button
                variant="primary"
                size="sm"
                className="ml-auto"
                loading={applyBusy}
                disabled={selected.length === 0}
                onClick={() => void apply()}
              >
                Применить выбранные ({selected.length})
              </Button>
            </div>
            <ul className="max-h-64 space-y-1 overflow-auto rounded-md border border-border bg-surface-2/40 p-2 text-sm">
              {suggestions.map((item) => {
                const accepted = !rejected.has(item.id)
                const context = contextEntries[item.index]
                return (
                  <li
                    key={item.id}
                    className={cn(
                      'flex cursor-pointer items-start gap-2 rounded-md p-1 transition-colors',
                      activeSuggestionId === item.id ? 'bg-primary-soft' : 'hover:bg-surface-2',
                    )}
                    title="Перейти к месту правки в стенограмме"
                    onClick={() => onSelect?.(item)}
                    onMouseEnter={() => onHover?.(item.id)}
                    onMouseLeave={() => onHover?.(null)}
                  >
                    <Checkbox
                      className="mt-1"
                      checked={accepted}
                      onChange={() => toggleSuggestion(item.id)}
                      onClick={(event) => event.stopPropagation()}
                      aria-label="Применить правку"
                    />
                    <span className="min-w-0 flex-1 break-words">
                      <span className="mr-2 rounded bg-surface-3 px-1.5 py-0.5 text-[10px] uppercase text-muted">
                        семантика (LLM)
                      </span>
                      <span className="rounded bg-danger-soft px-1 text-danger-soft-fg line-through">
                        {item.before}
                      </span>
                      <span className="mx-1 text-muted">→</span>
                      <span className="rounded bg-success-soft px-1 text-success-soft-fg">
                        {item.after}
                      </span>
                      <span className="ml-2 text-xs text-muted">
                        реплика #{item.index + 1} · {item.reason}
                      </span>
                      {context && (
                        <span className="mt-0.5 block text-xs text-muted">
                          {renderHighlightedText(
                            context.text,
                            [{ id: item.id, start: item.start, end: item.end }],
                            `semantic-ctx-${item.id}`,
                          )}
                        </span>
                      )}
                    </span>
                  </li>
                )
              })}
            </ul>
          </div>
        )}
      </div>
    </Card>
  )
}

export default SemanticSuggestionsPanel
