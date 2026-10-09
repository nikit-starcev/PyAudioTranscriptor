import { useCallback, useEffect, useMemo, useState } from 'react'

import {
  api,
  errorMessage,
  type CorrectTextResponse,
  type GlossaryApplyResponse,
  type TextSuggestion,
  type TranscriptResult,
} from '../api'
import { useActionProgress } from '../actionProgress'
import { Alert, Button, Card, CardHeader, Checkbox } from './ui'
import ActionProgressCard from './ActionProgressCard'

type Props = {
  jobId: string
  /** Заменить текущий результат после применения правок. */
  onResult: (result: TranscriptResult) => void
}

type Notice = { kind: 'info' | 'error'; text: string }

//: Человекочитаемые названия видов редакторских правок.
const KIND_LABELS: Record<string, string> = {
  common: 'частые ошибки',
  spelling: 'орфография',
  semantic: 'семантика (LLM)',
}

function kindLabel(kind: string): string {
  return KIND_LABELS[kind] ?? kind
}

/**
 * Панель редакторских действий по текущей стенограмме:
 * применение глоссария по кнопке (#32) и проверка/исправление текста (#51).
 * Результат обеих операций сохраняется на сервере и подменяется в UI.
 */
function EditorPanel({ jobId, onResult }: Props) {
  // Применение глоссария (#32).
  const [glossaryBusy, setGlossaryBusy] = useState(false)
  const [glossaryNotice, setGlossaryNotice] = useState<Notice | null>(null)
  const [respectGlossaryEdits, setRespectGlossaryEdits] = useState(true)

  // Редакторская проверка (#51).
  const [fixCommon, setFixCommon] = useState(true)
  const [checkSpelling, setCheckSpelling] = useState(true)
  const [checkSemantic, setCheckSemantic] = useState(true)
  const [respectEdits, setRespectEdits] = useState(true)
  const [checkBusy, setCheckBusy] = useState(false)
  const [applyBusy, setApplyBusy] = useState(false)
  const [editError, setEditError] = useState<string | null>(null)
  const [appliedNotice, setAppliedNotice] = useState<Notice | null>(null)
  const [suggestions, setSuggestions] = useState<TextSuggestion[]>([])
  const [rejected, setRejected] = useState<Set<string>>(new Set())

  // Прогресс длительных действий редактора (#58): глоссарий и проверка текста.
  const {
    run: actionRun,
    runTask: runActionTask,
    reset: resetActionProgress,
  } = useActionProgress()

  // Смена задачи делает прежние предложения и сообщения неактуальными.
  useEffect(() => {
    setSuggestions([])
    setRejected(new Set())
    setGlossaryNotice(null)
    setAppliedNotice(null)
    setEditError(null)
    resetActionProgress()
  }, [jobId, resetActionProgress])

  const selected = useMemo(
    () => suggestions.filter((item) => !rejected.has(item.id)),
    [suggestions, rejected],
  )

  const applyGlossary = useCallback(async () => {
    setGlossaryBusy(true)
    setGlossaryNotice(null)
    try {
      const response = await runActionTask(
        'glossary',
        (actionId) =>
          api<GlossaryApplyResponse>(`/api/jobs/${jobId}/apply-glossary`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-Action-Id': actionId },
            body: JSON.stringify({ respect_edited: respectGlossaryEdits }),
          }),
        (value) => ({
          text: value.error ? value.error : `Применено замен: ${value.replacements}`,
          error: value.error != null,
        }),
      )
      onResult(response.result)
      if (response.error) {
        setGlossaryNotice({ kind: 'error', text: response.error })
      } else {
        const parts = [`Применено замен: ${response.replacements}`]
        if (response.skipped_edited > 0) {
          parts.push(`пропущено реплик с ручными правками: ${response.skipped_edited}`)
        }
        setGlossaryNotice({ kind: 'info', text: parts.join(' · ') })
      }
    } catch (cause) {
      setGlossaryNotice({ kind: 'error', text: errorMessage(cause) })
    } finally {
      setGlossaryBusy(false)
    }
  }, [jobId, onResult, respectGlossaryEdits, runActionTask])

  const checkText = useCallback(async () => {
    setCheckBusy(true)
    setEditError(null)
    setAppliedNotice(null)
    try {
      const response = await runActionTask(
        'correction',
        (actionId) =>
          api<CorrectTextResponse>(`/api/jobs/${jobId}/correct-text`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-Action-Id': actionId },
            body: JSON.stringify({
              dry_run: true,
              fix_common: fixCommon,
              check_spelling: checkSpelling,
              check_semantic: checkSemantic,
              respect_edited: respectEdits,
            }),
          }),
        (value) => ({
          text:
            value.suggestions.length > 0
              ? `Найдено правок: ${value.suggestions.length}`
              : 'Правок не найдено',
        }),
      )
      setSuggestions(response.suggestions)
      setRejected(new Set())
      if (response.suggestions.length === 0) {
        const suffix =
          response.skipped_edited > 0
            ? ` Пропущено реплик с ручными правками: ${response.skipped_edited}.`
            : ''
        setAppliedNotice({ kind: 'info', text: `Правок не найдено.${suffix}` })
      }
    } catch (cause) {
      setEditError(errorMessage(cause))
    } finally {
      setCheckBusy(false)
    }
  }, [jobId, fixCommon, checkSpelling, checkSemantic, respectEdits, runActionTask])

  const applySelected = useCallback(async () => {
    setApplyBusy(true)
    setEditError(null)
    setAppliedNotice(null)
    try {
      const response = await runActionTask(
        'correction',
        (actionId) =>
          api<CorrectTextResponse>(`/api/jobs/${jobId}/correct-text`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-Action-Id': actionId },
            body: JSON.stringify({
              selection: selected.map((item) => item.id),
              fix_common: fixCommon,
              check_spelling: checkSpelling,
              check_semantic: checkSemantic,
              respect_edited: respectEdits,
            }),
          }),
        (value) => ({ text: `Применено правок: ${value.applied_count}` }),
      )
      onResult(response.result)
      setSuggestions(response.suggestions)
      setRejected(new Set())
      const parts = [`Применено правок: ${response.applied_count}`]
      if (response.skipped_edited > 0) {
        parts.push(`пропущено реплик с ручными правками: ${response.skipped_edited}`)
      }
      setAppliedNotice({ kind: 'info', text: parts.join(' · ') })
    } catch (cause) {
      setEditError(errorMessage(cause))
    } finally {
      setApplyBusy(false)
    }
  }, [
    jobId,
    onResult,
    selected,
    fixCommon,
    checkSpelling,
    checkSemantic,
    respectEdits,
    runActionTask,
  ])

  const toggleSuggestion = useCallback((id: string) => {
    setRejected((current) => {
      const next = new Set(current)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }, [])

  const busy = checkBusy || applyBusy || glossaryBusy

  return (
    <Card>
      <CardHeader
        title="Редактор текста"
        description="Глоссарий и проверка орфографии по готовой стенограмме"
      />
      <div className="space-y-4 p-4">
        <div className="flex flex-wrap items-center gap-3">
          <Button
            variant="secondary"
            loading={glossaryBusy}
            title="Применить матчер глоссария к текущей стенограмме без повторного распознавания"
            onClick={() => void applyGlossary()}
          >
            Применить глоссарий
          </Button>
          <Checkbox
            label="Не трогать реплики, изменённые вручную"
            checked={respectGlossaryEdits}
            onChange={(event) => setRespectGlossaryEdits(event.target.checked)}
          />
        </div>
        {glossaryNotice && (
          <Alert tone={glossaryNotice.kind === 'error' ? 'danger' : 'success'} live>
            {glossaryNotice.text}
          </Alert>
        )}

        <div className="border-t border-border pt-4">
          <div className="flex flex-wrap items-center gap-3">
            <Button
              variant="secondary"
              loading={checkBusy}
              disabled={busy || (!fixCommon && !checkSpelling && !checkSemantic)}
              title="Найти опечатки и частые ошибки в текущей стенограмме"
              onClick={() => void checkText()}
            >
              Проверить текст
            </Button>
            <Checkbox
              label="Частые ошибки"
              checked={fixCommon}
              onChange={(event) => setFixCommon(event.target.checked)}
            />
            <Checkbox
              label="Орфография"
              checked={checkSpelling}
              onChange={(event) => setCheckSpelling(event.target.checked)}
            />
            <Checkbox
              label="Семантика (LLM)"
              checked={checkSemantic}
              onChange={(event) => setCheckSemantic(event.target.checked)}
              title="Показать сохранённые предложения семантической правки LLM (если они есть у задачи)"
            />
            <Checkbox
              label="Не трогать изменённые вручную"
              checked={respectEdits}
              onChange={(event) => setRespectEdits(event.target.checked)}
            />
          </div>

          {editError && (
            <Alert tone="danger" live className="mt-2">
              {editError}
            </Alert>
          )}
          {appliedNotice && (
            <Alert tone="info" live className="mt-2">
              {appliedNotice.text}
            </Alert>
          )}

          {suggestions.length > 0 && (
            <div className="mt-3 space-y-2">
              <div className="flex flex-wrap items-center gap-3 text-xs text-muted">
                <span>
                  Найдено правок: {suggestions.length} · выбрано: {selected.length}
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
                  onClick={() => void applySelected()}
                >
                  Применить выбранные ({selected.length})
                </Button>
              </div>
              <ul className="max-h-64 space-y-1 overflow-auto rounded-md border border-border bg-surface-2/40 p-2 text-sm">
                {suggestions.map((item) => {
                  const accepted = !rejected.has(item.id)
                  return (
                    <li key={item.id} className="flex items-start gap-2">
                      <Checkbox
                        className="mt-1"
                        checked={accepted}
                        onChange={() => toggleSuggestion(item.id)}
                        aria-label="Применить правку"
                      />
                      <span className="min-w-0 flex-1 break-words">
                        <span className="mr-2 rounded bg-surface-3 px-1.5 py-0.5 text-[10px] uppercase text-muted">
                          {kindLabel(item.kind)}
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
                      </span>
                    </li>
                  )
                })}
              </ul>
            </div>
          )}
        </div>
      </div>
      <ActionProgressCard run={actionRun} onClose={resetActionProgress} />
    </Card>
  )
}

export default EditorPanel
