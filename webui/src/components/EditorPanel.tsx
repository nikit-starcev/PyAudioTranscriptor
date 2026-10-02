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
  }, [jobId, fixCommon, checkSpelling, respectEdits, runActionTask])

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
  }, [jobId, onResult, selected, fixCommon, checkSpelling, respectEdits, runActionTask])

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
    <div className="space-y-3 rounded-md border border-slate-200 bg-slate-50 p-3 dark:border-slate-800 dark:bg-slate-800/50">
      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={() => void applyGlossary()}
          disabled={glossaryBusy}
          title="Применить матчер глоссария к текущей стенограмме без повторного распознавания"
          className="rounded-md border border-emerald-300 px-3 py-1.5 text-sm text-emerald-700 hover:bg-emerald-50 disabled:cursor-not-allowed disabled:opacity-40 dark:border-emerald-800 dark:text-emerald-300 dark:hover:bg-emerald-950/50"
        >
          {glossaryBusy ? 'Применяю глоссарий…' : 'Применить глоссарий'}
        </button>
        <label className="flex items-center gap-1.5 text-xs text-slate-500 dark:text-slate-400">
          <input
            type="checkbox"
            checked={respectGlossaryEdits}
            onChange={(event) => setRespectGlossaryEdits(event.target.checked)}
          />
          Не трогать реплики, изменённые вручную
        </label>
      </div>
      {glossaryNotice && (
        <p
          role="status"
          className={`rounded-md px-3 py-1.5 text-xs ${
            glossaryNotice.kind === 'error'
              ? 'bg-red-50 text-red-700 dark:bg-red-950/50 dark:text-red-300'
              : 'bg-emerald-50 text-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-300'
          }`}
        >
          {glossaryNotice.text}
        </p>
      )}

      <div className="border-t border-slate-200 pt-3 dark:border-slate-700">
        <div className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={() => void checkText()}
            disabled={busy || (!fixCommon && !checkSpelling)}
            title="Найти опечатки и частые ошибки в текущей стенограмме"
            className="rounded-md border border-sky-300 px-3 py-1.5 text-sm text-sky-700 hover:bg-sky-50 disabled:cursor-not-allowed disabled:opacity-40 dark:border-sky-800 dark:text-sky-300 dark:hover:bg-sky-950/50"
          >
            {checkBusy ? 'Проверяю текст…' : 'Проверить текст'}
          </button>
          <label className="flex items-center gap-1.5 text-xs text-slate-500 dark:text-slate-400">
            <input
              type="checkbox"
              checked={fixCommon}
              onChange={(event) => setFixCommon(event.target.checked)}
            />
            Частые ошибки
          </label>
          <label className="flex items-center gap-1.5 text-xs text-slate-500 dark:text-slate-400">
            <input
              type="checkbox"
              checked={checkSpelling}
              onChange={(event) => setCheckSpelling(event.target.checked)}
            />
            Орфография
          </label>
          <label className="flex items-center gap-1.5 text-xs text-slate-500 dark:text-slate-400">
            <input
              type="checkbox"
              checked={respectEdits}
              onChange={(event) => setRespectEdits(event.target.checked)}
            />
            Не трогать изменённые вручную
          </label>
        </div>

        {editError && (
          <p className="mt-2 rounded-md bg-red-50 px-3 py-1.5 text-xs text-red-700 dark:bg-red-950/50 dark:text-red-300">
            {editError}
          </p>
        )}
        {appliedNotice && (
          <p
            role="status"
            className={`mt-2 rounded-md px-3 py-1.5 text-xs ${
              appliedNotice.kind === 'error'
                ? 'bg-red-50 text-red-700 dark:bg-red-950/50 dark:text-red-300'
                : 'bg-sky-50 text-sky-700 dark:bg-sky-950/50 dark:text-sky-300'
            }`}
          >
            {appliedNotice.text}
          </p>
        )}

        {suggestions.length > 0 && (
          <div className="mt-3 space-y-2">
            <div className="flex flex-wrap items-center gap-3 text-xs text-slate-500 dark:text-slate-400">
              <span>
                Найдено правок: {suggestions.length} · выбрано: {selected.length}
              </span>
              <button
                type="button"
                onClick={() => setRejected(new Set())}
                className="rounded border border-slate-300 px-2 py-0.5 hover:bg-slate-100 dark:border-slate-600 dark:hover:bg-slate-800"
              >
                Принять все
              </button>
              <button
                type="button"
                onClick={() => setRejected(new Set(suggestions.map((item) => item.id)))}
                className="rounded border border-slate-300 px-2 py-0.5 hover:bg-slate-100 dark:border-slate-600 dark:hover:bg-slate-800"
              >
                Отклонить все
              </button>
              <button
                type="button"
                onClick={() => void applySelected()}
                disabled={applyBusy || selected.length === 0}
                className="ml-auto rounded-md bg-slate-800 px-3 py-1 text-white hover:bg-slate-700 disabled:opacity-40 dark:bg-slate-200 dark:text-slate-900 dark:hover:bg-white"
              >
                {applyBusy ? 'Применяю…' : `Применить выбранные (${selected.length})`}
              </button>
            </div>
            <ul className="max-h-64 space-y-1 overflow-auto rounded-md border border-slate-200 bg-white p-2 text-sm dark:border-slate-700 dark:bg-slate-900">
              {suggestions.map((item) => {
                const accepted = !rejected.has(item.id)
                return (
                  <li key={item.id} className="flex items-start gap-2">
                    <input
                      type="checkbox"
                      checked={accepted}
                      onChange={() => toggleSuggestion(item.id)}
                      className="mt-1"
                      aria-label="Применить правку"
                    />
                    <span className="min-w-0 flex-1 break-words">
                      <span className="mr-2 rounded bg-slate-100 px-1.5 py-0.5 text-[10px] uppercase text-slate-500 dark:bg-slate-800 dark:text-slate-400">
                        {kindLabel(item.kind)}
                      </span>
                      <span className="rounded bg-red-50 px-1 text-red-700 line-through dark:bg-red-950/40 dark:text-red-300">
                        {item.before}
                      </span>
                      <span className="mx-1 text-slate-400">→</span>
                      <span className="rounded bg-emerald-50 px-1 text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-300">
                        {item.after}
                      </span>
                      <span className="ml-2 text-xs text-slate-400 dark:text-slate-500">
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
      <ActionProgressCard run={actionRun} onClose={resetActionProgress} />
    </div>
  )
}

export default EditorPanel
