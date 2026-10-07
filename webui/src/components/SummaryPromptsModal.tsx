import { useCallback, useEffect, useState } from 'react'

import {
  activateSummaryPrompt,
  createSummaryPrompt,
  deleteSummaryPrompt,
  errorMessage,
  fetchSummaryPrompts,
  updateSummaryPrompt,
  type SummaryPrompt,
} from '../api'

type Props = {
  open: boolean
  onClose: () => void
  /** Вызывается при смене активного шаблона (id или null). */
  onChanged?: (activeId: number | null) => void
}

type PromptDraft = { name: string; body: string }

const EMPTY_DRAFT: PromptDraft = { name: '', body: '' }

/**
 * Панель управления пользовательскими шаблонами промпта резюме (#97):
 * список, создание/редактирование/удаление, выбор активного. Активный шаблон
 * применяется к задачам и по кнопке «Сформировать протокол».
 */
function SummaryPromptsModal({ open, onClose, onChanged }: Props) {
  const [prompts, setPrompts] = useState<SummaryPrompt[]>([])
  const [activeId, setActiveId] = useState<number | null>(null)
  const [loading, setLoading] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [status, setStatus] = useState<string | null>(null)
  const [draft, setDraft] = useState<PromptDraft>(EMPTY_DRAFT)
  const [edit, setEdit] = useState<{ id: number; name: string; body: string } | null>(null)

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      const data = await fetchSummaryPrompts()
      setPrompts(data.prompts)
      setActiveId(data.active_id)
      setError(null)
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (open) void refresh()
  }, [open, refresh])

  useEffect(() => {
    if (!open) return
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])

  if (!open) return null

  const run = async (action: () => Promise<unknown>, message: string) => {
    setBusy(true)
    setError(null)
    setStatus(null)
    try {
      await action()
      setStatus(message)
      await refresh()
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setBusy(false)
    }
  }

  const addPrompt = () =>
    run(async () => {
      const name = draft.name.trim()
      const body = draft.body.trim()
      if (!name) throw new Error('Укажите имя шаблона')
      if (!body) throw new Error('Тело шаблона не может быть пустым')
      await createSummaryPrompt(name, body)
      setDraft(EMPTY_DRAFT)
    }, 'Шаблон добавлен')

  const saveEdit = () => {
    if (!edit) return
    const current = edit
    void run(async () => {
      const body = current.body.trim()
      if (!body) throw new Error('Тело шаблона не может быть пустым')
      await updateSummaryPrompt(current.id, { name: current.name.trim(), body })
      setEdit(null)
    }, 'Шаблон сохранён')
  }

  const removePrompt = (prompt: SummaryPrompt) => {
    if (!window.confirm(`Удалить шаблон «${prompt.name}»?`)) return
    void run(async () => {
      const result = await deleteSummaryPrompt(prompt.id)
      setActiveId(result.active_id)
      onChanged?.(result.active_id)
    }, 'Шаблон удалён')
  }

  const activate = (prompt: SummaryPrompt) => {
    void run(async () => {
      const result = await activateSummaryPrompt(prompt.id)
      setActiveId(result.active_id)
      onChanged?.(result.active_id)
    }, `Активный шаблон: «${prompt.name}»`)
  }

  const inputClass =
    'mt-1 w-full rounded-md border border-slate-300 px-2 py-1 text-sm focus:border-blue-400 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100'

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-slate-900/40 p-4 sm:items-center dark:bg-black/60"
      onClick={onClose}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Промпты резюме"
        className="flex max-h-[88vh] w-full max-w-3xl flex-col overflow-hidden rounded-lg bg-white shadow-xl dark:bg-slate-900"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3 dark:border-slate-800">
          <h2 className="font-medium">Промпты резюме</h2>
          <button
            type="button"
            onClick={onClose}
            className="rounded-md border border-slate-300 px-2 py-1 text-xs hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800"
          >
            Закрыть
          </button>
        </div>

        <div className="space-y-4 overflow-y-auto px-4 py-3">
          {error && (
            <p className="rounded-md bg-red-50 px-3 py-1.5 text-xs text-red-700 dark:bg-red-950/50 dark:text-red-300">
              {error}
            </p>
          )}
          {status && (
            <p
              role="status"
              className="rounded-md bg-emerald-50 px-3 py-1.5 text-xs text-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-300"
            >
              {status}
            </p>
          )}

          <p className="text-xs text-slate-500 dark:text-slate-400">
            Активный шаблон используется при постановке задачи и по кнопке «Сформировать
            протокол». Свои шаблоны можно создавать, редактировать и удалять.
          </p>

          {loading ? (
            <p className="py-6 text-center text-sm text-slate-400 dark:text-slate-500">Загрузка…</p>
          ) : (
            <ul className="divide-y divide-slate-100 dark:divide-slate-800">
              {prompts.map((prompt) => {
                const isActive = prompt.id === activeId
                const isEditing = edit?.id === prompt.id
                return (
                  <li key={prompt.id} className="py-3">
                    <div className="flex flex-wrap items-center gap-2">
                      <label className="flex min-w-0 flex-1 items-center gap-2 text-sm">
                        <input
                          type="radio"
                          name="active-prompt"
                          checked={isActive}
                          disabled={busy}
                          onChange={() => activate(prompt)}
                        />
                        <span className="min-w-0 break-words font-medium">{prompt.name}</span>
                      </label>
                      {prompt.builtin && (
                        <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-500 dark:bg-slate-800 dark:text-slate-400">
                          встроенный
                        </span>
                      )}
                      {isActive && (
                        <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-xs text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-300">
                          активный
                        </span>
                      )}
                      {!isEditing && (
                        <>
                          <button
                            type="button"
                            disabled={busy}
                            onClick={() =>
                              setEdit({ id: prompt.id, name: prompt.name, body: prompt.body })
                            }
                            className="rounded-md border border-slate-300 px-2 py-1 text-xs hover:bg-slate-100 disabled:opacity-40 dark:border-slate-700 dark:hover:bg-slate-800"
                          >
                            Изменить
                          </button>
                          <button
                            type="button"
                            disabled={busy}
                            onClick={() => removePrompt(prompt)}
                            className="rounded-md border border-slate-300 px-2 py-1 text-xs text-slate-500 hover:bg-red-50 hover:text-red-600 disabled:opacity-40 dark:border-slate-700 dark:text-slate-400 dark:hover:bg-red-950/50 dark:hover:text-red-300"
                          >
                            Удалить
                          </button>
                        </>
                      )}
                    </div>

                    {isEditing && edit ? (
                      <div className="mt-2 space-y-2 rounded-md border border-slate-200 p-2 dark:border-slate-800">
                        <label className="block text-xs text-slate-500 dark:text-slate-400">
                          Имя
                          <input
                            value={edit.name}
                            onChange={(event) => setEdit({ ...edit, name: event.target.value })}
                            className={inputClass}
                          />
                        </label>
                        <label className="block text-xs text-slate-500 dark:text-slate-400">
                          Текст промпта
                          <textarea
                            value={edit.body}
                            rows={6}
                            onChange={(event) => setEdit({ ...edit, body: event.target.value })}
                            className={`${inputClass} font-mono`}
                          />
                        </label>
                        <div className="flex items-center gap-2">
                          <button
                            type="button"
                            disabled={busy}
                            onClick={saveEdit}
                            className="rounded-md bg-slate-800 px-3 py-1 text-xs text-white hover:bg-slate-700 disabled:opacity-40 dark:bg-slate-200 dark:text-slate-900 dark:hover:bg-white"
                          >
                            Сохранить
                          </button>
                          <button
                            type="button"
                            disabled={busy}
                            onClick={() => setEdit(null)}
                            className="rounded-md border border-slate-300 px-3 py-1 text-xs hover:bg-slate-100 disabled:opacity-40 dark:border-slate-700 dark:hover:bg-slate-800"
                          >
                            Отмена
                          </button>
                        </div>
                      </div>
                    ) : (
                      <p className="mt-1 line-clamp-3 whitespace-pre-wrap text-xs text-slate-500 dark:text-slate-400">
                        {prompt.body}
                      </p>
                    )}
                  </li>
                )
              })}
            </ul>
          )}

          <fieldset className="space-y-2 rounded-md border border-slate-200 p-3 dark:border-slate-800">
            <legend className="px-1 text-xs font-medium text-slate-500 dark:text-slate-400">
              Новый шаблон
            </legend>
            <label className="block text-xs text-slate-500 dark:text-slate-400">
              Имя
              <input
                value={draft.name}
                onChange={(event) => setDraft({ ...draft, name: event.target.value })}
                placeholder="Например, «Кратко для email»"
                className={inputClass}
              />
            </label>
            <label className="block text-xs text-slate-500 dark:text-slate-400">
              Текст промпта (системная инструкция для модели)
              <textarea
                value={draft.body}
                rows={6}
                onChange={(event) => setDraft({ ...draft, body: event.target.value })}
                placeholder="Ты — ассистент, который…"
                className={`${inputClass} font-mono`}
              />
            </label>
            <button
              type="button"
              disabled={busy}
              onClick={addPrompt}
              className="rounded-md bg-slate-800 px-3 py-1.5 text-sm text-white hover:bg-slate-700 disabled:opacity-40 dark:bg-slate-200 dark:text-slate-900 dark:hover:bg-white"
            >
              Добавить шаблон
            </button>
          </fieldset>
        </div>
      </div>
    </div>
  )
}

export default SummaryPromptsModal
