import { useCallback, useEffect, useState } from 'react'

import { api, errorMessage, type GlossaryEntry, type GlossaryQuickRequest } from '../api'

type Props = {
  open: boolean
  /** Выделенный в стенограмме фрагмент — предзаполняет канон (#17). */
  term: string
  onClose: () => void
  /** Термин сохранён в глоссарии — можно обновить список. */
  onSaved?: (entry: GlossaryEntry) => void
}

/**
 * Модальное окно быстрого добавления термина в глоссарий из выделения.
 *
 * Поля: канон (предзаполнен выделением), ошибочная форма, источник, заметка.
 * Пустой канон означает «канон = выделение»; если канон отличается от
 * выделения, а ошибочная форма не задана, выделение сохраняется как вариант.
 */
function GlossaryQuickModal({ open, term, onClose, onSaved }: Props) {
  const [canonical, setCanonical] = useState(term)
  const [variant, setVariant] = useState('')
  const [note, setNote] = useState('')
  const [source, setSource] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [saved, setSaved] = useState<GlossaryEntry | null>(null)

  useEffect(() => {
    if (!open) return
    setCanonical(term)
    setVariant('')
    setNote('')
    setSource('')
    setError(null)
    setSaved(null)
  }, [open, term])

  useEffect(() => {
    if (!open) return
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])

  const submit = useCallback(async () => {
    const trimmed = canonical.trim() || term.trim()
    if (!trimmed) {
      setError('Укажите канонический термин')
      return
    }
    setBusy(true)
    setError(null)
    try {
      const body: GlossaryQuickRequest = {
        term,
        canonical: canonical.trim() || null,
        variant: variant.trim() || null,
        note: note.trim() || null,
        source: source.trim() || null,
      }
      const entry = await api<GlossaryEntry>('/api/glossary/quick', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      setSaved(entry)
      onSaved?.(entry)
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setBusy(false)
    }
  }, [canonical, term, variant, note, source, onSaved])

  if (!open) return null

  return (
    <div
      className="fixed inset-0 z-[60] flex items-start justify-center overflow-y-auto bg-slate-900/40 p-4 sm:items-center dark:bg-black/60"
      onClick={onClose}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Добавить термин в глоссарий"
        className="w-full max-w-lg rounded-lg bg-white shadow-xl dark:bg-slate-900"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3 dark:border-slate-800">
          <h2 className="font-medium">Добавить в глоссарий</h2>
          <button
            type="button"
            onClick={onClose}
            className="rounded-md border border-slate-300 px-2 py-1 text-xs hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800"
          >
            Закрыть
          </button>
        </div>

        <div className="space-y-3 px-4 py-3">
          <p className="rounded-md bg-slate-50 px-3 py-2 text-xs text-slate-500 dark:bg-slate-800/60 dark:text-slate-400">
            Выделенный фрагмент: <span className="font-medium text-slate-700 dark:text-slate-200">{term}</span>
          </p>

          {error && (
            <p className="rounded-md bg-red-50 px-3 py-1.5 text-xs text-red-700 dark:bg-red-950/50 dark:text-red-300">
              {error}
            </p>
          )}
          {saved && (
            <p
              role="status"
              className="rounded-md bg-emerald-50 px-3 py-1.5 text-xs text-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-300"
            >
              Сохранено: «{saved.canonical}»{saved.variant ? ` ← ${saved.variant}` : ''} (id {saved.id})
            </p>
          )}

          <label className="block text-xs text-slate-500 dark:text-slate-400">
            Канон
            <input
              value={canonical}
              onChange={(event) => setCanonical(event.target.value)}
              placeholder="Правильное написание"
              className="mt-1 w-full rounded-md border border-slate-300 px-2 py-1.5 text-sm text-slate-800 focus:border-blue-400 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
            />
          </label>

          <label className="block text-xs text-slate-500 dark:text-slate-400">
            Ошибочная форма
            <input
              value={variant}
              onChange={(event) => setVariant(event.target.value)}
              placeholder="Как распознано (вариант)"
              className="mt-1 w-full rounded-md border border-slate-300 px-2 py-1.5 text-sm text-slate-800 focus:border-blue-400 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
            />
          </label>

          <div className="grid gap-3 sm:grid-cols-2">
            <label className="block text-xs text-slate-500 dark:text-slate-400">
              Источник
              <input
                value={source}
                onChange={(event) => setSource(event.target.value)}
                placeholder="Например, встреча"
                className="mt-1 w-full rounded-md border border-slate-300 px-2 py-1.5 text-sm text-slate-800 focus:border-blue-400 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
              />
            </label>
            <label className="block text-xs text-slate-500 dark:text-slate-400">
              Заметка
              <input
                value={note}
                onChange={(event) => setNote(event.target.value)}
                placeholder="Комментарий"
                className="mt-1 w-full rounded-md border border-slate-300 px-2 py-1.5 text-sm text-slate-800 focus:border-blue-400 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
              />
            </label>
          </div>
        </div>

        <div className="flex items-center justify-end gap-2 border-t border-slate-200 px-4 py-3 dark:border-slate-800">
          {saved ? (
            <button
              type="button"
              onClick={onClose}
              className="rounded-md bg-slate-800 px-3 py-1.5 text-sm text-white hover:bg-slate-700 dark:bg-slate-200 dark:text-slate-900 dark:hover:bg-white"
            >
              Готово
            </button>
          ) : (
            <>
              <button
                type="button"
                onClick={onClose}
                className="rounded-md border border-slate-300 px-3 py-1.5 text-sm hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800"
              >
                Отмена
              </button>
              <button
                type="button"
                onClick={() => void submit()}
                disabled={busy}
                className="rounded-md bg-slate-800 px-3 py-1.5 text-sm text-white hover:bg-slate-700 disabled:opacity-40 dark:bg-slate-200 dark:text-slate-900 dark:hover:bg-white"
              >
                {busy ? 'Сохранение…' : 'Сохранить'}
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

export default GlossaryQuickModal
