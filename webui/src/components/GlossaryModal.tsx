import { useCallback, useEffect, useState } from 'react'

import {
  api,
  errorMessage,
  type GlossaryEntriesPage,
  type GlossaryEntry,
  type GlossaryImportReport,
  type GlossarySource,
  type GlossaryStats,
} from '../api'

type Props = {
  open: boolean
  onClose: () => void
  onChanged?: () => void
}

type EntryDraft = {
  canonical: string
  variant: string
  note: string
  source: string
}

const EMPTY_DRAFT: EntryDraft = { canonical: '', variant: '', note: '', source: '' }
const PAGE_SIZE = 10
//: Задержка перед запросом поиска глоссария (мс): не дёргаем API на каждый
//: символ, а ждём паузы в наборе (issue #89).
const SEARCH_DEBOUNCE_MS = 300

function GlossaryModal({ open, onClose, onChanged }: Props) {
  const [sources, setSources] = useState<GlossarySource[]>([])
  const [stats, setStats] = useState<GlossaryStats | null>(null)
  const [entries, setEntries] = useState<GlossaryEntry[]>([])
  const [total, setTotal] = useState(0)
  const [offset, setOffset] = useState(0)
  const [search, setSearch] = useState('')
  const [debouncedSearch, setDebouncedSearch] = useState('')
  const [sourceFilter, setSourceFilter] = useState('')
  const [loading, setLoading] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [status, setStatus] = useState<string | null>(null)
  const [draft, setDraft] = useState<EntryDraft>(EMPTY_DRAFT)
  const [edit, setEdit] = useState<{ id: number; canonical: string; variant: string; note: string } | null>(
    null,
  )
  const [importSource, setImportSource] = useState('')
  const [importKind, setImportKind] = useState('auto')
  const [importFile, setImportFile] = useState<File | null>(null)
  const [report, setReport] = useState<GlossaryImportReport | null>(null)

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      const params = new URLSearchParams()
      if (sourceFilter) params.set('source', sourceFilter)
      if (debouncedSearch.trim()) params.set('search', debouncedSearch.trim())
      params.set('limit', String(PAGE_SIZE))
      params.set('offset', String(offset))
      const [sourcesData, statsData, entriesData] = await Promise.all([
        api<GlossarySource[]>('/api/glossary/sources'),
        api<GlossaryStats>('/api/glossary/stats'),
        api<GlossaryEntriesPage>(`/api/glossary/entries?${params.toString()}`),
      ])
      setSources(sourcesData)
      setStats(statsData)
      setEntries(entriesData.entries)
      setTotal(entriesData.total)
      setError(null)
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setLoading(false)
    }
  }, [offset, debouncedSearch, sourceFilter])

  useEffect(() => {
    if (open) void refresh()
  }, [open, refresh])

  // Debounce поиска: обновляем запрос только после паузы в наборе и сразу
  // возвращаемся на первую страницу (offset сбрасывается вместе с запросом).
  useEffect(() => {
    const timer = window.setTimeout(() => {
      setDebouncedSearch(search)
      setOffset(0)
    }, SEARCH_DEBOUNCE_MS)
    return () => window.clearTimeout(timer)
  }, [search])

  useEffect(() => {
    if (!open) return
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])

  if (!open) return null

  const applyChange = async (action: () => Promise<unknown>, message: string) => {
    setBusy(true)
    setError(null)
    setStatus(null)
    try {
      await action()
      setStatus(message)
      await refresh()
      onChanged?.()
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setBusy(false)
    }
  }

  const addEntry = () =>
    applyChange(async () => {
      const canonical = draft.canonical.trim()
      if (!canonical) throw new Error('Укажите канонический термин')
      await api<GlossaryEntry>('/api/glossary/entries', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          canonical,
          variant: draft.variant.trim() || null,
          note: draft.note.trim() || null,
          source: draft.source.trim() || null,
        }),
      })
      setDraft(EMPTY_DRAFT)
    }, 'Запись добавлена')

  const saveEdit = () => {
    if (!edit) return
    const current = edit
    void applyChange(async () => {
      await api<GlossaryEntry>(`/api/glossary/entries/${current.id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          canonical: current.canonical.trim() || undefined,
          variant: current.variant.trim(),
          note: current.note.trim(),
        }),
      })
      setEdit(null)
    }, 'Запись обновлена')
  }

  const toggleEntry = (entry: GlossaryEntry) =>
    applyChange(
      () =>
        api<GlossaryEntry>(`/api/glossary/entries/${entry.id}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ enabled: !entry.enabled }),
        }),
      entry.enabled ? 'Запись отключена' : 'Запись включена',
    )

  const deleteEntry = (entry: GlossaryEntry) => {
    if (!window.confirm(`Удалить запись «${entry.canonical}»?`)) return
    void applyChange(
      () => api<{ deleted: number }>(`/api/glossary/entries/${entry.id}`, { method: 'DELETE' }),
      'Запись удалена',
    )
  }

  const toggleSource = (source: GlossarySource) =>
    applyChange(
      () =>
        api<GlossarySource>(`/api/glossary/sources/${encodeURIComponent(source.name)}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ enabled: !source.enabled }),
        }),
      source.enabled ? 'Источник отключён' : 'Источник включён',
    )

  const deleteSource = (source: GlossarySource) => {
    if (!window.confirm(`Удалить источник «${source.name}» и все его записи?`)) return
    void applyChange(
      () =>
        api<{ deleted: string }>(`/api/glossary/sources/${encodeURIComponent(source.name)}`, {
          method: 'DELETE',
        }),
      'Источник удалён',
    )
  }

  const importFileSubmit = () =>
    applyChange(async () => {
      if (!importFile) throw new Error('Выберите файл .txt или .csv')
      const body = new FormData()
      body.append('file', importFile)
      if (importSource.trim()) body.append('source', importSource.trim())
      if (importKind !== 'auto') body.append('kind', importKind)
      const result = await api<GlossaryImportReport>('/api/glossary/import', {
        method: 'POST',
        body,
      })
      setReport(result)
      setImportFile(null)
      setImportSource('')
      setImportKind('auto')
    }, 'Импорт завершён')

  const pageEnd = Math.min(offset + entries.length, total)
  const canPrev = offset > 0
  const canNext = offset + PAGE_SIZE < total

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-slate-900/40 p-4 sm:items-center dark:bg-black/60"
      onClick={onClose}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Глоссарий"
        className="flex max-h-[88vh] w-full max-w-4xl flex-col overflow-hidden rounded-lg bg-white shadow-xl dark:bg-slate-900"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3 dark:border-slate-800">
          <h2 className="font-medium">Глоссарий</h2>
          <div className="flex items-center gap-3">
            {stats && (
              <span className="text-xs text-slate-400 dark:text-slate-500">
                {stats.sources} источников · {stats.entries} записей ({stats.enabled} вкл.)
              </span>
            )}
            <button
              type="button"
              onClick={onClose}
              className="rounded-md border border-slate-300 px-2 py-1 text-xs hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800"
            >
              Закрыть
            </button>
          </div>
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

          <section className="rounded-md border border-slate-200 p-3 dark:border-slate-800">
            <p className="mb-2 text-xs font-medium text-slate-500 dark:text-slate-400">Источники</p>
            {sources.length === 0 ? (
              <p className="text-sm text-slate-400 dark:text-slate-500">Источников пока нет</p>
            ) : (
              <ul className="divide-y divide-slate-100 dark:divide-slate-800">
                {sources.map((source) => (
                  <li key={source.name} className="flex flex-wrap items-center gap-3 py-2">
                    <label className="flex items-center gap-2 text-sm">
                      <input
                        type="checkbox"
                        checked={source.enabled}
                        disabled={busy}
                        onChange={() => void toggleSource(source)}
                      />
                      <span className="font-medium">{source.name}</span>
                    </label>
                    <span className="rounded bg-slate-100 px-1.5 py-0.5 text-xs text-slate-500 dark:bg-slate-800 dark:text-slate-300">
                      {source.kind}
                    </span>
                    <span className="text-xs text-slate-400 dark:text-slate-500">{source.count} записей</span>
                    {source.path && (
                      <span className="truncate text-xs text-slate-300 dark:text-slate-600" title={source.path}>
                        {source.path}
                      </span>
                    )}
                    <button
                      type="button"
                      onClick={() => void deleteSource(source)}
                      disabled={busy}
                      className="ml-auto rounded-md border border-slate-300 px-2 py-1 text-xs text-slate-500 hover:bg-red-50 hover:text-red-600 disabled:opacity-40 dark:border-slate-700 dark:text-slate-400 dark:hover:bg-red-950/50 dark:hover:text-red-300"
                    >
                      Удалить
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </section>

          <section className="rounded-md border border-slate-200 p-3 dark:border-slate-800">
            <p className="mb-2 text-xs font-medium text-slate-500 dark:text-slate-400">Импорт файла (.txt / .csv)</p>
            <div className="flex flex-wrap items-center gap-2">
              <input
                type="file"
                accept=".txt,.csv,text/plain,text/csv"
                onChange={(event) => setImportFile(event.target.files?.[0] ?? null)}
                className="text-xs"
              />
              <input
                value={importSource}
                onChange={(event) => setImportSource(event.target.value)}
                placeholder="Имя источника"
                className="w-40 rounded-md border border-slate-300 px-2 py-1 text-sm focus:border-blue-400 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
              />
              <select
                value={importKind}
                onChange={(event) => setImportKind(event.target.value)}
                className="rounded-md border border-slate-300 px-2 py-1 text-sm focus:border-blue-400 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
              >
                <option value="auto">авто</option>
                <option value="txt">txt</option>
                <option value="csv">csv</option>
              </select>
              <button
                type="button"
                onClick={() => void importFileSubmit()}
                disabled={busy}
                className="rounded-md bg-slate-800 px-3 py-1.5 text-xs text-white hover:bg-slate-700 disabled:opacity-40 dark:bg-slate-200 dark:text-slate-900 dark:hover:bg-white"
              >
                Импортировать
              </button>
            </div>
            {report && (
              <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
                «{report.source}» ({report.kind}): добавлено {report.added}, пропущено{' '}
                {report.skipped}, всего {report.total}
              </p>
            )}
          </section>

          <section className="rounded-md border border-slate-200 p-3 dark:border-slate-800">
            <p className="mb-2 text-xs font-medium text-slate-500 dark:text-slate-400">Добавить запись</p>
            <div className="grid gap-2 sm:grid-cols-4">
              <input
                value={draft.canonical}
                onChange={(event) => setDraft({ ...draft, canonical: event.target.value })}
                placeholder="Канон *"
                className="rounded-md border border-slate-300 px-2 py-1 text-sm focus:border-blue-400 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
              />
              <input
                value={draft.variant}
                onChange={(event) => setDraft({ ...draft, variant: event.target.value })}
                placeholder="Вариант (ошибка)"
                className="rounded-md border border-slate-300 px-2 py-1 text-sm focus:border-blue-400 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
              />
              <input
                value={draft.note}
                onChange={(event) => setDraft({ ...draft, note: event.target.value })}
                placeholder="Заметка"
                className="rounded-md border border-slate-300 px-2 py-1 text-sm focus:border-blue-400 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
              />
              <input
                value={draft.source}
                onChange={(event) => setDraft({ ...draft, source: event.target.value })}
                placeholder="Источник"
                className="rounded-md border border-slate-300 px-2 py-1 text-sm focus:border-blue-400 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
              />
            </div>
            <button
              type="button"
              onClick={() => void addEntry()}
              disabled={busy}
              className="mt-2 rounded-md bg-slate-800 px-3 py-1.5 text-xs text-white hover:bg-slate-700 disabled:opacity-40 dark:bg-slate-200 dark:text-slate-900 dark:hover:bg-white"
            >
              Добавить
            </button>
          </section>

          <section className="rounded-md border border-slate-200 p-3 dark:border-slate-800">
            <div className="mb-2 flex flex-wrap items-center gap-2">
              <input
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                placeholder="Поиск по записям…"
                className="w-56 rounded-md border border-slate-300 px-2 py-1 text-sm focus:border-blue-400 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
              />
              <select
                value={sourceFilter}
                onChange={(event) => {
                  setSourceFilter(event.target.value)
                  setOffset(0)
                }}
                className="rounded-md border border-slate-300 px-2 py-1 text-sm focus:border-blue-400 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
              >
                <option value="">Все источники</option>
                {sources.map((source) => (
                  <option key={source.name} value={source.name}>
                    {source.name}
                  </option>
                ))}
              </select>
              <span className="ml-auto text-xs text-slate-400 dark:text-slate-500">
                {total === 0 ? '0' : `${offset + 1}–${pageEnd}`} из {total}
              </span>
              <button
                type="button"
                onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
                disabled={!canPrev || loading}
                className="rounded-md border border-slate-300 px-2 py-1 text-xs hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800 disabled:opacity-40"
              >
                ←
              </button>
              <button
                type="button"
                onClick={() => setOffset(offset + PAGE_SIZE)}
                disabled={!canNext || loading}
                className="rounded-md border border-slate-300 px-2 py-1 text-xs hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800 disabled:opacity-40"
              >
                →
              </button>
            </div>

            {loading ? (
              <p className="py-6 text-center text-sm text-slate-400 dark:text-slate-500">Загрузка…</p>
            ) : entries.length === 0 ? (
              <p className="py-6 text-center text-sm text-slate-400 dark:text-slate-500">Записей нет</p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full border-collapse text-sm">
                  <thead className="text-left text-xs uppercase text-slate-500 dark:text-slate-400">
                    <tr>
                      <th className="px-2 py-1.5 font-medium">Вкл</th>
                      <th className="px-2 py-1.5 font-medium">Канон</th>
                      <th className="px-2 py-1.5 font-medium">Вариант</th>
                      <th className="px-2 py-1.5 font-medium">Заметка</th>
                      <th className="px-2 py-1.5 font-medium">Источник</th>
                      <th className="px-2 py-1.5" />
                    </tr>
                  </thead>
                  <tbody>
                    {entries.map((entry) => (
                      <tr key={entry.id} className="border-t border-slate-100 dark:border-slate-800">
                        <td className="px-2 py-1.5">
                          <input
                            type="checkbox"
                            checked={entry.enabled}
                            disabled={busy}
                            onChange={() => void toggleEntry(entry)}
                          />
                        </td>
                        {edit?.id === entry.id ? (
                          <>
                            <td className="px-2 py-1.5">
                              <input
                                value={edit.canonical}
                                onChange={(event) => setEdit({ ...edit, canonical: event.target.value })}
                                className="w-32 rounded border border-slate-300 px-1 py-0.5 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
                              />
                            </td>
                            <td className="px-2 py-1.5">
                              <input
                                value={edit.variant}
                                onChange={(event) => setEdit({ ...edit, variant: event.target.value })}
                                className="w-32 rounded border border-slate-300 px-1 py-0.5 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
                              />
                            </td>
                            <td className="px-2 py-1.5">
                              <input
                                value={edit.note}
                                onChange={(event) => setEdit({ ...edit, note: event.target.value })}
                                className="w-40 rounded border border-slate-300 px-1 py-0.5 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
                              />
                            </td>
                          </>
                        ) : (
                          <>
                            <td className="px-2 py-1.5">{entry.canonical}</td>
                            <td className="px-2 py-1.5 text-slate-500 dark:text-slate-400">{entry.variant ?? '—'}</td>
                            <td className="px-2 py-1.5 text-slate-500 dark:text-slate-400">{entry.note ?? '—'}</td>
                          </>
                        )}
                        <td className="px-2 py-1.5 text-xs text-slate-400 dark:text-slate-500">{entry.source ?? '—'}</td>
                        <td className="whitespace-nowrap px-2 py-1.5 text-right">
                          {edit?.id === entry.id ? (
                            <span className="flex gap-1">
                              <button
                                type="button"
                                onClick={saveEdit}
                                disabled={busy}
                                className="rounded bg-slate-800 px-2 py-0.5 text-xs text-white hover:bg-slate-700 disabled:opacity-40 dark:bg-slate-200 dark:text-slate-900 dark:hover:bg-white"
                              >
                                ОК
                              </button>
                              <button
                                type="button"
                                onClick={() => setEdit(null)}
                                className="rounded border border-slate-300 px-2 py-0.5 text-xs hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800"
                              >
                                Отмена
                              </button>
                            </span>
                          ) : (
                            <span className="flex gap-1">
                              <button
                                type="button"
                                onClick={() =>
                                  setEdit({
                                    id: entry.id,
                                    canonical: entry.canonical,
                                    variant: entry.variant ?? '',
                                    note: entry.note ?? '',
                                  })
                                }
                                className="rounded border border-slate-300 px-2 py-0.5 text-xs hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800"
                              >
                                Правка
                              </button>
                              <button
                                type="button"
                                onClick={() => void deleteEntry(entry)}
                                disabled={busy}
                                className="rounded border border-slate-300 px-2 py-0.5 text-xs text-slate-500 hover:bg-red-50 hover:text-red-600 disabled:opacity-40 dark:border-slate-700 dark:text-slate-400 dark:hover:bg-red-950/50 dark:hover:text-red-300"
                              >
                                Удалить
                              </button>
                            </span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </div>
      </div>
    </div>
  )
}

export default GlossaryModal
