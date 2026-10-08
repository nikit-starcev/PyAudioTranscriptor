import { useCallback, useEffect, useState } from 'react'
import { ChevronLeft, ChevronRight, Pencil, Search, Trash2 } from 'lucide-react'

import {
  api,
  errorMessage,
  type GlossaryEntriesPage,
  type GlossaryEntry,
  type GlossaryImportReport,
  type GlossarySource,
  type GlossaryStats,
} from '../api'
import {
  Alert,
  Badge,
  Button,
  Card,
  CardContent,
  CardHeader,
  Checkbox,
  EmptyState,
  Field,
  FileInput,
  IconButton,
  Input,
  Select,
  Spinner,
  Tooltip,
} from './ui'

type Props = {
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

function GlossaryPanel({ onChanged }: Props) {
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
  const [edit, setEdit] = useState<{
    id: number
    canonical: string
    variant: string
    note: string
  } | null>(null)
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
    void refresh()
  }, [refresh])

  // Debounce поиска: обновляем запрос только после паузы в наборе и сразу
  // возвращаемся на первую страницу (offset сбрасывается вместе с запросом).
  useEffect(() => {
    const timer = window.setTimeout(() => {
      setDebouncedSearch(search)
      setOffset(0)
    }, SEARCH_DEBOUNCE_MS)
    return () => window.clearTimeout(timer)
  }, [search])

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
    <div className="space-y-5">
      {stats && (
        <p className="text-xs text-muted">
          {stats.sources} источников · {stats.entries} записей ({stats.enabled} вкл.)
        </p>
      )}

      {error && (
        <Alert tone="danger" live onDismiss={() => setError(null)}>
          {error}
        </Alert>
      )}
      {status && (
        <Alert tone="success" live onDismiss={() => setStatus(null)}>
          {status}
        </Alert>
      )}

      <Card>
        <CardHeader title="Источники" description="Отдельные словари глоссария" />
        <CardContent>
          {sources.length === 0 ? (
            <p className="text-sm text-muted">Источников пока нет</p>
          ) : (
            <ul className="divide-y divide-border">
              {sources.map((source) => (
                <li key={source.name} className="flex flex-wrap items-center gap-3 py-2">
                  <Checkbox
                    label={<span className="font-medium">{source.name}</span>}
                    checked={source.enabled}
                    disabled={busy}
                    onChange={() => void toggleSource(source)}
                  />
                  <Badge tone="neutral">{source.kind}</Badge>
                  <span className="text-xs tabular-nums text-muted">{source.count} записей</span>
                  {source.path && (
                    <span className="truncate text-xs text-muted" title={source.path}>
                      {source.path}
                    </span>
                  )}
                  <Tooltip label="Удалить источник" align="right" className="ml-auto">
                    <IconButton
                      aria-label={`Удалить источник ${source.name}`}
                      size="sm"
                      className="text-danger"
                      disabled={busy}
                      onClick={() => void deleteSource(source)}
                    >
                      <Trash2 aria-hidden className="h-4 w-4" />
                    </IconButton>
                  </Tooltip>
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader title="Импорт файла" description=".txt / .csv" />
        <CardContent className="space-y-3">
          <div className="flex flex-wrap items-end gap-3">
            <Field label="Файл" htmlFor="glossary-import-file" className="min-w-48 flex-1">
              <FileInput
                id="glossary-import-file"
                accept=".txt,.csv,text/plain,text/csv"
                value={importFile}
                onChange={setImportFile}
              />
            </Field>
            <Field label="Имя источника" htmlFor="glossary-import-source">
              <Input
                id="glossary-import-source"
                value={importSource}
                onChange={(event) => setImportSource(event.target.value)}
                placeholder="Имя источника"
                className="w-44"
              />
            </Field>
            <Field label="Формат" htmlFor="glossary-import-kind">
              <Select
                id="glossary-import-kind"
                value={importKind}
                onChange={(event) => setImportKind(event.target.value)}
              >
                <option value="auto">авто</option>
                <option value="txt">txt</option>
                <option value="csv">csv</option>
              </Select>
            </Field>
            <Button variant="primary" loading={busy} onClick={() => void importFileSubmit()}>
              Импортировать
            </Button>
          </div>
          {report && (
            <p className="text-xs text-muted">
              «{report.source}» ({report.kind}): добавлено {report.added}, пропущено{' '}
              {report.skipped}, всего {report.total}
            </p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader title="Добавить запись" />
        <CardContent className="space-y-3">
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <Field label="Канон" htmlFor="glossary-draft-canonical" required>
              <Input
                id="glossary-draft-canonical"
                value={draft.canonical}
                onChange={(event) => setDraft({ ...draft, canonical: event.target.value })}
                placeholder="Правильное написание"
              />
            </Field>
            <Field label="Вариант (ошибка)" htmlFor="glossary-draft-variant">
              <Input
                id="glossary-draft-variant"
                value={draft.variant}
                onChange={(event) => setDraft({ ...draft, variant: event.target.value })}
                placeholder="Как распознаётся"
              />
            </Field>
            <Field label="Заметка" htmlFor="glossary-draft-note">
              <Input
                id="glossary-draft-note"
                value={draft.note}
                onChange={(event) => setDraft({ ...draft, note: event.target.value })}
                placeholder="Комментарий"
              />
            </Field>
            <Field label="Источник" htmlFor="glossary-draft-source">
              <Input
                id="glossary-draft-source"
                value={draft.source}
                onChange={(event) => setDraft({ ...draft, source: event.target.value })}
                placeholder="Встреча"
              />
            </Field>
          </div>
          <Button variant="primary" loading={busy} onClick={() => void addEntry()}>
            Добавить
          </Button>
        </CardContent>
      </Card>

      <Card>
        <CardHeader
          title="Записи"
          actions={
            <>
              <div className="relative min-w-40">
                <Search
                  aria-hidden
                  className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted"
                />
                <Input
                  aria-label="Поиск по записям"
                  value={search}
                  onChange={(event) => setSearch(event.target.value)}
                  placeholder="Поиск…"
                  className="pl-8"
                />
              </div>
              <Select
                aria-label="Фильтр по источнику"
                value={sourceFilter}
                onChange={(event) => {
                  setSourceFilter(event.target.value)
                  setOffset(0)
                }}
                className="w-40"
              >
                <option value="">Все источники</option>
                {sources.map((source) => (
                  <option key={source.name} value={source.name}>
                    {source.name}
                  </option>
                ))}
              </Select>
            </>
          }
        />
        <CardContent className="space-y-3">
          <div className="flex items-center justify-end gap-2 text-xs text-muted">
            <span className="tabular-nums">
              {total === 0 ? '0' : `${offset + 1}–${pageEnd}`} из {total}
            </span>
            <IconButton
              aria-label="Предыдущая страница"
              size="sm"
              disabled={!canPrev || loading}
              onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
            >
              <ChevronLeft aria-hidden className="h-4 w-4" />
            </IconButton>
            <IconButton
              aria-label="Следующая страница"
              size="sm"
              disabled={!canNext || loading}
              onClick={() => setOffset(offset + PAGE_SIZE)}
            >
              <ChevronRight aria-hidden className="h-4 w-4" />
            </IconButton>
          </div>

          {loading ? (
            <div className="flex items-center justify-center py-6">
              <Spinner size={20} label="Загрузка записей" />
            </div>
          ) : entries.length === 0 ? (
            <EmptyState title="Записей нет" description="Добавьте запись или импортируйте файл" />
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full border-collapse text-sm">
                <thead className="text-left text-xs uppercase text-muted">
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
                    <tr key={entry.id} className="border-t border-border">
                      <td className="px-2 py-1.5">
                        <Checkbox
                          checked={entry.enabled}
                          disabled={busy}
                          onChange={() => void toggleEntry(entry)}
                          aria-label={`Включить запись ${entry.canonical}`}
                        />
                      </td>
                      {edit?.id === entry.id ? (
                        <>
                          <td className="px-2 py-1.5">
                            <Input
                              aria-label="Канон"
                              value={edit.canonical}
                              onChange={(event) =>
                                setEdit({ ...edit, canonical: event.target.value })
                              }
                              className="w-32"
                            />
                          </td>
                          <td className="px-2 py-1.5">
                            <Input
                              aria-label="Вариант"
                              value={edit.variant}
                              onChange={(event) => setEdit({ ...edit, variant: event.target.value })}
                              className="w-32"
                            />
                          </td>
                          <td className="px-2 py-1.5">
                            <Input
                              aria-label="Заметка"
                              value={edit.note}
                              onChange={(event) => setEdit({ ...edit, note: event.target.value })}
                              className="w-40"
                            />
                          </td>
                        </>
                      ) : (
                        <>
                          <td className="px-2 py-1.5 font-medium text-text">{entry.canonical}</td>
                          <td className="px-2 py-1.5 text-muted">{entry.variant ?? '—'}</td>
                          <td className="px-2 py-1.5 text-muted">{entry.note ?? '—'}</td>
                        </>
                      )}
                      <td className="px-2 py-1.5 text-xs text-muted">{entry.source ?? '—'}</td>
                      <td className="whitespace-nowrap px-2 py-1.5 text-right">
                        {edit?.id === entry.id ? (
                          <span className="flex justify-end gap-1">
                            <Button variant="primary" size="sm" disabled={busy} onClick={saveEdit}>
                              ОК
                            </Button>
                            <Button variant="ghost" size="sm" onClick={() => setEdit(null)}>
                              Отмена
                            </Button>
                          </span>
                        ) : (
                          <span className="flex items-center justify-end gap-1">
                            <Button
                              variant="secondary"
                              size="sm"
                              icon={<Pencil aria-hidden className="h-3.5 w-3.5" />}
                              onClick={() =>
                                setEdit({
                                  id: entry.id,
                                  canonical: entry.canonical,
                                  variant: entry.variant ?? '',
                                  note: entry.note ?? '',
                                })
                              }
                            >
                              Правка
                            </Button>
                            <Tooltip label="Удалить запись" align="right">
                              <IconButton
                                aria-label={`Удалить запись ${entry.canonical}`}
                                size="sm"
                                className="text-danger"
                                disabled={busy}
                                onClick={() => void deleteEntry(entry)}
                              >
                                <Trash2 aria-hidden className="h-4 w-4" />
                              </IconButton>
                            </Tooltip>
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

export default GlossaryPanel
