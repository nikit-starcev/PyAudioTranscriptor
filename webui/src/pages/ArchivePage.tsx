import { useCallback, useEffect, useMemo, useState } from 'react'
import { Archive, ExternalLink, RotateCcw, Search, User } from 'lucide-react'

import {
  errorMessage,
  fetchArchive,
  formatClock,
  patchArchiveRecord,
  searchArchive,
  type ArchiveRecord,
  type ArchiveSearchResponse,
} from '../api'
import { navigateTo } from '../app/routes'
import {
  Alert,
  Badge,
  Button,
  Card,
  CardContent,
  CardHeader,
  EmptyState,
  Field,
  Input,
  Select,
  Spinner,
  Textarea,
  cn,
} from '../components/ui'

const DATE_FORMAT = new Intl.DateTimeFormat('ru-RU', {
  day: 'numeric',
  month: 'long',
  year: 'numeric',
})

/** Человекочитаемая дата встречи (``ГГГГ-ММ-ДД`` → «4 июня 2026 г.»). */
function formatArchiveDate(value: string | null): string {
  if (!value) return 'Без даты'
  const parsed = new Date(`${value}T00:00:00`)
  if (Number.isNaN(parsed.getTime())) return value
  return DATE_FORMAT.format(parsed)
}

/** Подпись записи: заголовок, иначе имя файла, иначе id задачи. */
function recordLabel(record: ArchiveRecord): string {
  return record.title?.trim() || record.name || record.job_id
}

function ArchivePage() {
  const [records, setRecords] = useState<ArchiveRecord[]>([])
  const [speakers, setSpeakers] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)

  const [title, setTitle] = useState('')
  const [meetingDate, setMeetingDate] = useState('')
  const [participants, setParticipants] = useState('')
  const [notes, setNotes] = useState('')

  const [query, setQuery] = useState('')
  const [speakerFilter, setSpeakerFilter] = useState('')
  const [dateFrom, setDateFrom] = useState('')
  const [dateTo, setDateTo] = useState('')
  const [searching, setSearching] = useState(false)
  const [results, setResults] = useState<ArchiveSearchResponse | null>(null)
  const [searchError, setSearchError] = useState<string | null>(null)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await fetchArchive()
      setRecords(data.records)
      setSpeakers(data.speakers)
      setSelectedId((current) =>
        current && data.records.some((record) => record.job_id === current)
          ? current
          : (data.records[0]?.job_id ?? null),
      )
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  const selected = useMemo(
    () => records.find((record) => record.job_id === selectedId) ?? null,
    [records, selectedId],
  )

  useEffect(() => {
    if (!selected) return
    setTitle(selected.title ?? '')
    setMeetingDate(selected.meeting_date ?? '')
    setParticipants(selected.participants.join(', '))
    setNotes(selected.notes ?? '')
  }, [selected])

  useEffect(() => {
    setNotice(null)
  }, [selectedId])

  const groups = useMemo(() => {
    const map = new Map<string, ArchiveRecord[]>()
    for (const record of records) {
      const key = record.meeting_date ?? ''
      const list = map.get(key)
      if (list) list.push(record)
      else map.set(key, [record])
    }
    return [...map.entries()]
  }, [records])

  const save = useCallback(async () => {
    if (!selected) return
    setSaving(true)
    setError(null)
    setNotice(null)
    try {
      const updated = await patchArchiveRecord(selected.job_id, {
        title,
        meeting_date: meetingDate,
        participants: participants
          .split(',')
          .map((item) => item.trim())
          .filter(Boolean),
        notes,
      })
      setRecords((current) =>
        current.map((record) =>
          record.job_id === updated.job_id ? { ...record, ...updated } : record,
        ),
      )
      setNotice('Запись сохранена')
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setSaving(false)
    }
  }, [selected, title, meetingDate, participants, notes])

  const resetSearch = useCallback(() => {
    setQuery('')
    setSpeakerFilter('')
    setDateFrom('')
    setDateTo('')
    setResults(null)
    setSearchError(null)
  }, [])

  const runSearch = useCallback(async () => {
    const text = query.trim()
    if (!text) {
      setResults(null)
      return
    }
    setSearching(true)
    setSearchError(null)
    try {
      const data = await searchArchive({
        q: text,
        speaker: speakerFilter || undefined,
        date_from: dateFrom || undefined,
        date_to: dateTo || undefined,
      })
      setResults(data)
      setSpeakers(data.speakers)
    } catch (cause) {
      setSearchError(errorMessage(cause))
    } finally {
      setSearching(false)
    }
  }, [query, speakerFilter, dateFrom, dateTo])

  const openJob = (jobId: string, entry?: number) => {
    const suffix = entry != null ? `?entry=${entry}` : ''
    navigateTo(`/jobs/${jobId}${suffix}`)
  }

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-lg font-semibold">Архив</h1>
        <p className="text-sm text-muted">
          Стенограммы по датам и встречам, заметки и полнотекстовый поиск
        </p>
      </div>

      {error && (
        <Alert tone="danger" live onDismiss={() => setError(null)}>
          {error}
        </Alert>
      )}
      {notice && (
        <Alert tone="success" live onDismiss={() => setNotice(null)}>
          {notice}
        </Alert>
      )}

      <Card>
        <CardHeader
          title="Поиск по архиву"
          description="Текст стенограмм, говорящий и диапазон дат"
          actions={
            <Button
              variant="ghost"
              size="sm"
              icon={<RotateCcw aria-hidden className="h-4 w-4" />}
              onClick={resetSearch}
            >
              Сбросить
            </Button>
          }
        />
        <CardContent>
          <form
            className="grid gap-3 md:grid-cols-[1.5fr_1fr_1fr_1fr_auto]"
            onSubmit={(event) => {
              event.preventDefault()
              void runSearch()
            }}
          >
            <Field label="Текст" htmlFor="archive-q" className="md:col-span-1">
              <div className="relative">
                <Search
                  aria-hidden
                  className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted"
                />
                <Input
                  id="archive-q"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="Слова из стенограммы…"
                  className="pl-8"
                />
              </div>
            </Field>
            <Field label="Говорящий" htmlFor="archive-speaker">
              <Select
                id="archive-speaker"
                value={speakerFilter}
                onChange={(event) => setSpeakerFilter(event.target.value)}
              >
                <option value="">— любой —</option>
                {speakers.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </Select>
            </Field>
            <Field label="Дата с" htmlFor="archive-from">
              <Input
                id="archive-from"
                type="date"
                value={dateFrom}
                onChange={(event) => setDateFrom(event.target.value)}
              />
            </Field>
            <Field label="Дата по" htmlFor="archive-to">
              <Input
                id="archive-to"
                type="date"
                value={dateTo}
                onChange={(event) => setDateTo(event.target.value)}
              />
            </Field>
            <div className="flex items-end">
              <Button type="submit" variant="primary" loading={searching} fullWidth>
                Найти
              </Button>
            </div>
          </form>
          {searchError && (
            <Alert tone="danger" className="mt-3" onDismiss={() => setSearchError(null)}>
              {searchError}
            </Alert>
          )}
        </CardContent>
      </Card>

      <div className="grid items-start gap-5 lg:grid-cols-[minmax(0,1fr)_24rem]">
        <div className="space-y-4">
          {results ? (
            <SearchResults
              results={results}
              onOpen={(jobId, entry) => openJob(jobId, entry)}
            />
          ) : (
            <RecordList
              loading={loading}
              groups={groups}
              selectedId={selectedId}
              onSelect={setSelectedId}
            />
          )}
        </div>

        <div className="lg:sticky lg:top-20">
          <RecordEditor
            record={selected}
            loading={loading}
            saving={saving}
            title={title}
            meetingDate={meetingDate}
            participants={participants}
            notes={notes}
            onTitle={setTitle}
            onMeetingDate={setMeetingDate}
            onParticipants={setParticipants}
            onNotes={setNotes}
            onSave={() => void save()}
            onOpen={() => selected && openJob(selected.job_id)}
          />
        </div>
      </div>
    </div>
  )
}

function RecordList({
  loading,
  groups,
  selectedId,
  onSelect,
}: {
  loading: boolean
  groups: [string, ArchiveRecord[]][]
  selectedId: string | null
  onSelect: (jobId: string) => void
}) {
  if (loading) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center py-10">
          <Spinner label="Загрузка архива" />
        </CardContent>
      </Card>
    )
  }
  if (groups.length === 0) {
    return (
      <EmptyState
        icon={<Archive aria-hidden className="h-6 w-6" />}
        title="Архив пуст"
        description="Стенограммы появятся здесь после завершения обработки задач"
      />
    )
  }
  return (
    <div className="space-y-4">
      {groups.map(([day, items]) => (
        <section key={day || 'none'} className="space-y-2">
          <h2 className="text-xs font-semibold uppercase tracking-wide text-muted">
            {formatArchiveDate(day || null)}
          </h2>
          <div className="space-y-2">
            {items.map((record) => (
              <RecordCard
                key={record.job_id}
                record={record}
                active={record.job_id === selectedId}
                onSelect={() => onSelect(record.job_id)}
              />
            ))}
          </div>
        </section>
      ))}
    </div>
  )
}

function RecordCard({
  record,
  active,
  onSelect,
}: {
  record: ArchiveRecord
  active: boolean
  onSelect: () => void
}) {
  return (
    <button
      type="button"
      onClick={onSelect}
      aria-current={active ? 'true' : undefined}
      className={cn(
        'w-full rounded-lg border bg-surface px-4 py-3 text-left shadow-sm transition-colors',
        'focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring',
        active ? 'border-primary bg-primary-soft' : 'border-border hover:bg-surface-2',
      )}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="truncate text-sm font-medium text-text">{recordLabel(record)}</p>
          <p className="mt-0.5 truncate text-xs text-muted">
            {record.name ?? record.job_id}
            {record.duration != null ? ` · ${formatClock(record.duration)}` : ''}
          </p>
        </div>
        {active && <Badge tone="primary">выбрана</Badge>}
      </div>
      {record.participants.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1">
          {record.participants.map((person) => (
            <Badge key={person} tone="neutral" icon={<User aria-hidden className="h-3 w-3" />}>
              {person}
            </Badge>
          ))}
        </div>
      )}
      {record.notes && (
        <p className="mt-2 line-clamp-2 text-xs text-muted">{record.notes}</p>
      )}
    </button>
  )
}

function RecordEditor({
  record,
  loading,
  saving,
  title,
  meetingDate,
  participants,
  notes,
  onTitle,
  onMeetingDate,
  onParticipants,
  onNotes,
  onSave,
  onOpen,
}: {
  record: ArchiveRecord | null
  loading: boolean
  saving: boolean
  title: string
  meetingDate: string
  participants: string
  notes: string
  onTitle: (value: string) => void
  onMeetingDate: (value: string) => void
  onParticipants: (value: string) => void
  onNotes: (value: string) => void
  onSave: () => void
  onOpen: () => void
}) {
  if (loading) return null
  if (!record) {
    return (
      <Card>
        <CardContent>
          <p className="py-6 text-center text-sm text-muted">
            Выберите запись, чтобы изменить дату, участников и заметки
          </p>
        </CardContent>
      </Card>
    )
  }
  return (
    <Card>
      <CardHeader
        title="Карточка записи"
        description={record.name ?? record.job_id}
        actions={
          <Button
            variant="secondary"
            size="sm"
            icon={<ExternalLink aria-hidden className="h-4 w-4" />}
            onClick={onOpen}
          >
            Открыть
          </Button>
        }
      />
      <CardContent className="space-y-3">
        <Field label="Заголовок / встреча" htmlFor="archive-title">
          <Input
            id="archive-title"
            value={title}
            onChange={(event) => onTitle(event.target.value)}
            placeholder="Например, планёрка по поставке"
          />
        </Field>
        <Field label="Дата" htmlFor="archive-date">
          <Input
            id="archive-date"
            type="date"
            value={meetingDate}
            onChange={(event) => onMeetingDate(event.target.value)}
          />
        </Field>
        <Field
          label="Участники / теги"
          htmlFor="archive-participants"
          hint="Через запятую"
        >
          <Input
            id="archive-participants"
            value={participants}
            onChange={(event) => onParticipants(event.target.value)}
            placeholder="Иван, Мария, поставка"
          />
        </Field>
        <Field label="Заметки" htmlFor="archive-notes">
          <Textarea
            id="archive-notes"
            value={notes}
            onChange={(event) => onNotes(event.target.value)}
            rows={5}
            placeholder="Свободные заметки по встрече…"
          />
        </Field>
        <div className="flex items-center justify-end">
          <Button variant="primary" loading={saving} onClick={onSave}>
            Сохранить
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}

function SearchResults({
  results,
  onOpen,
}: {
  results: ArchiveSearchResponse
  onOpen: (jobId: string, entry: number) => void
}) {
  if (results.hits.length === 0) {
    return (
      <EmptyState
        icon={<Search aria-hidden className="h-6 w-6" />}
        title="Ничего не найдено"
        description={`По запросу «${results.query}» совпадений нет`}
      />
    )
  }
  return (
    <Card>
      <CardHeader
        title="Результаты поиска"
        description={`Найдено реплик: ${results.total}`}
      />
      <CardContent className="space-y-2">
        {results.hits.map((hit) => (
          <button
            key={`${hit.job_id}-${hit.entry_index}`}
            type="button"
            onClick={() => onOpen(hit.job_id, hit.entry_index)}
            className={cn(
              'w-full rounded-md border border-border px-3 py-2 text-left transition-colors',
              'hover:bg-surface-2 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring',
            )}
          >
            <div className="flex flex-wrap items-center gap-2 text-xs text-muted">
              <span className="truncate font-medium text-text">
                {hit.title?.trim() || hit.name || hit.job_id}
              </span>
              {hit.meeting_date && <span>· {formatArchiveDate(hit.meeting_date)}</span>}
              {hit.speaker && (
                <Badge tone="info" icon={<User aria-hidden className="h-3 w-3" />}>
                  {hit.speaker}
                </Badge>
              )}
            </div>
            <p className="mt-1 text-sm text-text">{hit.snippet}</p>
          </button>
        ))}
      </CardContent>
    </Card>
  )
}

export default ArchivePage
