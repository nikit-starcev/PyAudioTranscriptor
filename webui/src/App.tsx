import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

type ConfigInfo = {
  input_dir: string
  output_dir: string
  export_formats: string[]
  llm_enabled: boolean
  glossary_enabled: boolean
  voices_dir: string
}

type FileItem = {
  name: string
  path: string
  size: number
  duration: number | null
}

type Job = {
  id: string
  name: string
  source_path: string
  status: string
  stage: string | null
  fraction: number | null
  created_at: string
  started_at: string | null
  finished_at: string | null
  language: string | null
  duration: number | null
  error: string | null
}

type Summary = {
  language: string | null
  duration: number | null
  entries: number
  speakers: number
  samples: number
}

type JobDetails = Job & { summary: Summary | null }

type SpeakerInfo = { id: string; display_name: string; has_sample: boolean }

type Entry = {
  start: number
  end: number
  speaker_id: string | null
  text: string
  low_confidence: boolean
  overlap: boolean
}

type Mark = { key: string; symbol: string; label: string }

type TranscriptResult = {
  language: string | null
  duration: number
  speakers: SpeakerInfo[]
  entries: Entry[]
  marks: Mark[]
  summary: string | null
  samples: Record<string, string>
}

type JobEvent = {
  stage: string
  fraction: number | null
  message: string
  status: string
}

const STAGES: { key: string; label: string }[] = [
  { key: 'denoise', label: 'Шумоподавление' },
  { key: 'asr', label: 'Распознавание речи' },
  { key: 'diarization', label: 'Определение говорящих' },
  { key: 'merge', label: 'Объединение сегментов' },
  { key: 'clean', label: 'Очистка артефактов' },
  { key: 'correction', label: 'Автоисправление' },
  { key: 'llm', label: 'LLM-постобработка' },
  { key: 'export', label: 'Экспорт' },
]

const STATUS_LABELS: Record<string, string> = {
  queued: 'В очереди',
  running: 'Обработка',
  done: 'Готово',
  error: 'Ошибка',
  cancelled: 'Отменено',
}

const STATUS_STYLES: Record<string, string> = {
  queued: 'bg-slate-100 text-slate-600',
  running: 'bg-blue-100 text-blue-700',
  done: 'bg-emerald-100 text-emerald-700',
  error: 'bg-red-100 text-red-700',
  cancelled: 'bg-amber-100 text-amber-700',
}

function isTerminal(status: string): boolean {
  return status === 'done' || status === 'error' || status === 'cancelled'
}

async function api<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init)
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`
    try {
      const body = await response.json()
      if (body && typeof body.detail === 'string') detail = body.detail
    } catch {
      // тело не JSON — оставляем статус
    }
    throw new Error(detail)
  }
  return (await response.json()) as T
}

function formatTime(seconds: number): string {
  const minutes = Math.floor(seconds / 60)
  const rest = seconds - minutes * 60
  return `${minutes}:${rest.toFixed(1).padStart(4, '0')}`
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} Б`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} КБ`
  return `${(bytes / (1024 * 1024)).toFixed(1)} МБ`
}

function formatDuration(seconds: number | null): string {
  return seconds == null ? '—' : formatTime(seconds)
}

function speakerName(speakers: SpeakerInfo[], id: string | null): string {
  if (id == null) return '—'
  return speakers.find((speaker) => speaker.id === id)?.display_name ?? id
}

function App() {
  const [version, setVersion] = useState<string>('')
  const [config, setConfig] = useState<ConfigInfo | null>(null)
  const [files, setFiles] = useState<FileItem[]>([])
  const [jobs, setJobs] = useState<Job[]>([])
  const [activeJobId, setActiveJobId] = useState<string | null>(null)
  const [progress, setProgress] = useState<JobEvent | null>(null)
  const [result, setResult] = useState<TranscriptResult | null>(null)
  const [query, setQuery] = useState('')
  const [error, setError] = useState<string | null>(null)
  const fileInput = useRef<HTMLInputElement>(null)

  const refreshJobs = useCallback(async () => {
    try {
      setJobs(await api<Job[]>('/api/jobs'))
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
    }
  }, [])

  const refreshFiles = useCallback(async () => {
    try {
      setFiles(await api<FileItem[]>('/api/files'))
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
    }
  }, [])

  const loadResult = useCallback(async (jobId: string) => {
    try {
      setResult(await api<TranscriptResult>(`/api/jobs/${jobId}/result`))
    } catch {
      setResult(null)
    }
  }, [])

  useEffect(() => {
    void (async () => {
      try {
        const health = await api<{ status: string; version: string }>('/api/health')
        setVersion(health.version)
        setConfig(await api<ConfigInfo>('/api/config'))
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : String(cause))
      }
    })()
    void refreshFiles()
    void refreshJobs()
  }, [refreshFiles, refreshJobs])

  useEffect(() => {
    if (!activeJobId) return
    const source = new EventSource(`/api/jobs/${activeJobId}/events`)
    source.onmessage = (message) => {
      const event = JSON.parse(message.data) as JobEvent
      setProgress(event)
      if (isTerminal(event.status)) {
        source.close()
        void refreshJobs()
        if (event.status === 'done') void loadResult(activeJobId)
      }
    }
    source.onerror = () => source.close()
    return () => source.close()
  }, [activeJobId, refreshJobs, loadResult])

  const enqueue = useCallback(
    async (path: string) => {
      setError(null)
      try {
        const job = await api<Job>('/api/jobs', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path }),
        })
        setActiveJobId(job.id)
        setProgress({ stage: 'queued', fraction: 0, message: 'В очереди', status: 'queued' })
        setResult(null)
        await refreshJobs()
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : String(cause))
      }
    },
    [refreshJobs],
  )

  const runJob = useCallback(
    async (jobId: string) => {
      setError(null)
      try {
        await api<Job>(`/api/jobs/${jobId}/run`, { method: 'POST' })
        setActiveJobId(jobId)
        setResult(null)
        await refreshJobs()
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : String(cause))
      }
    },
    [refreshJobs],
  )

  const openJob = useCallback(
    async (jobId: string) => {
      setError(null)
      setActiveJobId(jobId)
      setResult(null)
      setProgress(null)
      try {
        const details = await api<JobDetails>(`/api/jobs/${jobId}`)
        setProgress({
          stage: details.stage ?? details.status,
          fraction: details.fraction,
          message: '',
          status: details.status,
        })
        if (details.status === 'done') await loadResult(jobId)
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : String(cause))
      }
    },
    [loadResult],
  )

  const deleteJob = useCallback(
    async (jobId: string) => {
      setError(null)
      try {
        await api<{ deleted: string }>(`/api/jobs/${jobId}`, { method: 'DELETE' })
        if (activeJobId === jobId) {
          setActiveJobId(null)
          setResult(null)
          setProgress(null)
        }
        await refreshJobs()
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : String(cause))
      }
    },
    [activeJobId, refreshJobs],
  )

  const upload = useCallback(
    async (file: File) => {
      setError(null)
      const body = new FormData()
      body.append('file', file)
      try {
        await api<FileItem>('/api/files/upload', { method: 'POST', body })
        await refreshFiles()
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : String(cause))
      }
    },
    [refreshFiles],
  )

  const stageIndex = useMemo(() => {
    if (!progress) return -1
    return STAGES.findIndex((stage) => stage.key === progress.stage)
  }, [progress])

  const filteredEntries = useMemo(() => {
    if (!result) return []
    const needle = query.trim().toLowerCase()
    if (!needle) return result.entries
    return result.entries.filter((entry) => entry.text.toLowerCase().includes(needle))
  }, [result, query])

  const activeJob = jobs.find((job) => job.id === activeJobId) ?? null

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl items-baseline gap-3 px-6 py-4">
          <h1 className="text-xl font-semibold">AudioTranscriber</h1>
          <span className="text-sm text-slate-400">веб-интерфейс · этап 1</span>
          {version && <span className="ml-auto text-xs text-slate-400">v{version}</span>}
        </div>
      </header>

      <main className="mx-auto max-w-6xl space-y-6 px-6 py-6">
        {error && (
          <div className="rounded-md border border-red-200 bg-red-50 px-4 py-2 text-sm text-red-700">
            {error}
          </div>
        )}

        {config && (
          <p className="text-xs text-slate-500">
            Файлы: <code>{config.input_dir}</code> · Результаты: <code>{config.output_dir}</code> ·
            Экспорт: {config.export_formats.join(', ')}
            {config.llm_enabled ? ' · LLM вкл.' : ''}
            {config.glossary_enabled ? ' · глоссарий вкл.' : ''}
          </p>
        )}

        <div className="grid gap-6 lg:grid-cols-2">
          <section className="rounded-lg border border-slate-200 bg-white p-4">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="font-medium">Файлы</h2>
              <div>
                <input
                  ref={fileInput}
                  type="file"
                  className="hidden"
                  onChange={(event) => {
                    const file = event.target.files?.[0]
                    if (file) void upload(file)
                    event.target.value = ''
                  }}
                />
                <button
                  onClick={() => fileInput.current?.click()}
                  className="rounded-md bg-slate-800 px-3 py-1.5 text-sm text-white hover:bg-slate-700"
                >
                  Загрузить файл
                </button>
              </div>
            </div>
            {files.length === 0 ? (
              <p className="py-6 text-center text-sm text-slate-400">Файлов пока нет</p>
            ) : (
              <ul className="divide-y divide-slate-100">
                {files.map((file) => (
                  <li key={file.path} className="flex items-center gap-3 py-2">
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm">{file.name}</p>
                      <p className="text-xs text-slate-400">
                        {formatSize(file.size)} · {formatDuration(file.duration)}
                      </p>
                    </div>
                    <button
                      onClick={() => void enqueue(file.name)}
                      className="rounded-md border border-slate-300 px-3 py-1 text-xs hover:bg-slate-100"
                    >
                      В очередь
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </section>

          <section className="rounded-lg border border-slate-200 bg-white p-4">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="font-medium">Задачи</h2>
              <button
                onClick={() => void refreshJobs()}
                className="rounded-md border border-slate-300 px-3 py-1 text-xs hover:bg-slate-100"
              >
                Обновить
              </button>
            </div>
            {jobs.length === 0 ? (
              <p className="py-6 text-center text-sm text-slate-400">Задач пока нет</p>
            ) : (
              <ul className="divide-y divide-slate-100">
                {jobs.map((job) => (
                  <li key={job.id} className="flex items-center gap-3 py-2">
                    <button
                      onClick={() => void openJob(job.id)}
                      className="min-w-0 flex-1 text-left"
                    >
                      <p className="truncate text-sm">{job.name}</p>
                      <p className="text-xs text-slate-400">
                        {job.stage ? `${job.stage} · ` : ''}
                        {job.fraction != null ? `${Math.round(job.fraction * 100)}%` : '—'}
                      </p>
                    </button>
                    <span
                      className={`rounded-full px-2 py-0.5 text-xs ${
                        STATUS_STYLES[job.status] ?? 'bg-slate-100 text-slate-600'
                      }`}
                    >
                      {STATUS_LABELS[job.status] ?? job.status}
                    </span>
                    {(job.status === 'queued' || job.status === 'done' || job.status === 'error') && (
                      <button
                        onClick={() => void runJob(job.id)}
                        className="rounded-md bg-emerald-600 px-3 py-1 text-xs text-white hover:bg-emerald-500"
                      >
                        Запустить
                      </button>
                    )}
                    {job.status !== 'running' && (
                      <button
                        onClick={() => void deleteJob(job.id)}
                        className="rounded-md border border-slate-300 px-2 py-1 text-xs text-slate-500 hover:bg-red-50 hover:text-red-600"
                      >
                        Удалить
                      </button>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </section>
        </div>

        {activeJobId && (
          <section className="rounded-lg border border-slate-200 bg-white p-4">
            <h2 className="mb-3 font-medium">
              Прогресс{activeJob ? ` · ${activeJob.name}` : ''}
            </h2>
            <div className="mb-2 h-2 w-full overflow-hidden rounded-full bg-slate-100">
              <div
                className={`h-full rounded-full bg-blue-500 transition-all ${
                  progress?.fraction == null ? 'animate-pulse' : ''
                }`}
                style={{ width: `${Math.round((progress?.fraction ?? 0) * 100)}%` }}
              />
            </div>
            <p className="mb-3 text-sm text-slate-600">
              {progress
                ? STATUS_LABELS[progress.status] ?? progress.status
                : 'Ожидание...'}
              {progress?.message ? ` — ${progress.message}` : ''}
            </p>
            <ol className="grid grid-cols-2 gap-x-6 gap-y-1 text-sm sm:grid-cols-4">
              {STAGES.map((stage, index) => {
                const state =
                  stageIndex > index
                    ? 'done'
                    : stageIndex === index
                      ? 'current'
                      : 'pending'
                return (
                  <li key={stage.key} className="flex items-center gap-2">
                    <span
                      className={
                        state === 'done'
                          ? 'text-emerald-600'
                          : state === 'current'
                            ? 'text-blue-600'
                            : 'text-slate-300'
                      }
                    >
                      {state === 'done' ? '✓' : state === 'current' ? '●' : '·'}
                    </span>
                    <span className={state === 'pending' ? 'text-slate-400' : ''}>
                      {stage.label}
                    </span>
                  </li>
                )
              })}
            </ol>
          </section>
        )}

        {result && (
          <section className="space-y-4 rounded-lg border border-slate-200 bg-white p-4">
            <div className="flex flex-wrap items-center gap-3">
              <h2 className="font-medium">Стенограмма</h2>
              <span className="text-xs text-slate-400">
                {result.entries.length} реплик · {result.speakers.length} говорящих ·{' '}
                {formatDuration(result.duration)} · язык {result.language ?? '—'}
              </span>
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Поиск по тексту..."
                className="ml-auto w-64 rounded-md border border-slate-300 px-3 py-1.5 text-sm focus:border-blue-400 focus:outline-none"
              />
            </div>

            <div className="flex flex-wrap items-center gap-4 text-xs text-slate-500">
              {result.marks.map((mark) => (
                <span key={mark.key}>
                  <span className="mr-1 text-base">{mark.symbol}</span>
                  {mark.label}
                </span>
              ))}
              <span>{filteredEntries.length} из {result.entries.length}</span>
            </div>

            {result.speakers.length > 0 && (
              <div className="flex flex-wrap gap-3">
                {result.speakers.map((speaker) => (
                  <div
                    key={speaker.id}
                    className="flex items-center gap-2 rounded-md border border-slate-200 px-3 py-1.5"
                  >
                    <span className="text-sm">{speaker.display_name}</span>
                    {speaker.has_sample ? (
                      <audio
                        controls
                        preload="none"
                        className="h-8"
                        src={`/api/jobs/${activeJobId}/samples/${speaker.id}`}
                      />
                    ) : (
                      <span className="text-xs text-slate-400">нет образца</span>
                    )}
                  </div>
                ))}
              </div>
            )}

            <div className="max-h-[28rem] overflow-auto rounded-md border border-slate-200">
              <table className="w-full border-collapse text-sm">
                <thead className="sticky top-0 bg-slate-100 text-left text-xs uppercase text-slate-500">
                  <tr>
                    <th className="px-3 py-2 font-medium">Время</th>
                    <th className="px-3 py-2 font-medium">Говорящий</th>
                    <th className="px-3 py-2 font-medium">Метки</th>
                    <th className="px-3 py-2 font-medium">Текст</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredEntries.map((entry, index) => (
                    <tr key={`${entry.start}-${index}`} className="border-t border-slate-100">
                      <td className="whitespace-nowrap px-3 py-1.5 font-mono text-xs text-slate-500">
                        {formatTime(entry.start)}
                      </td>
                      <td className="whitespace-nowrap px-3 py-1.5">
                        {speakerName(result.speakers, entry.speaker_id)}
                      </td>
                      <td className="whitespace-nowrap px-3 py-1.5 text-base">
                        {entry.low_confidence && <span title="низкая уверенность">⚠</span>}
                        {entry.overlap && <span title="наложение речи">⇄</span>}
                      </td>
                      <td className="px-3 py-1.5">{entry.text}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {filteredEntries.length === 0 && (
                <p className="py-6 text-center text-sm text-slate-400">Ничего не найдено</p>
              )}
            </div>
          </section>
        )}
      </main>
    </div>
  )
}

export default App
