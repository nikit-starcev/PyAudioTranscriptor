import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import {
  api,
  errorMessage,
  formatDuration,
  formatSize,
  formatTime,
  isTerminal,
  speakerName,
  type ApplyNamesResponse,
  type ConfigInfo,
  type FileItem,
  type Job,
  type JobDetails,
  type JobEvent,
  type ProtocolResponse,
  type SampleMeta,
  type TranscriptResult,
  type VoiceInfo,
  type WebSettings,
} from './api'
import GlossaryModal from './components/GlossaryModal'
import SettingsModal from './components/SettingsModal'
import SpeakersPanel from './components/SpeakersPanel'
import VoicesModal from './components/VoicesModal'

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

function App() {
  const [version, setVersion] = useState<string>('')
  const [config, setConfig] = useState<ConfigInfo | null>(null)
  const [files, setFiles] = useState<FileItem[]>([])
  const [jobs, setJobs] = useState<Job[]>([])
  const [activeJobId, setActiveJobId] = useState<string | null>(null)
  const [progress, setProgress] = useState<JobEvent | null>(null)
  const [result, setResult] = useState<TranscriptResult | null>(null)
  const [samplesMeta, setSamplesMeta] = useState<Record<string, SampleMeta>>({})
  const [voicesOpen, setVoicesOpen] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [glossaryOpen, setGlossaryOpen] = useState(false)
  const [protocol, setProtocol] = useState<ProtocolResponse | null>(null)
  const [protocolBusy, setProtocolBusy] = useState(false)
  const [protocolError, setProtocolError] = useState<string | null>(null)
  const [summary, setSummary] = useState<string | null>(null)
  const [query, setQuery] = useState('')
  const [error, setError] = useState<string | null>(null)
  const fileInput = useRef<HTMLInputElement>(null)

  const refreshJobs = useCallback(async () => {
    try {
      setJobs(await api<Job[]>('/api/jobs'))
    } catch (cause) {
      setError(errorMessage(cause))
    }
  }, [])

  const refreshFiles = useCallback(async () => {
    try {
      setFiles(await api<FileItem[]>('/api/files'))
    } catch (cause) {
      setError(errorMessage(cause))
    }
  }, [])

  const refreshSamples = useCallback(async (jobId: string) => {
    try {
      const items = await api<SampleMeta[]>(`/api/jobs/${jobId}/samples`)
      const map: Record<string, SampleMeta> = {}
      for (const item of items) map[item.speaker_id] = item
      setSamplesMeta(map)
    } catch {
      setSamplesMeta({})
    }
  }, [])

  const loadResult = useCallback(
    async (jobId: string) => {
      try {
        const loaded = await api<TranscriptResult>(`/api/jobs/${jobId}/result`)
        setResult(loaded)
        setSummary(loaded.summary)
        setProtocol(null)
        setProtocolError(null)
        await refreshSamples(jobId)
      } catch {
        setResult(null)
        setSummary(null)
        setSamplesMeta({})
      }
    },
    [refreshSamples],
  )

  useEffect(() => {
    void (async () => {
      try {
        const health = await api<{ status: string; version: string }>('/api/health')
        setVersion(health.version)
        setConfig(await api<ConfigInfo>('/api/config'))
      } catch (cause) {
        setError(errorMessage(cause))
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
        setSummary(null)
        setProtocol(null)
        setSamplesMeta({})
        await refreshJobs()
      } catch (cause) {
        setError(errorMessage(cause))
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
        setSummary(null)
        setProtocol(null)
        setSamplesMeta({})
        await refreshJobs()
      } catch (cause) {
        setError(errorMessage(cause))
      }
    },
    [refreshJobs],
  )

  const openJob = useCallback(
    async (jobId: string) => {
      setError(null)
      setActiveJobId(jobId)
      setResult(null)
      setSummary(null)
      setProtocol(null)
      setSamplesMeta({})
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
        setError(errorMessage(cause))
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
          setSummary(null)
          setProtocol(null)
          setSamplesMeta({})
          setProgress(null)
        }
        await refreshJobs()
      } catch (cause) {
        setError(errorMessage(cause))
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
        setError(errorMessage(cause))
      }
    },
    [refreshFiles],
  )

  const patchSpeakers = useCallback(
    async (jobId: string, body: Record<string, unknown>) => {
      const updated = await api<TranscriptResult>(`/api/jobs/${jobId}/speakers`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      setResult(updated)
      await refreshSamples(jobId)
    },
    [refreshSamples],
  )

  const applyNames = useCallback(
    async (jobId: string): Promise<ApplyNamesResponse> => {
      const response = await api<ApplyNamesResponse>(`/api/jobs/${jobId}/apply-names`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      })
      setResult(response.result)
      await refreshSamples(jobId)
      return response
    },
    [refreshSamples],
  )

  const saveToLibrary = useCallback(
    async (jobId: string, speakerId: string, name: string): Promise<VoiceInfo> => {
      return api<VoiceInfo>(`/api/jobs/${jobId}/speakers/${speakerId}/to-library`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
      })
    },
    [],
  )

  const generateProtocol = useCallback(async (jobId: string) => {
    setProtocolBusy(true)
    setProtocolError(null)
    try {
      const response = await api<ProtocolResponse>(`/api/jobs/${jobId}/protocol`, {
        method: 'POST',
      })
      setProtocol(response)
      setSummary(response.summary)
    } catch (cause) {
      setProtocolError(errorMessage(cause))
    } finally {
      setProtocolBusy(false)
    }
  }, [])

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
          <span className="text-sm text-slate-400">веб-интерфейс · этап 3</span>
          <div className="ml-auto flex items-center gap-2">
            <button
              onClick={() => setGlossaryOpen(true)}
              className="rounded-md border border-slate-300 px-3 py-1.5 text-sm hover:bg-slate-100"
            >
              Глоссарий
            </button>
            <button
              onClick={() => setSettingsOpen(true)}
              className="rounded-md border border-slate-300 px-3 py-1.5 text-sm hover:bg-slate-100"
            >
              Настройки
            </button>
          </div>
          {version && <span className="text-xs text-slate-400">v{version}</span>}
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

        {result && activeJobId && (
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

            <SpeakersPanel
              jobId={activeJobId}
              result={result}
              sampleMeta={samplesMeta}
              onRename={(speakerId, name) =>
                patchSpeakers(activeJobId, { renames: { [speakerId]: name } })
              }
              onMerge={(source, target) =>
                patchSpeakers(activeJobId, { merges: [{ source, target }] })
              }
              onToLibrary={(speakerId, name) => saveToLibrary(activeJobId, speakerId, name)}
              onApplyNames={() => applyNames(activeJobId)}
              onOpenVoices={() => setVoicesOpen(true)}
            />

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

            <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
              <div className="flex flex-wrap items-center gap-3">
                <button
                  onClick={() => void generateProtocol(activeJobId)}
                  disabled={protocolBusy}
                  className="rounded-md bg-slate-800 px-3 py-1.5 text-sm text-white hover:bg-slate-700 disabled:opacity-40"
                >
                  {protocolBusy ? 'Формирование протокола…' : 'Сформировать протокол'}
                </button>
                {protocolBusy && (
                  <span className="animate-pulse text-xs text-slate-500">
                    Считается резюме и экспорт — это может занять время
                  </span>
                )}
                {protocol && (
                  <span className="flex items-center gap-2 text-sm">
                    <a
                      className="rounded-md border border-slate-300 bg-white px-3 py-1 hover:bg-slate-100"
                      href={`/api/jobs/${activeJobId}/protocol/download?fmt=txt`}
                    >
                      Скачать .txt
                    </a>
                    <a
                      className="rounded-md border border-slate-300 bg-white px-3 py-1 hover:bg-slate-100"
                      href={`/api/jobs/${activeJobId}/protocol/download?fmt=docx`}
                    >
                      Скачать .docx
                    </a>
                  </span>
                )}
              </div>
              {protocolError && (
                <p className="mt-2 rounded-md bg-red-50 px-3 py-1.5 text-xs text-red-700">
                  {protocolError}
                </p>
              )}
              {summary && (
                <div className="mt-3">
                  <p className="mb-1 text-xs font-medium uppercase text-slate-500">
                    Резюме встречи
                  </p>
                  <p className="whitespace-pre-wrap text-sm text-slate-700">{summary}</p>
                </div>
              )}
            </div>
          </section>
        )}
      </main>

      <VoicesModal open={voicesOpen} onClose={() => setVoicesOpen(false)} />
      <SettingsModal
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        onSaved={(saved: WebSettings) =>
          setConfig((current) =>
            current
              ? {
                  ...current,
                  glossary_enabled: saved.glossary_enabled,
                  export_formats: saved.export_formats,
                  llm_enabled: saved.llm_enabled,
                  voices_dir: saved.voices_dir_resolved,
                }
              : current,
          )
        }
      />
      <GlossaryModal open={glossaryOpen} onClose={() => setGlossaryOpen(false)} />
    </div>
  )
}

export default App
