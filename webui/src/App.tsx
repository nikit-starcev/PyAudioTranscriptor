import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import {
  api,
  EXPORT_FORMATS,
  errorMessage,
  formatDuration,
  formatSize,
  isTerminal,
  type ApplyNamesResponse,
  type ConfigInfo,
  type DoctorReport,
  type FileItem,
  type Job,
  type JobDetails,
  type JobEvent,
  type LibraryWindow,
  type ProtocolResponse,
  type SampleMeta,
  type StageTime,
  type TranscriptResult,
  type VoiceInfo,
  type WebSettings,
} from './api'
import GlossaryModal from './components/GlossaryModal'
import ModelsModal from './components/ModelsModal'
import ReadinessBanner from './components/ReadinessBanner'
import SettingsModal from './components/SettingsModal'
import SetupWizard from './components/SetupWizard'
import SpeakersPanel from './components/SpeakersPanel'
import StageTimes from './components/StageTimes'
import ThemeToggle from './components/ThemeToggle'
import TranscriptTable from './components/TranscriptTable'
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
  queued: 'bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300',
  running: 'bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300',
  done: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300',
  error: 'bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300',
  cancelled: 'bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300',
}

//: Задача реально выполняется: running и её ведёт воркер (``active``).
//: Осиротевшая running-задача (active=false) «не идёт» — прогресс не тикает.
function isLiveJob(job: { status: string; active?: boolean }): boolean {
  return job.status === 'running' && job.active !== false
}

// «Говорящих»: пусто — авто (null); иначе целое >= 1. `undefined` — ошибка ввода.
function parseSpeakerCount(raw: string | undefined): number | null | undefined {
  const trimmed = (raw ?? '').trim()
  if (trimmed === '') return null
  const value = Number(trimmed)
  if (!Number.isInteger(value) || value < 1) return undefined
  return value
}

// Настройка числа говорящих задачи для строки списка:
// точное число приоритетнее диапазона, пустое — «авто».
function formatSpeakerSetting(job: Job): string {
  if (job.num_speakers != null) return `${job.num_speakers}`
  if (job.min_speakers != null || job.max_speakers != null) {
    return `${job.min_speakers ?? '—'}–${job.max_speakers ?? '—'}`
  }
  return 'авто'
}

function App() {
  const [version, setVersion] = useState<string>('')
  const [config, setConfig] = useState<ConfigInfo | null>(null)
  const [doctor, setDoctor] = useState<DoctorReport | null>(null)
  const [doctorLoading, setDoctorLoading] = useState(false)
  const [doctorError, setDoctorError] = useState<string | null>(null)
  const [files, setFiles] = useState<FileItem[]>([])
  const [jobs, setJobs] = useState<Job[]>([])
  const [activeJobId, setActiveJobId] = useState<string | null>(null)
  const [progress, setProgress] = useState<JobEvent | null>(null)
  const [result, setResult] = useState<TranscriptResult | null>(null)
  const [samplesMeta, setSamplesMeta] = useState<Record<string, SampleMeta>>({})
  const [voicesOpen, setVoicesOpen] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [glossaryOpen, setGlossaryOpen] = useState(false)
  const [modelsOpen, setModelsOpen] = useState(false)
  const [wizardOpen, setWizardOpen] = useState(false)
  const [protocol, setProtocol] = useState<ProtocolResponse | null>(null)
  const [protocolBusy, setProtocolBusy] = useState(false)
  const [protocolError, setProtocolError] = useState<string | null>(null)
  // Формат прямой выгрузки стенограммы; по умолчанию — первый из настроек.
  const [exportFormat, setExportFormat] = useState('txt')
  const [summary, setSummary] = useState<string | null>(null)
  const [query, setQuery] = useState('')
  const [error, setError] = useState<string | null>(null)
  // Число говорящих по каждому файлу (пустая строка — авто). Ключ — путь файла.
  // Точное число (`speakerCounts`) приоритетнее диапазона `мин`/`макс`.
  const [speakerCounts, setSpeakerCounts] = useState<Record<string, string>>({})
  const [speakerMins, setSpeakerMins] = useState<Record<string, string>>({})
  const [speakerMaxs, setSpeakerMaxs] = useState<Record<string, string>>({})
  // Тайминги стадий активной задачи: завершённые (с сервера) + живой счётчик.
  const [stageTimes, setStageTimes] = useState<StageTime[]>([])
  const [finalTotalSeconds, setFinalTotalSeconds] = useState<number | null>(null)
  const [totalStartedAt, setTotalStartedAt] = useState<number | null>(null)
  const [liveStage, setLiveStage] = useState<string | null>(null)
  const [liveStageStartedAt, setLiveStageStartedAt] = useState<number | null>(null)
  const liveStageRef = useRef<string | null>(null)
  const fileInput = useRef<HTMLInputElement>(null)
  const wizardAutoShown = useRef(false)
  const exportDefaultApplied = useRef(false)

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

  const refreshDoctor = useCallback(async () => {
    setDoctorLoading(true)
    try {
      setDoctor(await api<DoctorReport>('/api/doctor'))
      setDoctorError(null)
    } catch (cause) {
      setDoctorError(errorMessage(cause))
    } finally {
      setDoctorLoading(false)
    }
  }, [])

  const recheckDoctor = useCallback(async () => {
    setDoctorLoading(true)
    try {
      setDoctor(await api<DoctorReport>('/api/doctor/recheck', { method: 'POST' }))
      setDoctorError(null)
    } catch (cause) {
      setDoctorError(errorMessage(cause))
    } finally {
      setDoctorLoading(false)
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

  const resetTiming = useCallback(() => {
    setStageTimes([])
    setFinalTotalSeconds(null)
    setTotalStartedAt(null)
    setLiveStage(null)
    setLiveStageStartedAt(null)
    liveStageRef.current = null
  }, [])

  const refreshJobTiming = useCallback(async (jobId: string) => {
    try {
      const details = await api<JobDetails>(`/api/jobs/${jobId}`)
      setStageTimes(details.stage_times ?? [])
      setFinalTotalSeconds(details.total_seconds)
      if (isLiveJob(details)) {
        setTotalStartedAt(
          details.total_seconds != null ? Date.now() - details.total_seconds * 1000 : Date.now(),
        )
        setLiveStage(details.stage ?? null)
        setLiveStageStartedAt(
          details.stage_elapsed != null ? Date.now() - details.stage_elapsed * 1000 : Date.now(),
        )
        liveStageRef.current = details.stage ?? null
      } else {
        setLiveStage(null)
        setLiveStageStartedAt(null)
        liveStageRef.current = null
      }
    } catch {
      // Тайминги не критичны: ошибку не показываем, оставляем прежние значения.
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
    void refreshDoctor()
  }, [refreshFiles, refreshJobs, refreshDoctor])

  useEffect(() => {
    if (!doctor) return
    if (doctor.summary.critical_failures > 0 && !wizardAutoShown.current) {
      wizardAutoShown.current = true
      setWizardOpen(true)
    }
  }, [doctor])

  useEffect(() => {
    // Формат выгрузки по умолчанию — первый из настроек; применяем один раз,
    // чтобы не перезаписывать выбор пользователя при обновлении конфигурации.
    if (exportDefaultApplied.current) return
    const first = config?.export_formats?.[0]
    if (!first) return
    exportDefaultApplied.current = true
    setExportFormat(first)
  }, [config])

  useEffect(() => {
    if (!activeJobId) return
    const source = new EventSource(`/api/jobs/${activeJobId}/events`)
    source.onmessage = (message) => {
      const event = JSON.parse(message.data) as JobEvent
      setProgress(event)
      if (event.stage_times) setStageTimes(event.stage_times)
      if (isTerminal(event.status)) {
        source.close()
        setLiveStage(null)
        setLiveStageStartedAt(null)
        liveStageRef.current = null
        void refreshJobs()
        void refreshJobTiming(activeJobId)
        if (event.status === 'done') void loadResult(activeJobId)
        return
      }
      // Живой счётчик текущей стадии: при её смене перезапускаем отсчёт.
      if (event.stage !== liveStageRef.current) {
        liveStageRef.current = event.stage
        setLiveStage(event.stage)
        setLiveStageStartedAt(Date.now() - (event.stage_elapsed ?? 0) * 1000)
      } else if (event.stage_elapsed != null) {
        setLiveStageStartedAt(Date.now() - event.stage_elapsed * 1000)
      }
      if (event.elapsed != null) {
        setTotalStartedAt(Date.now() - event.elapsed * 1000)
      }
    }
    source.onerror = () => source.close()
    return () => source.close()
  }, [activeJobId, refreshJobs, loadResult, refreshJobTiming])

  const enqueue = useCallback(
    async (path: string) => {
      setError(null)
      const numSpeakers = parseSpeakerCount(speakerCounts[path])
      const minSpeakers = parseSpeakerCount(speakerMins[path])
      const maxSpeakers = parseSpeakerCount(speakerMaxs[path])
      if (
        numSpeakers === undefined ||
        minSpeakers === undefined ||
        maxSpeakers === undefined
      ) {
        setError('Число говорящих должно быть целым числом не меньше 1 или пустым (авто)')
        return
      }
      if (minSpeakers != null && maxSpeakers != null && minSpeakers > maxSpeakers) {
        setError('Минимум говорящих не может быть больше максимума')
        return
      }
      try {
        const job = await api<Job>('/api/jobs', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            path,
            num_speakers: numSpeakers,
            min_speakers: minSpeakers,
            max_speakers: maxSpeakers,
          }),
        })
        setActiveJobId(job.id)
        setProgress({ stage: 'queued', fraction: 0, message: 'В очереди', status: 'queued' })
        setResult(null)
        setSummary(null)
        setProtocol(null)
        setSamplesMeta({})
        resetTiming()
        await refreshJobs()
      } catch (cause) {
        setError(errorMessage(cause))
      }
    },
    [refreshJobs, resetTiming, speakerCounts, speakerMins, speakerMaxs],
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
        resetTiming()
        await refreshJobs()
      } catch (cause) {
        setError(errorMessage(cause))
      }
    },
    [refreshJobs, resetTiming],
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
      resetTiming()
      try {
        const details = await api<JobDetails>(`/api/jobs/${jobId}`)
        setProgress({
          stage: details.stage ?? details.status,
          fraction: details.fraction,
          message: '',
          status: details.status,
          active: details.active,
        })
        setStageTimes(details.stage_times ?? [])
        setFinalTotalSeconds(details.total_seconds)
        if (isLiveJob(details)) {
          setTotalStartedAt(
            details.total_seconds != null ? Date.now() - details.total_seconds * 1000 : Date.now(),
          )
          setLiveStage(details.stage ?? null)
          setLiveStageStartedAt(
            details.stage_elapsed != null ? Date.now() - details.stage_elapsed * 1000 : Date.now(),
          )
          liveStageRef.current = details.stage ?? null
        }
        if (details.status === 'done') await loadResult(jobId)
      } catch (cause) {
        setError(errorMessage(cause))
      }
    },
    [loadResult, resetTiming],
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
          resetTiming()
        }
        await refreshJobs()
      } catch (cause) {
        setError(errorMessage(cause))
      }
    },
    [activeJobId, refreshJobs, resetTiming],
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

  const deleteFile = useCallback(
    async (name: string) => {
      if (!window.confirm(`Удалить загруженный файл «${name}»?`)) return
      setError(null)
      try {
        await api<{ deleted: string }>(`/api/files/${encodeURIComponent(name)}`, {
          method: 'DELETE',
        })
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
    async (
      jobId: string,
      speakerId: string,
      name: string,
      window?: LibraryWindow,
    ): Promise<VoiceInfo> => {
      return api<VoiceInfo>(`/api/jobs/${jobId}/speakers/${speakerId}/to-library`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, start: window?.start, end: window?.end }),
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
  const readinessBlocked = (doctor?.summary.critical_failures ?? 0) > 0
  const blockedHint = readinessBlocked
    ? 'Запуск заблокирован: сначала устраните критичные проблемы Готовности'
    : undefined

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 dark:bg-slate-950 dark:text-slate-100">
      <header className="border-b border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-3 gap-y-2 px-6 py-4">
          <h1 className="text-xl font-semibold">AudioTranscriber</h1>
          <span className="text-sm text-slate-400 dark:text-slate-500">
            веб-интерфейс · этап 3
          </span>
          <div className="ml-auto flex flex-wrap items-center gap-2">
            <button
              onClick={() => setWizardOpen(true)}
              className="rounded-md border border-slate-300 px-3 py-1.5 text-sm hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800"
            >
              Мастер настройки
            </button>
            <button
              onClick={() => setModelsOpen(true)}
              className="rounded-md border border-slate-300 px-3 py-1.5 text-sm hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800"
            >
              Модели
            </button>
            <button
              onClick={() => setVoicesOpen(true)}
              className="rounded-md border border-slate-300 px-3 py-1.5 text-sm hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800"
            >
              Коллекция голосов
            </button>
            <button
              onClick={() => setGlossaryOpen(true)}
              className="rounded-md border border-slate-300 px-3 py-1.5 text-sm hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800"
            >
              Глоссарий
            </button>
            <button
              onClick={() => setSettingsOpen(true)}
              className="rounded-md border border-slate-300 px-3 py-1.5 text-sm hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800"
            >
              Настройки
            </button>
            <ThemeToggle />
          </div>
          {version && (
            <span className="text-xs text-slate-400 dark:text-slate-500">v{version}</span>
          )}
        </div>
      </header>

      <main className="mx-auto max-w-6xl space-y-6 px-6 py-6">
        {error && (
          <div className="rounded-md border border-red-200 bg-red-50 px-4 py-2 text-sm text-red-700 dark:border-red-900 dark:bg-red-950/50 dark:text-red-300">
            {error}
          </div>
        )}

        <ReadinessBanner
          report={doctor}
          loading={doctorLoading}
          error={doctorError}
          onRecheck={() => void recheckDoctor()}
        />

        {config && (
          <p className="text-xs text-slate-500 dark:text-slate-400">
            Файлы: <code>{config.input_dir}</code> · Результаты: <code>{config.output_dir}</code> ·
            Экспорт: {config.export_formats.join(', ')}
            {config.llm_enabled ? ' · LLM вкл.' : ''}
            {config.glossary_enabled ? ' · глоссарий вкл.' : ''}
          </p>
        )}

        <div className="grid gap-6 lg:grid-cols-2">
          <section className="rounded-lg border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900">
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
                  className="rounded-md bg-slate-800 px-3 py-1.5 text-sm text-white hover:bg-slate-700 dark:bg-slate-200 dark:text-slate-900 dark:hover:bg-white"
                >
                  Загрузить файл
                </button>
              </div>
            </div>
            {files.length === 0 ? (
              <p className="py-6 text-center text-sm text-slate-400 dark:text-slate-500">
                Файлов пока нет
              </p>
            ) : (
              <ul className="divide-y divide-slate-100 dark:divide-slate-800">
                {files.map((file) => (
                  <li key={file.path} className="flex items-center gap-3 py-2">
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm">{file.name}</p>
                      <p className="text-xs text-slate-400 dark:text-slate-500">
                        {formatSize(file.size)} · {formatDuration(file.duration)}
                      </p>
                    </div>
                    <div className="flex items-center gap-1">
                      <label
                        htmlFor={`speakers-${file.path}`}
                        className="whitespace-nowrap text-xs text-slate-400 dark:text-slate-500"
                      >
                        Точно
                      </label>
                      <input
                        id={`speakers-${file.path}`}
                        type="number"
                        min={1}
                        step={1}
                        placeholder="авто"
                        title="Точное число говорящих: пусто — автоопределение"
                        value={speakerCounts[file.path] ?? ''}
                        onChange={(event) =>
                          setSpeakerCounts((prev) => ({
                            ...prev,
                            [file.path]: event.target.value,
                          }))
                        }
                        className="w-14 rounded-md border border-slate-300 px-2 py-1 text-xs focus:border-blue-400 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100 dark:placeholder-slate-500"
                      />
                      <label
                        htmlFor={`min-speakers-${file.path}`}
                        className="whitespace-nowrap text-xs text-slate-400 dark:text-slate-500"
                      >
                        Мин
                      </label>
                      <input
                        id={`min-speakers-${file.path}`}
                        type="number"
                        min={1}
                        step={1}
                        placeholder="—"
                        title="Нижняя граница числа говорящих: пусто — без ограничения"
                        value={speakerMins[file.path] ?? ''}
                        onChange={(event) =>
                          setSpeakerMins((prev) => ({
                            ...prev,
                            [file.path]: event.target.value,
                          }))
                        }
                        className="w-14 rounded-md border border-slate-300 px-2 py-1 text-xs focus:border-blue-400 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100 dark:placeholder-slate-500"
                      />
                      <label
                        htmlFor={`max-speakers-${file.path}`}
                        className="whitespace-nowrap text-xs text-slate-400 dark:text-slate-500"
                      >
                        Макс
                      </label>
                      <input
                        id={`max-speakers-${file.path}`}
                        type="number"
                        min={1}
                        step={1}
                        placeholder="—"
                        title="Верхняя граница числа говорящих: пусто — без ограничения"
                        value={speakerMaxs[file.path] ?? ''}
                        onChange={(event) =>
                          setSpeakerMaxs((prev) => ({
                            ...prev,
                            [file.path]: event.target.value,
                          }))
                        }
                        className="w-14 rounded-md border border-slate-300 px-2 py-1 text-xs focus:border-blue-400 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100 dark:placeholder-slate-500"
                      />
                    </div>
                    <button
                      onClick={() => void enqueue(file.path)}
                      disabled={readinessBlocked}
                      title={blockedHint}
                      className="rounded-md border border-slate-300 px-3 py-1 text-xs hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-40 dark:border-slate-700 dark:hover:bg-slate-800"
                    >
                      В очередь
                    </button>
                    <button
                      onClick={() => void deleteFile(file.name)}
                      title="Удалить загруженный файл"
                      className="rounded-md border border-slate-300 px-2 py-1 text-xs text-slate-500 hover:bg-red-50 hover:text-red-600 dark:border-slate-700 dark:text-slate-400 dark:hover:bg-red-950/50 dark:hover:text-red-300"
                    >
                      Удалить
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </section>

          <section className="rounded-lg border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="font-medium">Задачи</h2>
              <button
                onClick={() => void refreshJobs()}
                className="rounded-md border border-slate-300 px-3 py-1 text-xs hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800"
              >
                Обновить
              </button>
            </div>
            {jobs.length === 0 ? (
              <p className="py-6 text-center text-sm text-slate-400 dark:text-slate-500">
                Задач пока нет
              </p>
            ) : (
              <ul className="divide-y divide-slate-100 dark:divide-slate-800">
                {jobs.map((job) => (
                  <li key={job.id} className="flex items-center gap-3 py-2">
                    <button
                      onClick={() => void openJob(job.id)}
                      className="min-w-0 flex-1 text-left"
                    >
                      <p className="truncate text-sm">{job.name}</p>
                      <p className="text-xs text-slate-400 dark:text-slate-500">
                        {job.stage ? `${job.stage} · ` : ''}
                        {job.fraction != null ? `${Math.round(job.fraction * 100)}%` : '—'}
                        {' · говорящих: '}
                        {formatSpeakerSetting(job)}
                      </p>
                    </button>
                    <span
                      className={`rounded-full px-2 py-0.5 text-xs ${
                        STATUS_STYLES[job.status] ??
                        'bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300'
                      }`}
                    >
                      {STATUS_LABELS[job.status] ?? job.status}
                    </span>
                    {(job.status === 'queued' ||
                      job.status === 'done' ||
                      job.status === 'error' ||
                      (job.status === 'running' && job.active === false)) && (
                      <button
                        onClick={() => void runJob(job.id)}
                        disabled={readinessBlocked}
                        title={blockedHint}
                        className="rounded-md bg-emerald-600 px-3 py-1 text-xs text-white hover:bg-emerald-500 disabled:cursor-not-allowed disabled:opacity-40"
                      >
                        Запустить
                      </button>
                    )}
                    {(job.status !== 'running' || job.active === false) && (
                      <button
                        onClick={() => void deleteJob(job.id)}
                        className="rounded-md border border-slate-300 px-2 py-1 text-xs text-slate-500 hover:bg-red-50 hover:text-red-600 dark:border-slate-700 dark:text-slate-400 dark:hover:bg-red-950/50 dark:hover:text-red-300"
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
          <section className="rounded-lg border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900">
            <h2 className="mb-3 font-medium">
              Прогресс{activeJob ? ` · ${activeJob.name}` : ''}
            </h2>
            <div className="mb-2 h-2 w-full overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
              <div
                className={`h-full rounded-full bg-blue-500 transition-all ${
                  progress?.fraction == null ? 'animate-pulse' : ''
                }`}
                style={{ width: `${Math.round((progress?.fraction ?? 0) * 100)}%` }}
              />
            </div>
            <p className="mb-3 text-sm text-slate-600 dark:text-slate-300">
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
                          ? 'text-emerald-600 dark:text-emerald-400'
                          : state === 'current'
                            ? 'text-blue-600 dark:text-blue-400'
                            : 'text-slate-300 dark:text-slate-600'
                      }
                    >
                      {state === 'done' ? '✓' : state === 'current' ? '●' : '·'}
                    </span>
                    <span
                      className={state === 'pending' ? 'text-slate-400 dark:text-slate-500' : ''}
                    >
                      {stage.label}
                    </span>
                  </li>
                )
              })}
            </ol>
            <StageTimes
              times={stageTimes}
              running={progress != null && !isTerminal(progress.status) && progress.active !== false}
              totalStartedAt={totalStartedAt}
              finalTotalSeconds={finalTotalSeconds}
              currentStage={liveStage}
              currentStartedAt={liveStageStartedAt}
            />
          </section>
        )}

        {result && activeJobId && (
          <section className="space-y-4 rounded-lg border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900">
            <div className="flex flex-wrap items-center gap-3">
              <h2 className="font-medium">Стенограмма</h2>
              <span className="text-xs text-slate-400 dark:text-slate-500">
                {result.entries.length} реплик · {result.speakers.length} говорящих ·{' '}
                {formatDuration(result.duration)} · язык {result.language ?? '—'}
              </span>
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Поиск по тексту..."
                className="ml-auto w-64 rounded-md border border-slate-300 px-3 py-1.5 text-sm focus:border-blue-400 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100 dark:placeholder-slate-500"
              />
            </div>

            <div className="flex flex-wrap items-center gap-4 text-xs text-slate-500 dark:text-slate-400">
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
              onToLibrary={(speakerId, name, window) =>
                saveToLibrary(activeJobId, speakerId, name, window)
              }
              onApplyNames={() => applyNames(activeJobId)}
              onOpenVoices={() => setVoicesOpen(true)}
            />

            <TranscriptTable
              jobId={activeJobId}
              entries={filteredEntries}
              speakers={result.speakers}
            />

            <div className="rounded-md border border-slate-200 bg-slate-50 p-3 dark:border-slate-800 dark:bg-slate-800/50">
              <div className="flex flex-wrap items-center gap-3">
                <button
                  onClick={() => void generateProtocol(activeJobId)}
                  disabled={protocolBusy}
                  className="rounded-md bg-slate-800 px-3 py-1.5 text-sm text-white hover:bg-slate-700 disabled:opacity-40 dark:bg-slate-200 dark:text-slate-900 dark:hover:bg-white"
                >
                  {protocolBusy ? 'Формирование протокола…' : 'Сформировать протокол'}
                </button>
                {protocolBusy && (
                  <span className="animate-pulse text-xs text-slate-500 dark:text-slate-400">
                    Считается резюме и экспорт — это может занять время
                  </span>
                )}
                <span className="flex items-center gap-2">
                  <label
                    htmlFor="export-format"
                    className="text-xs text-slate-500 dark:text-slate-400"
                  >
                    Формат
                  </label>
                  <select
                    id="export-format"
                    value={exportFormat}
                    onChange={(event) => setExportFormat(event.target.value)}
                    className="rounded-md border border-slate-300 bg-white px-2 py-1.5 text-sm focus:border-blue-400 focus:outline-none dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
                  >
                    {EXPORT_FORMATS.map((fmt) => (
                      <option key={fmt} value={fmt}>
                        {fmt.toUpperCase()}
                      </option>
                    ))}
                  </select>
                  <a
                    href={`/api/jobs/${activeJobId}/export?fmt=${exportFormat}`}
                    download
                    className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm hover:bg-slate-100 dark:border-slate-700 dark:bg-slate-900 dark:hover:bg-slate-800"
                  >
                    Скачать расшифровку
                  </a>
                </span>
                {protocol && (
                  <span className="flex items-center gap-2 text-sm">
                    <a
                      className="rounded-md border border-slate-300 bg-white px-3 py-1 hover:bg-slate-100 dark:border-slate-700 dark:bg-slate-900 dark:hover:bg-slate-800"
                      href={`/api/jobs/${activeJobId}/protocol/download?fmt=txt`}
                    >
                      Скачать .txt
                    </a>
                    <a
                      className="rounded-md border border-slate-300 bg-white px-3 py-1 hover:bg-slate-100 dark:border-slate-700 dark:bg-slate-900 dark:hover:bg-slate-800"
                      href={`/api/jobs/${activeJobId}/protocol/download?fmt=docx`}
                    >
                      Скачать .docx
                    </a>
                  </span>
                )}
              </div>
              {protocolError && (
                <p className="mt-2 rounded-md bg-red-50 px-3 py-1.5 text-xs text-red-700 dark:bg-red-950/50 dark:text-red-300">
                  {protocolError}
                </p>
              )}
              {summary && (
                <div className="mt-3">
                  <p className="mb-1 text-xs font-medium uppercase text-slate-500 dark:text-slate-400">
                    Резюме встречи
                  </p>
                  <p className="whitespace-pre-wrap text-sm text-slate-700 dark:text-slate-200">
                    {summary}
                  </p>
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
        onSaved={(saved: WebSettings) => {
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
          void refreshDoctor()
        }}
      />
      <GlossaryModal open={glossaryOpen} onClose={() => setGlossaryOpen(false)} />
      <ModelsModal
        open={modelsOpen}
        onClose={() => setModelsOpen(false)}
        onChanged={() => {
          void refreshDoctor()
          void refreshFiles()
        }}
      />
      <SetupWizard
        open={wizardOpen}
        onClose={() => setWizardOpen(false)}
        report={doctor}
        onRecheck={() => void recheckDoctor()}
        onChanged={() => void refreshDoctor()}
      />
    </div>
  )
}

export default App
