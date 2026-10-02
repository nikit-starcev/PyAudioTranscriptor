import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import {
  api,
  describeApply,
  EXPORT_FORMATS,
  errorMessage,
  formatDuration,
  formatSize,
  isTerminal,
  type ApplyNamesResponse,
  type ConfigInfo,
  type DoctorReport,
  type Entry,
  type FileItem,
  type Job,
  type JobDetails,
  type JobEvent,
  type LibraryWindow,
  type ProtocolResponse,
  type SampleMeta,
  type StageTime,
  type TranscriptEditsRequest,
  type TranscriptResult,
  type VoiceInfo,
  type WebSettings,
} from './api'
import GlossaryModal from './components/GlossaryModal'
import ModelsModal from './components/ModelsModal'
import ProgressSummary from './components/ProgressSummary'
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

//: Задержка ручного переподключения SSE, когда браузер не ретраит сам
//: (``EventSource`` фатально закрывается при не-200 ответе). Меньше дефолтных
//: ~3 с, чтобы UI быстрее подхватывал состояние после короткого сбоя.
const SSE_RECONNECT_DELAY_MS = 2000

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
  // Показывать ли в списке «Файлы» уже обработанные файлы (#16). По умолчанию
  // они скрыты; значение дублируется в ref, чтобы `refreshFiles` не менял
  // идентичность и не заставлял переподключать SSE.
  const [showProcessed, setShowProcessed] = useState(false)
  const showProcessedRef = useRef(false)
  const [jobs, setJobs] = useState<Job[]>([])
  // Показывать ли в списке «Задачи» мягко удалённые (#30). По умолчанию скрыты;
  // значение дублируется в ref, чтобы `refreshJobs` не менял идентичность и не
  // заставлял переподключать SSE.
  const [showDeleted, setShowDeleted] = useState(false)
  const showDeletedRef = useRef(false)
  const [activeJobId, setActiveJobId] = useState<string | null>(null)
  // Счётчик запусков: повторный запуск той же задачи должен переподключить SSE
  // (значение activeJobId при этом не меняется, и эффект не сработал бы).
  const [runSeq, setRunSeq] = useState(0)
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
  // Отдельное действие «Переопределить говорящих» (#37): переиспользует
  // enrollment (POST apply-names), поэтому показывает свой итог в баннере.
  const [applyBusy, setApplyBusy] = useState(false)
  const [applyNotice, setApplyNotice] = useState<{
    kind: 'info' | 'error'
    text: string
  } | null>(null)
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

  // Возвращает актуальный список задач или ``null``, если запрос не удался.
  // ``silent`` подавляет баннер ошибки: используется при восстановлении SSE,
  // где обрыв сети/рестарт сервера — ожидаемая ситуация, а не ошибка пользователя.
  const refreshJobs = useCallback(
    async (options?: { silent?: boolean }): Promise<Job[] | null> => {
      try {
        const suffix = showDeletedRef.current ? '?include_deleted=true' : ''
        const list = await api<Job[]>(`/api/jobs${suffix}`)
        setJobs(list)
        return list
      } catch (cause) {
        if (!options?.silent) setError(errorMessage(cause))
        return null
      }
    },
    [],
  )

  const toggleDeleted = useCallback(
    (value: boolean) => {
      showDeletedRef.current = value
      setShowDeleted(value)
      void refreshJobs()
    },
    [refreshJobs],
  )

  const refreshFiles = useCallback(async () => {
    try {
      const suffix = showProcessedRef.current ? '?include_processed=true' : ''
      setFiles(await api<FileItem[]>(`/api/files${suffix}`))
    } catch (cause) {
      setError(errorMessage(cause))
    }
  }, [])

  const toggleProcessed = useCallback(
    (value: boolean) => {
      showProcessedRef.current = value
      setShowProcessed(value)
      void refreshFiles()
    },
    [refreshFiles],
  )

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
    let disposed = false
    let source: EventSource | null = null
    let retryTimer: number | undefined

    const clearRetry = () => {
      if (retryTimer !== undefined) {
        window.clearTimeout(retryTimer)
        retryTimer = undefined
      }
    }

    // Ручное переподключение на случай, когда браузер не ретраит сам
    // (``EventSource`` фатально закрывается при не-200 ответе / неверном MIME).
    const scheduleReconnect = () => {
      if (disposed || retryTimer !== undefined) return
      retryTimer = window.setTimeout(() => {
        retryTimer = undefined
        if (!disposed) connect()
      }, SSE_RECONNECT_DELAY_MS)
    }

    const connect = () => {
      source?.close()
      const stream = new EventSource(`/api/jobs/${activeJobId}/events`)
      source = stream

      // Терминальное состояние: закрываем поток и досинхронизируем UI ровно так
      // же, как по обычному конечному событию.
      const finish = (status: string) => {
        stream.close()
        setLiveStage(null)
        setLiveStageStartedAt(null)
        liveStageRef.current = null
        void refreshJobs()
        void refreshJobTiming(activeJobId)
        if (status === 'done') {
          // Успешный прогон убирает файл из основного списка «Файлы» (#16).
          void refreshFiles()
          void loadResult(activeJobId)
        }
      }

      // Задача исчезла (удалена/очищена): ретраить нечего, сбрасываем панель.
      const clearActiveJob = () => {
        stream.close()
        setActiveJobId(null)
        setResult(null)
        setSummary(null)
        setProtocol(null)
        setSamplesMeta({})
        setProgress(null)
        setStageTimes([])
        setFinalTotalSeconds(null)
        setTotalStartedAt(null)
        setLiveStage(null)
        setLiveStageStartedAt(null)
        liveStageRef.current = null
      }

      stream.onmessage = (message) => {
        const event = JSON.parse(message.data) as JobEvent
        setProgress(event)
        if (event.stage_times) setStageTimes(event.stage_times)
        if (isTerminal(event.status)) {
          finish(event.status)
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

      // Обрыв SSE — не повод «зависать». Не закрываем поток безусловно:
      // подтягиваем актуальное состояние из API и решаем по нему, продолжать ли
      // слежение. Браузер сам ретраит соединение (readyState CONNECTING); при
      // фатальном закрытии (CLOSED) переподключаемся вручную.
      stream.onerror = () => {
        if (disposed) return
        void (async () => {
          const list = await refreshJobs({ silent: true })
          if (disposed) return
          if (list == null) {
            // API недоступен — состояние неизвестно. Не закрываем поток.
            if (source === stream && stream.readyState === EventSource.CLOSED) {
              scheduleReconnect()
            }
            return
          }
          const current = list.find((job) => job.id === activeJobId)
          if (!current || current.deleted) {
            clearActiveJob()
            return
          }
          if (isTerminal(current.status)) {
            // Снимок задачи актуализирует прогресс, если событие было пропущено.
            setProgress({
              stage: current.stage ?? current.status,
              fraction: current.fraction,
              message: '',
              status: current.status,
              active: current.active,
              stage_times: current.stage_times,
              progress_percent: current.progress_percent,
              eta_seconds: current.eta_seconds,
              eta_by_stage: current.eta_by_stage,
              health: current.health,
            })
            finish(current.status)
            return
          }
          // Задача ещё идёт: синхронизируем тайминги и ждём переподключения.
          await refreshJobTiming(activeJobId)
          if (disposed) return
          if (source === stream && stream.readyState === EventSource.CLOSED) {
            scheduleReconnect()
          }
        })()
      }
    }

    connect()
    return () => {
      disposed = true
      clearRetry()
      source?.close()
    }
  }, [activeJobId, runSeq, refreshJobs, refreshFiles, loadResult, refreshJobTiming])

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
        setRunSeq((value) => value + 1)
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
        setRunSeq((value) => value + 1)
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

  const stopJob = useCallback(
    async (jobId: string) => {
      setError(null)
      try {
        await api<{ id: string; status: string }>(`/api/jobs/${jobId}/cancel`, {
          method: 'POST',
        })
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

  // Обычное «Удалить» — мягкое: задача скрывается, артефакты сохраняются (#30).
  const deleteJob = useCallback(
    async (jobId: string) => {
      setError(null)
      try {
        await api<Job>(`/api/jobs/${jobId}`, { method: 'DELETE' })
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

  const restoreJob = useCallback(
    async (jobId: string) => {
      setError(null)
      try {
        await api<Job>(`/api/jobs/${jobId}/restore`, { method: 'POST' })
        await refreshJobs()
      } catch (cause) {
        setError(errorMessage(cause))
      }
    },
    [refreshJobs],
  )

  // «Удалить навсегда» — окончательная очистка записи и артефактов (с подтверждением).
  const purgeJob = useCallback(
    async (jobId: string, name: string) => {
      if (
        !window.confirm(
          `Удалить задачу «${name}» навсегда?\nРезультат и образцы говорящих будут удалены безвозвратно.`,
        )
      ) {
        return
      }
      setError(null)
      try {
        await api<{ purged: string }>(`/api/jobs/${jobId}?purge=true`, {
          method: 'DELETE',
        })
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

  const restoreFile = useCallback(
    async (name: string) => {
      setError(null)
      try {
        await api<{ restored: string; processed: boolean }>(
          `/api/files/${encodeURIComponent(name)}/restore`,
          { method: 'POST' },
        )
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

  const patchTranscript = useCallback(
    async (jobId: string, body: TranscriptEditsRequest) => {
      const updated = await api<TranscriptResult>(`/api/jobs/${jobId}/transcript`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      setResult(updated)
    },
    [],
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

  //: «Переопределить говорящих» (#37): применяет enrollment текущего результата
  //: по актуальной библиотеке `voices/` без повторного распознавания и
  //: показывает, сколько имён сопоставлено и лучших недобранных. Ручные правки
  //: текста (#26) и текущие имена сохраняются на бэкенде.
  const runApplyNames = useCallback(
    async (jobId: string) => {
      setError(null)
      setApplyBusy(true)
      setApplyNotice(null)
      try {
        const response = await applyNames(jobId)
        setApplyNotice({
          kind: response.error ? 'error' : 'info',
          text: describeApply(response, response.result.speakers.length),
        })
      } catch (cause) {
        setApplyNotice({ kind: 'error', text: errorMessage(cause) })
      } finally {
        setApplyBusy(false)
      }
    },
    [applyNames],
  )

  //: Смена активной задачи делает прежний итог сопоставления неактуальным.
  useEffect(() => {
    setApplyNotice(null)
  }, [activeJobId])

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
  // Идёт ли прогон прямо сейчас: running и задачу ведёт воркер (``active``).
  const progressRunning =
    progress != null && !isTerminal(progress.status) && progress.active !== false
  // Полоса показывает сводный процент прогона (fallback — доля текущей стадии).
  const overallPercent =
    progress?.progress_percent ?? (progress?.fraction != null ? progress.fraction * 100 : 0)
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
              onClick={() => {
                if (activeJobId) void runApplyNames(activeJobId)
              }}
              disabled={!result || !activeJobId || applyBusy}
              title="Сопоставить говорящих с именами по актуальной библиотеке голосов (enrollment, без повторного распознавания)"
              className="rounded-md border border-blue-300 px-3 py-1.5 text-sm text-blue-700 hover:bg-blue-50 disabled:cursor-not-allowed disabled:opacity-40 dark:border-blue-800 dark:text-blue-300 dark:hover:bg-blue-950/50"
            >
              {applyBusy ? 'Переопределяю…' : 'Переопределить говорящих'}
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

        {applyNotice && (
          <div
            role="status"
            className={`flex items-start justify-between gap-3 rounded-md border px-4 py-2 text-sm ${
              applyNotice.kind === 'error'
                ? 'border-red-200 bg-red-50 text-red-700 dark:border-red-900 dark:bg-red-950/50 dark:text-red-300'
                : 'border-blue-200 bg-blue-50 text-blue-800 dark:border-blue-900 dark:bg-blue-950/50 dark:text-blue-200'
            }`}
          >
            <p className="min-w-0 break-words">{applyNotice.text}</p>
            <button
              type="button"
              onClick={() => setApplyNotice(null)}
              title="Скрыть сообщение"
              className="shrink-0 rounded border border-current/30 px-2 py-0.5 text-xs hover:bg-white/40 dark:hover:bg-black/20"
            >
              Скрыть
            </button>
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
            {config.llm_enabled ? ` · LLM вкл. (${config.llm_provider})` : ''}
            {config.llm_external ? ' · внешний LLM: текст уходит за пределы машины' : ''}
            {config.glossary_enabled ? ' · глоссарий вкл.' : ''}
          </p>
        )}

        <div className="grid gap-6 lg:grid-cols-2">
          <section className="@container rounded-lg border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900">
            <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
              <h2 className="font-medium">Файлы</h2>
              <div className="flex items-center gap-3">
                <label
                  className="flex items-center gap-1.5 text-xs text-slate-500 dark:text-slate-400"
                  title="Показать файлы, уже успешно обработанные: их можно вернуть в основной список"
                >
                  <input
                    type="checkbox"
                    checked={showProcessed}
                    onChange={(event) => toggleProcessed(event.target.checked)}
                    className="h-3.5 w-3.5 rounded border-slate-300 text-blue-600 focus:ring-blue-400 dark:border-slate-600 dark:bg-slate-800"
                  />
                  Обработанные
                </label>
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
                  <li
                    key={file.path}
                    className="grid grid-cols-[minmax(0,1fr)] items-center gap-x-3 gap-y-2 py-3 @xl:grid-cols-[minmax(0,1fr)_4rem_4rem_6.5rem_auto]"
                  >
                    {/* Имя файла: до двух строк с переносом, полное — в tooltip.
                        `minmax(0,1fr)` не даёт кнопкам «съесть» имя. */}
                    <div className="min-w-0 @xl:col-start-1 @xl:row-start-1 @xl:flex @xl:min-h-[2.5rem] @xl:items-center">
                      <p
                        className="line-clamp-2 break-words text-sm leading-snug"
                        title={file.name}
                      >
                        {file.name}
                      </p>
                    </div>

                    {/* Мета: на узких — одной строкой под именем, на широких — колонки. */}
                    <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-slate-400 dark:text-slate-500 @xl:contents">
                      <span
                        className="tabular-nums @xl:col-start-2 @xl:row-start-1 @xl:text-right"
                        title="Размер файла"
                      >
                        {formatSize(file.size)}
                      </span>
                      <span
                        className="tabular-nums @xl:col-start-3 @xl:row-start-1 @xl:text-right"
                        title="Длительность"
                      >
                        {formatDuration(file.duration)}
                      </span>
                      <span className="@xl:col-start-4 @xl:row-start-1">
                        {file.processed ? (
                          <span className="whitespace-nowrap rounded-full bg-emerald-100 px-2 py-0.5 text-xs text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-300">
                            Обработан
                          </span>
                        ) : (
                          <span className="whitespace-nowrap rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-500 dark:bg-slate-800 dark:text-slate-400">
                            Не обработан
                          </span>
                        )}
                      </span>
                    </div>

                    {/* Действия — на месте, переносятся и не перекрывают имя. */}
                    <div className="flex flex-wrap items-center justify-end gap-1.5 @xl:col-start-5 @xl:row-start-1">
                      {file.processed ? (
                        <>
                          <button
                            onClick={() => void restoreFile(file.name)}
                            title="Вернуть файл в основной список «Файлы»"
                            className="whitespace-nowrap rounded-md border border-emerald-300 px-3 py-1 text-xs text-emerald-700 hover:bg-emerald-50 dark:border-emerald-800 dark:text-emerald-300 dark:hover:bg-emerald-950/50"
                          >
                            Вернуть
                          </button>
                          <button
                            onClick={() => void deleteFile(file.name)}
                            title="Удалить загруженный файл"
                            className="whitespace-nowrap rounded-md border border-slate-300 px-2 py-1 text-xs text-slate-500 hover:bg-red-50 hover:text-red-600 dark:border-slate-700 dark:text-slate-400 dark:hover:bg-red-950/50 dark:hover:text-red-300"
                          >
                            Удалить
                          </button>
                        </>
                      ) : (
                        <>
                          <button
                            onClick={() => void enqueue(file.path)}
                            disabled={readinessBlocked}
                            title={blockedHint}
                            className="whitespace-nowrap rounded-md border border-slate-300 px-3 py-1 text-xs hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-40 dark:border-slate-700 dark:hover:bg-slate-800"
                          >
                            В очередь
                          </button>
                          <button
                            onClick={() => void deleteFile(file.name)}
                            title="Удалить загруженный файл"
                            className="whitespace-nowrap rounded-md border border-slate-300 px-2 py-1 text-xs text-slate-500 hover:bg-red-50 hover:text-red-600 dark:border-slate-700 dark:text-slate-400 dark:hover:bg-red-950/50 dark:hover:text-red-300"
                          >
                            Удалить
                          </button>
                        </>
                      )}
                    </div>

                    {/* Число говорящих — только для необработанных; на широких
                        экранах отдельной строкой во всю ширину. */}
                    {!file.processed && (
                      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 @xl:col-span-full @xl:row-start-2">
                        <span className="flex items-center gap-1 whitespace-nowrap">
                          <label
                            htmlFor={`speakers-${file.path}`}
                            className="text-xs text-slate-400 dark:text-slate-500"
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
                        </span>
                        <span className="flex items-center gap-1 whitespace-nowrap">
                          <label
                            htmlFor={`min-speakers-${file.path}`}
                            className="text-xs text-slate-400 dark:text-slate-500"
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
                        </span>
                        <span className="flex items-center gap-1 whitespace-nowrap">
                          <label
                            htmlFor={`max-speakers-${file.path}`}
                            className="text-xs text-slate-400 dark:text-slate-500"
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
                        </span>
                      </div>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </section>

          <section className="rounded-lg border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900">
            <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
              <h2 className="font-medium">Задачи</h2>
              <div className="flex items-center gap-3">
                <label
                  className="flex items-center gap-1.5 text-xs text-slate-500 dark:text-slate-400"
                  title="Показать мягко удалённые задачи: их можно восстановить или удалить навсегда"
                >
                  <input
                    type="checkbox"
                    checked={showDeleted}
                    onChange={(event) => toggleDeleted(event.target.checked)}
                    className="h-3.5 w-3.5 rounded border-slate-300 text-blue-600 focus:ring-blue-400 dark:border-slate-600 dark:bg-slate-800"
                  />
                  Показать удалённые
                </label>
                <button
                  onClick={() => void refreshJobs()}
                  className="rounded-md border border-slate-300 px-3 py-1 text-xs hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800"
                >
                  Обновить
                </button>
              </div>
            </div>
            {jobs.length === 0 ? (
              <p className="py-6 text-center text-sm text-slate-400 dark:text-slate-500">
                Задач пока нет
              </p>
            ) : (
              <ul className="divide-y divide-slate-100 dark:divide-slate-800">
                {jobs.map((job) => (
                  <li
                    key={job.id}
                    className={`flex flex-wrap items-center gap-x-3 gap-y-1.5 py-2 ${
                      job.deleted ? 'opacity-60' : ''
                    }`}
                  >
                    <button
                      onClick={() => void openJob(job.id)}
                      className="min-w-0 w-full text-left sm:w-auto sm:flex-1"
                    >
                      <p
                        className={`truncate text-sm ${job.deleted ? 'line-through' : ''}`}
                        title={job.name}
                      >
                        {job.name}
                      </p>
                      <p className="text-xs text-slate-400 dark:text-slate-500">
                        {job.stage ? `${job.stage} · ` : ''}
                        {job.fraction != null ? `${Math.round(job.fraction * 100)}%` : '—'}
                        {' · говорящих: '}
                        {formatSpeakerSetting(job)}
                      </p>
                    </button>
                    {job.deleted ? (
                      <span className="whitespace-nowrap rounded-full bg-red-100 px-2 py-0.5 text-xs text-red-700 dark:bg-red-950/60 dark:text-red-300">
                        Удалено
                      </span>
                    ) : (
                      <span
                        className={`whitespace-nowrap rounded-full px-2 py-0.5 text-xs ${
                          STATUS_STYLES[job.status] ??
                          'bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300'
                        }`}
                      >
                        {STATUS_LABELS[job.status] ?? job.status}
                      </span>
                    )}
                    {job.deleted ? (
                      <>
                        <button
                          onClick={() => void restoreJob(job.id)}
                          title="Вернуть задачу в обычный список"
                          className="whitespace-nowrap rounded-md border border-emerald-300 px-3 py-1 text-xs text-emerald-700 hover:bg-emerald-50 dark:border-emerald-800 dark:text-emerald-300 dark:hover:bg-emerald-950/50"
                        >
                          Восстановить
                        </button>
                        <button
                          onClick={() => void purgeJob(job.id, job.name)}
                          title="Удалить задачу и её артефакты безвозвратно"
                          className="whitespace-nowrap rounded-md border border-red-300 px-3 py-1 text-xs text-red-700 hover:bg-red-50 dark:border-red-900 dark:text-red-300 dark:hover:bg-red-950/50"
                        >
                          Удалить навсегда
                        </button>
                      </>
                    ) : (
                      <>
                        {isLiveJob(job) && (
                          <button
                            onClick={() => void stopJob(job.id)}
                            title="Остановить обработку задачи"
                            className="whitespace-nowrap rounded-md border border-amber-300 px-3 py-1 text-xs text-amber-700 hover:bg-amber-50 dark:border-amber-800 dark:text-amber-300 dark:hover:bg-amber-950/50"
                          >
                            Остановить
                          </button>
                        )}
                        {(job.status === 'queued' ||
                          job.status === 'done' ||
                          job.status === 'error' ||
                          job.status === 'cancelled' ||
                          (job.status === 'running' && job.active === false)) && (
                          <button
                            onClick={() => void runJob(job.id)}
                            disabled={readinessBlocked}
                            title={blockedHint}
                            className="whitespace-nowrap rounded-md bg-emerald-600 px-3 py-1 text-xs text-white hover:bg-emerald-500 disabled:cursor-not-allowed disabled:opacity-40"
                          >
                            {job.status === 'cancelled' ? 'Запустить снова' : 'Запустить'}
                          </button>
                        )}
                        {(job.status !== 'running' || job.active === false) && (
                          <button
                            onClick={() => void deleteJob(job.id)}
                            title="Скрыть задачу (мягкое удаление, можно восстановить)"
                            className="whitespace-nowrap rounded-md border border-slate-300 px-2 py-1 text-xs text-slate-500 hover:bg-red-50 hover:text-red-600 dark:border-slate-700 dark:text-slate-400 dark:hover:bg-red-950/50 dark:hover:text-red-300"
                          >
                            Удалить
                          </button>
                        )}
                      </>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </section>
        </div>

        {activeJobId && (
          <section className="rounded-lg border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900">
            <div className="mb-3 flex items-center justify-between gap-3">
              <h2
                className="min-w-0 truncate font-medium"
                title={activeJob ? `Прогресс · ${activeJob.name}` : 'Прогресс'}
              >
                Прогресс{activeJob ? ` · ${activeJob.name}` : ''}
              </h2>
              {progressRunning && activeJobId && (
                <button
                  onClick={() => void stopJob(activeJobId)}
                  title="Остановить обработку задачи"
                  className="whitespace-nowrap rounded-md border border-amber-300 px-3 py-1 text-xs text-amber-700 hover:bg-amber-50 dark:border-amber-800 dark:text-amber-300 dark:hover:bg-amber-950/50"
                >
                  Остановить
                </button>
              )}
            </div>
            <div className="mb-2 h-2 w-full overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
              <div
                className={`h-full rounded-full bg-blue-500 transition-all ${
                  progressRunning && progress?.progress_percent == null ? 'animate-pulse' : ''
                }`}
                style={{ width: `${Math.round(overallPercent)}%` }}
              />
            </div>
            <p className="mb-1 text-sm text-slate-600 dark:text-slate-300">
              {progress
                ? STATUS_LABELS[progress.status] ?? progress.status
                : 'Ожидание...'}
              {progress?.message ? ` — ${progress.message}` : ''}
            </p>
            <ProgressSummary progress={progress} running={progressRunning} />
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
              running={progressRunning}
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
                className="w-full rounded-md border border-slate-300 px-3 py-1.5 text-sm focus:border-blue-400 focus:outline-none sm:ml-auto sm:w-64 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100 dark:placeholder-slate-500"
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
              sourceName={activeJob?.name}
              onSaveText={(entry: Entry, text) => {
                const index = result.entries.indexOf(entry)
                if (index < 0) return Promise.resolve()
                return patchTranscript(activeJobId, { edits: [{ index, text }] })
              }}
              onResetText={(entry: Entry) => {
                const index = result.entries.indexOf(entry)
                if (index < 0) return Promise.resolve()
                return patchTranscript(activeJobId, { resets: [index] })
              }}
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
