import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import {
  api,
  describeApply,
  errorMessage,
  fetchSummaryPrompts,
  isTerminal,
  type ApplyNamesResponse,
  type AsrDeviceInfo,
  type AssignSpeakerResponse,
  type ConfigInfo,
  type DoctorReport,
  type Entry,
  type ExtraSpeakerResponse,
  type FileItem,
  type Job,
  type JobDetails,
  type JobEvent,
  type LibraryWindow,
  type ProtocolResponse,
  type ReassignRequest,
  type ReassignResponse,
  type SampleMeta,
  type SplitEntryResponse,
  type StageTime,
  type SummaryPrompt,
  type TranscriptEditsRequest,
  type TranscriptResult,
  type VoiceInfo,
  type WebSettings,
} from '../api'
import { useActionProgress } from '../actionProgress'
import { STAGE_KEYS } from '../components/StageTimes'
import {
  STAGES,
  formatSpeakerSetting,
  isLiveJob,
  parseSpeakerCount,
  splitPart,
} from './jobUtils'

const SSE_RECONNECT_DELAY_MS = 2000
const JOBS_POLL_INTERVAL_MS = 5000

export function useAppController() {
  const [version, setVersion] = useState<string>('')
  const [config, setConfig] = useState<ConfigInfo | null>(null)
  const [asrDevice, setAsrDevice] = useState<AsrDeviceInfo | null>(null)
  const [doctor, setDoctor] = useState<DoctorReport | null>(null)
  const [doctorLoading, setDoctorLoading] = useState(false)
  const [doctorError, setDoctorError] = useState<string | null>(null)
  const [files, setFiles] = useState<FileItem[]>([])
  const [showProcessed, setShowProcessed] = useState(false)
  const showProcessedRef = useRef(false)
  const [jobs, setJobs] = useState<Job[]>([])
  const jobsRef = useRef<Job[]>([])
  const [showDeleted, setShowDeleted] = useState(false)
  const showDeletedRef = useRef(false)
  const [activeJobId, setActiveJobId] = useState<string | null>(null)
  const [runSeq, setRunSeq] = useState(0)
  const [progress, setProgress] = useState<JobEvent | null>(null)
  const [result, setResult] = useState<TranscriptResult | null>(null)
  const [samplesMeta, setSamplesMeta] = useState<Record<string, SampleMeta>>({})
  const [speakerUndoAvailable, setSpeakerUndoAvailable] = useState(false)
  const [protocol, setProtocol] = useState<ProtocolResponse | null>(null)
  const [protocolBusy, setProtocolBusy] = useState(false)
  const [protocolError, setProtocolError] = useState<string | null>(null)
  const [summaryPrompts, setSummaryPrompts] = useState<SummaryPrompt[]>([])
  const [protocolPromptId, setProtocolPromptId] = useState<number | ''>('')
  const [applyBusy, setApplyBusy] = useState(false)
  const [applyNotice, setApplyNotice] = useState<{
    kind: 'info' | 'error'
    text: string
  } | null>(null)
  const {
    run: actionRun,
    runTask: runActionTask,
    reset: resetActionProgress,
  } = useActionProgress()
  const [exportFormat, setExportFormat] = useState('txt')
  const [summary, setSummary] = useState<string | null>(null)
  const [query, setQuery] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [speakerCounts, setSpeakerCounts] = useState<Record<string, string>>({})
  const [speakerMins, setSpeakerMins] = useState<Record<string, string>>({})
  const [speakerMaxs, setSpeakerMaxs] = useState<Record<string, string>>({})
  const [stageTimes, setStageTimes] = useState<StageTime[]>([])
  const [finalTotalSeconds, setFinalTotalSeconds] = useState<number | null>(null)
  const [totalStartedAt, setTotalStartedAt] = useState<number | null>(null)
  const [liveStage, setLiveStage] = useState<string | null>(null)
  const [liveStageStartedAt, setLiveStageStartedAt] = useState<number | null>(null)
  const liveStageRef = useRef<string | null>(null)
  const lastEventSeqRef = useRef<number>(0)
  const lastElapsedRef = useRef<number>(0)
  const lastStageElapsedRef = useRef<number>(0)
  const exportDefaultApplied = useRef(false)

  const refreshJobs = useCallback(
    async (options?: { silent?: boolean }): Promise<Job[] | null> => {
      try {
        const suffix = showDeletedRef.current ? '?include_deleted=true' : ''
        const list = await api<Job[]>(`/api/jobs${suffix}`)
        jobsRef.current = list
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

  const refreshAsrDevice = useCallback(async () => {
    try {
      setAsrDevice(await api<AsrDeviceInfo>('/api/asr/device'))
    } catch {
      setAsrDevice(null)
    }
  }, [])

  const refreshSummaryPrompts = useCallback(async () => {
    try {
      const data = await fetchSummaryPrompts()
      setSummaryPrompts(data.prompts)
      setProtocolPromptId(data.active_id ?? '')
    } catch {
      setSummaryPrompts([])
      setProtocolPromptId('')
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

  const handleModelsChanged = useCallback(() => {
    void refreshDoctor()
    void refreshFiles()
  }, [refreshDoctor, refreshFiles])

  const handleWizardChanged = useCallback(() => {
    void refreshDoctor()
  }, [refreshDoctor])

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
      lastElapsedRef.current = details.total_seconds ?? 0
      lastStageElapsedRef.current = details.stage_elapsed ?? 0
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
    void refreshAsrDevice()
    void refreshSummaryPrompts()
  }, [refreshFiles, refreshJobs, refreshDoctor, refreshAsrDevice, refreshSummaryPrompts])

  useEffect(() => {
    const timer = window.setInterval(() => {
      const hasActive = jobsRef.current.some((job) => !isTerminal(job.status))
      if (hasActive) void refreshJobs({ silent: true })
    }, JOBS_POLL_INTERVAL_MS)
    return () => window.clearInterval(timer)
  }, [refreshJobs])

  useEffect(() => {
    if (exportDefaultApplied.current) return
    const first = config?.export_formats?.[0]
    if (!first) return
    exportDefaultApplied.current = true
    setExportFormat(first)
  }, [config])

  useEffect(() => {
    if (!activeJobId) return
    lastEventSeqRef.current = 0
    lastElapsedRef.current = 0
    lastStageElapsedRef.current = 0
    let disposed = false
    let source: EventSource | null = null
    let retryTimer: number | undefined

    const clearRetry = () => {
      if (retryTimer !== undefined) {
        window.clearTimeout(retryTimer)
        retryTimer = undefined
      }
    }

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

      const finish = (status: string) => {
        stream.close()
        setLiveStage(null)
        setLiveStageStartedAt(null)
        liveStageRef.current = null
        void refreshJobs()
        void refreshJobTiming(activeJobId)
        if (status === 'done') {
          void refreshFiles()
          void loadResult(activeJobId)
        }
      }

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
        if (typeof event.seq === 'number') {
          if (event.seq <= lastEventSeqRef.current) return
          lastEventSeqRef.current = event.seq
        }
        setProgress(event)
        if (event.stage_times) setStageTimes(event.stage_times)
        if (isTerminal(event.status)) {
          finish(event.status)
          return
        }
        if (event.elapsed != null && event.elapsed >= lastElapsedRef.current) {
          lastElapsedRef.current = event.elapsed
          setTotalStartedAt(Date.now() - event.elapsed * 1000)
        }
        if (event.stage !== liveStageRef.current) {
          liveStageRef.current = event.stage
          setLiveStage(event.stage)
          const stageElapsed = event.stage_elapsed ?? 0
          lastStageElapsedRef.current = stageElapsed
          setLiveStageStartedAt(Date.now() - stageElapsed * 1000)
        } else if (
          event.stage_elapsed != null &&
          event.stage_elapsed >= lastStageElapsedRef.current
        ) {
          lastStageElapsedRef.current = event.stage_elapsed
          setLiveStageStartedAt(Date.now() - event.stage_elapsed * 1000)
        }
      }

      stream.onerror = () => {
        if (disposed) return
        void (async () => {
          const list = await refreshJobs({ silent: true })
          if (disposed) return
          if (list == null) {
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
            setProgress({
              stage: current.stage ?? current.status,
              fraction: current.fraction,
              message: '',
              status: current.status,
              active: current.active,
              stage_times: current.stage_times,
              planned_stages: current.planned_stages,
              failed_stage: current.failed_stage,
              progress_percent: current.progress_percent,
              eta_seconds: current.eta_seconds,
              eta_by_stage: current.eta_by_stage,
              health: current.health,
            })
            finish(current.status)
            return
          }
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
    async (path: string): Promise<string | null> => {
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
        return null
      }
      if (minSpeakers != null && maxSpeakers != null && minSpeakers > maxSpeakers) {
        setError('Минимум говорящих не может быть больше максимума')
        return null
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
        await api<Job>(`/api/jobs/${job.id}/run`, { method: 'POST' })
        await refreshJobs()
        return job.id
      } catch (cause) {
        setError(errorMessage(cause))
        return null
      }
    },
    [refreshJobs, resetTiming, speakerCounts, speakerMins, speakerMaxs],
  )

  const runJob = useCallback(
    async (jobId: string): Promise<boolean> => {
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
        return true
      } catch (cause) {
        setError(errorMessage(cause))
        return false
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
          duration: details.duration,
          stage_times: details.stage_times,
          planned_stages: details.planned_stages,
          failed_stage: details.failed_stage,
        })
        setStageTimes(details.stage_times ?? [])
        setFinalTotalSeconds(details.total_seconds)
        lastElapsedRef.current = details.total_seconds ?? 0
        lastStageElapsedRef.current = details.stage_elapsed ?? 0
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
      setSpeakerUndoAvailable(false)
      await refreshSamples(jobId)
    },
    [refreshSamples],
  )

  const reassignSpeaker = useCallback(
    async (
      jobId: string,
      speakerId: string,
      body: ReassignRequest,
    ): Promise<ReassignResponse> => {
      const response = await api<ReassignResponse>(
        `/api/jobs/${jobId}/speakers/${encodeURIComponent(speakerId)}/reassign`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        },
      )
      setResult(response.result)
      setSpeakerUndoAvailable(true)
      await refreshSamples(jobId)
      return response
    },
    [refreshSamples],
  )

  const undoSpeakers = useCallback(
    async (jobId: string) => {
      const restored = await api<TranscriptResult>(`/api/jobs/${jobId}/speakers/undo`, {
        method: 'POST',
      })
      setResult(restored)
      setSpeakerUndoAvailable(false)
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

  const assignSpeaker = useCallback(
    async (
      jobId: string,
      entries: Entry[],
      target: { speakerId?: string; newName?: string },
    ) => {
      const indexes = entries
        .map((entry) => (result ? result.entries.indexOf(entry) : -1))
        .filter((index) => index >= 0)
      if (indexes.length === 0) {
        throw new Error('Не удалось определить выбранные реплики')
      }
      const body: Record<string, unknown> = { indexes }
      if (target.newName) body.new_name = target.newName
      else body.target_speaker_id = target.speakerId
      const response = await api<AssignSpeakerResponse>(
        `/api/jobs/${jobId}/transcript/assign-speaker`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        },
      )
      setResult(response.result)
      setSpeakerUndoAvailable(true)
    },
    [result],
  )

  const splitEntry = useCallback(
    async (
      jobId: string,
      entry: Entry,
      boundary: number,
      first: { speakerId?: string; newName?: string },
      second: { speakerId?: string; newName?: string },
    ) => {
      const index = result ? result.entries.indexOf(entry) : -1
      if (index < 0) throw new Error('Не удалось определить выбранную реплику')
      const response = await api<SplitEntryResponse>(
        `/api/jobs/${jobId}/transcript/split`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            index,
            boundary,
            first: splitPart(first),
            second: splitPart(second),
          }),
        },
      )
      setResult(response.result)
      setSpeakerUndoAvailable(true)
    },
    [result],
  )

  const editExtraSpeaker = useCallback(
    async (
      jobId: string,
      entries: Entry[],
      target: { speakerId?: string; newName?: string },
      remove = false,
    ) => {
      const indexes = entries
        .map((entry) => (result ? result.entries.indexOf(entry) : -1))
        .filter((index) => index >= 0)
      if (indexes.length === 0) {
        throw new Error('Не удалось определить выбранные реплики')
      }
      const body: Record<string, unknown> = { indexes, remove }
      if (target.newName) body.new_name = target.newName
      else body.target_speaker_id = target.speakerId
      const response = await api<ExtraSpeakerResponse>(
        `/api/jobs/${jobId}/transcript/extra-speaker`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        },
      )
      setResult(response.result)
      setSpeakerUndoAvailable(true)
    },
    [result],
  )

  const applyNames = useCallback(
    async (jobId: string): Promise<ApplyNamesResponse> => {
      const response = await runActionTask(
        'enrollment',
        (actionId) =>
          api<ApplyNamesResponse>(`/api/jobs/${jobId}/apply-names`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-Action-Id': actionId },
            body: JSON.stringify({}),
          }),
        (value) => ({
          text: describeApply(value, value.result.speakers.length),
          error: value.error != null,
        }),
      )
      setResult(response.result)
      await refreshSamples(jobId)
      setSpeakerUndoAvailable(false)
      return response
    },
    [refreshSamples, runActionTask],
  )

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

  useEffect(() => {
    setApplyNotice(null)
    setSpeakerUndoAvailable(false)
    resetActionProgress()
  }, [activeJobId, resetActionProgress])

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

  const generateProtocol = useCallback(
    async (jobId: string) => {
      setProtocolBusy(true)
      setProtocolError(null)
      try {
        const response = await runActionTask(
          'protocol',
          (actionId) =>
            api<ProtocolResponse>(`/api/jobs/${jobId}/protocol`, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
                'X-Action-Id': actionId,
              },
              body: JSON.stringify({
                prompt_id: typeof protocolPromptId === 'number' ? protocolPromptId : null,
              }),
            }),
          (value) => ({
            text: value.summary
              ? 'Протокол сформирован, резюме готово'
              : 'Протокол сформирован',
          }),
        )
        setProtocol(response)
        setSummary(response.summary)
      } catch (cause) {
        setProtocolError(errorMessage(cause))
      } finally {
        setProtocolBusy(false)
      }
    },
    [runActionTask, protocolPromptId],
  )

  const filteredEntries = useMemo(() => {
    if (!result) return []
    const needle = query.trim().toLowerCase()
    if (!needle) return result.entries
    return result.entries.filter((entry) => entry.text.toLowerCase().includes(needle))
  }, [result, query])

  const activeJob = jobs.find((job) => job.id === activeJobId) ?? null

  const plannedStages = useMemo(() => {
    const server = progress?.planned_stages?.length
      ? progress.planned_stages
      : activeJob?.planned_stages?.length
        ? activeJob.planned_stages
        : null
    return server ?? STAGES.map((stage) => stage.key)
  }, [progress, activeJob])

  const failedStage = useMemo(() => {
    const status = progress?.status ?? activeJob?.status
    if (status !== 'error') return null
    const candidates = [
      progress?.failed_stage,
      activeJob?.failed_stage,
      activeJob?.stage,
      progress?.stage,
    ]
    for (const candidate of candidates) {
      if (candidate && STAGE_KEYS.includes(candidate)) return candidate
    }
    return null
  }, [progress, activeJob])

  const progressRunning =
    progress != null && !isTerminal(progress.status) && progress.active !== false
  const overallPercent =
    progress?.progress_percent ?? (progress?.fraction != null ? progress.fraction * 100 : 0)
  const readinessBlocked = (doctor?.summary.critical_failures ?? 0) > 0
  const blockedHint = readinessBlocked
    ? 'Запуск заблокирован: сначала устраните критичные проблемы Готовности'
    : undefined

  const handleSettingsSaved = useCallback(
    (saved: WebSettings) => {
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
      void refreshAsrDevice()
    },
    [refreshDoctor, refreshAsrDevice],
  )

  return {
    version,
    config,
    asrDevice,
    doctor,
    doctorLoading,
    doctorError,
    files,
    showProcessed,
    toggleProcessed,
    jobs,
    showDeleted,
    toggleDeleted,
    refreshJobs,
    activeJobId,
    activeJob,
    progress,
    result,
    setResult,
    samplesMeta,
    speakerUndoAvailable,
    protocol,
    protocolBusy,
    protocolError,
    summaryPrompts,
    protocolPromptId,
    setProtocolPromptId,
    refreshSummaryPrompts,
    applyBusy,
    applyNotice,
    setApplyNotice,
    actionRun,
    resetActionProgress,
    exportFormat,
    setExportFormat,
    summary,
    query,
    setQuery,
    error,
    setError,
    speakerCounts,
    setSpeakerCounts,
    speakerMins,
    setSpeakerMins,
    speakerMaxs,
    setSpeakerMaxs,
    stageTimes,
    finalTotalSeconds,
    totalStartedAt,
    liveStage,
    liveStageStartedAt,
    filteredEntries,
    plannedStages,
    failedStage,
    progressRunning,
    overallPercent,
    readinessBlocked,
    blockedHint,
    refreshFiles,
    refreshDoctor,
    refreshAsrDevice,
    recheckDoctor,
    openJob,
    runJob,
    stopJob,
    deleteJob,
    restoreJob,
    purgeJob,
    enqueue,
    upload,
    deleteFile,
    restoreFile,
    runApplyNames,
    saveToLibrary,
    generateProtocol,
    patchSpeakers,
    reassignSpeaker,
    undoSpeakers,
    patchTranscript,
    assignSpeaker,
    splitEntry,
    editExtraSpeaker,
    applyNames,
    handleModelsChanged,
    handleWizardChanged,
    handleSettingsSaved,
    formatSpeakerSetting,
  }
}
