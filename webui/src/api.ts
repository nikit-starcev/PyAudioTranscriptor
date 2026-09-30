// Общие типы API и утилиты для SPA транскрибера.

export type ConfigInfo = {
  input_dir: string
  output_dir: string
  export_formats: string[]
  llm_enabled: boolean
  glossary_enabled: boolean
  voices_dir: string
}

export type FileItem = {
  name: string
  path: string
  size: number
  duration: number | null
}

export type StageTime = {
  stage: string
  seconds: number
  cached: boolean
}

export type Job = {
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
  num_speakers: number | null
  stage_started_at: string | null
  stage_times: StageTime[]
  total_seconds: number | null
  stage_elapsed: number | null
  /** Обрабатывается ли задача текущим воркером (false — осиротевшая running). */
  active: boolean
}

export type Summary = {
  language: string | null
  duration: number | null
  entries: number
  speakers: number
  samples: number
}

export type JobDetails = Job & { summary: Summary | null }

export type SpeakerInfo = { id: string; display_name: string; has_sample: boolean }

export type Entry = {
  start: number
  end: number
  speaker_id: string | null
  extra_speaker_ids: string[]
  speaker_confidence: number | null
  low_speaker_confidence: boolean
  text: string
  low_confidence: boolean
  overlap: boolean
}

export type Mark = { key: string; symbol: string; label: string }

export type TranscriptResult = {
  language: string | null
  duration: number
  speakers: SpeakerInfo[]
  entries: Entry[]
  marks: Mark[]
  summary: string | null
  samples: Record<string, string>
}

export type SampleMeta = { speaker_id: string; duration: number; size: number }

export type VoiceInfo = { name: string; filename: string; duration: number; size: number }

export type BestCandidate = { name: string; score: number }

export type ApplyNamesResponse = {
  result: TranscriptResult
  matched: Record<string, string>
  best_candidates: Record<string, BestCandidate>
  threshold: number
  error: string | null
}

export type JobEvent = {
  stage: string
  fraction: number | null
  message: string
  status: string
  active?: boolean
  elapsed?: number | null
  stage_elapsed?: number | null
  stage_times?: StageTime[]
}

export type WebSettings = {
  glossary_enabled: boolean
  glossary_db: string
  voices_dir: string
  export_formats: string[]
  llm_enabled: boolean
  llm_summary: boolean
  denoise: boolean
  mark_overlap: boolean
  normalize_text: boolean
  clean_artifacts: boolean
  protocol_auto: boolean
  asr_backend: string
  device: string
  whisper_cpp_model: string
  whisper_cpp_binary: string
  llm_model: string
  llm_binary: string
  pyannote_local_model: string
  input_dir: string
  output_dir: string
  glossary_db_path: string
  voices_dir_resolved: string
  hf_token_set: boolean
  hf_token_masked: string | null
}

export const ASR_BACKENDS = ['faster-whisper', 'whisper-cpp'] as const
export const DEVICES = ['auto', 'cpu', 'cuda'] as const

export type ModelKind = 'whisper-cpp' | 'llm' | 'pyannote'
export type ModelDownloadStatus = 'idle' | 'downloading' | 'done' | 'error' | 'cancelled'

export type ModelLocalStatus = {
  present: boolean
  size: number
  expected_size: number
  missing_files: string[]
  path: string
  partial: boolean
}

export type ModelDownloadState = {
  status: ModelDownloadStatus
  fraction: number | null
  bytes_done: number
  total: number
  message: string
  error: string | null
}

export type ModelInfo = {
  id: string
  kind: ModelKind
  title: string
  repo: string
  files: string[]
  approx_size: number
  note: string
  gated: boolean
  setting_key: string
  target_dir: string
  target_path: string
  primary_path: string
  status: ModelLocalStatus
  download: ModelDownloadState
}

export type ModelsResponse = {
  models: ModelInfo[]
  disk: { free: number; models_dir: string }
}

export type ModelEvent = {
  id: string
  status: ModelDownloadStatus
  fraction: number | null
  bytes_done: number
  total: number
  message: string
}

export type HardwareOption = {
  id: string
  label: string
  asr_backend: string
  device: string
  note: string
}

export type SetupStepStatus = 'ok' | 'todo' | 'warn'

export type SetupStep = {
  id: string
  title: string
  description: string
  status: SetupStepStatus
  action: string
  required?: string[] | boolean
  missing?: string[]
  set?: boolean
}

export type BinaryRequirement = {
  key: string
  label: string
  needed: boolean
  available: boolean
  status: 'ok' | 'fail'
  instructions: string
  links: string[]
}

export type SetupPlan = {
  hardware: { options: HardwareOption[]; current: string }
  steps: SetupStep[]
  required_models: string[]
  missing_models: string[]
  binaries: BinaryRequirement[]
  hf_token: { required: boolean; set: boolean }
  summary: Record<string, number>
}

export type DoctorStatus = 'ok' | 'warn' | 'fail'

export type DoctorCheck = {
  id: string
  label: string
  status: DoctorStatus
  critical: boolean
  detail: string
  hint: string
  links: string[]
}

export type DoctorSummary = {
  ok: number
  warn: number
  fail: number
  critical_failures: number
}

export type DoctorReport = {
  checks: DoctorCheck[]
  summary: DoctorSummary
}

export type HfCheckStatus = 'ok' | 'no_token' | 'no_access' | 'error'

export type HfCheckResult = {
  status: HfCheckStatus
  message: string
  account: string | null
}

export const HF_TOKEN_URL = 'https://huggingface.co/settings/tokens'
export const PYANNOTE_MODEL_URL =
  'https://huggingface.co/pyannote/speaker-diarization-community-1'

export type GlossarySource = {
  name: string
  kind: string
  enabled: boolean
  count: number
  path: string | null
}

export type GlossaryEntry = {
  id: number
  canonical: string
  variant: string | null
  category: string | null
  note: string | null
  source: string | null
  enabled: boolean
}

export type GlossaryEntriesPage = { entries: GlossaryEntry[]; total: number }

export type GlossaryStats = { sources: number; entries: number; enabled: number }

export type GlossaryImportReport = {
  source: string
  kind: string
  added: number
  skipped: number
  total: number
  replaced: boolean
}

export type ProtocolResponse = {
  paths: string[]
  summary: string | null
  protocol: Record<string, string>
}

export const EXPORT_FORMATS = ['txt', 'docx', 'json', 'srt'] as const

export async function api<T>(url: string, init?: RequestInit): Promise<T> {
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

export function errorMessage(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause)
}

export function isTerminal(status: string): boolean {
  return status === 'done' || status === 'error' || status === 'cancelled'
}

export function formatTime(seconds: number): string {
  const minutes = Math.floor(seconds / 60)
  const rest = seconds - minutes * 60
  return `${minutes}:${rest.toFixed(1).padStart(4, '0')}`
}

export function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} Б`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} КБ`
  return `${(bytes / (1024 * 1024)).toFixed(1)} МБ`
}

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 Б'
  const units = ['Б', 'КБ', 'МБ', 'ГБ', 'ТБ']
  let value = bytes
  let unit = 0
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024
    unit += 1
  }
  return unit === 0 ? `${value} Б` : `${value.toFixed(1)} ${units[unit]}`
}

export function formatDuration(seconds: number | null): string {
  return seconds == null ? '—' : formatTime(seconds)
}

// Длительность стадии: миллисекунды, секунды или минуты — по величине.
export function formatStageTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return '—'
  if (seconds < 1) return `${Math.round(seconds * 1000)} мс`
  if (seconds < 60) return `${seconds.toFixed(1)} с`
  const minutes = Math.floor(seconds / 60)
  const rest = Math.round(seconds - minutes * 60)
  return rest ? `${minutes} мин ${rest} с` : `${minutes} мин`
}

export function speakerName(speakers: SpeakerInfo[], id: string | null): string {
  if (id == null) return '—'
  return speakers.find((speaker) => speaker.id === id)?.display_name ?? id
}
