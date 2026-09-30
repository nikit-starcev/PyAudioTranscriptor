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
  input_dir: string
  output_dir: string
  glossary_db_path: string
  voices_dir_resolved: string
  hf_token_set: boolean
  hf_token_masked: string | null
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

export function formatDuration(seconds: number | null): string {
  return seconds == null ? '—' : formatTime(seconds)
}

export function speakerName(speakers: SpeakerInfo[], id: string | null): string {
  if (id == null) return '—'
  return speakers.find((speaker) => speaker.id === id)?.display_name ?? id
}
