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
