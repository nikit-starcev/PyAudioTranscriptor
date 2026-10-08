import type { Job } from '../api'
import type { BadgeTone } from '../components/ui'

export const STAGES: { key: string; label: string }[] = [
  { key: 'denoise', label: 'Шумоподавление' },
  { key: 'asr', label: 'Распознавание речи' },
  { key: 'diarization', label: 'Определение говорящих' },
  { key: 'merge', label: 'Объединение сегментов' },
  { key: 'clean', label: 'Очистка артефактов' },
  { key: 'correction', label: 'Автоисправление' },
  { key: 'llm', label: 'LLM-постобработка' },
  { key: 'export', label: 'Экспорт' },
]

export const STATUS_LABELS: Record<string, string> = {
  queued: 'В очереди',
  running: 'Обработка',
  done: 'Готово',
  error: 'Ошибка',
  cancelled: 'Отменено',
}

export function statusTone(status: string): BadgeTone {
  switch (status) {
    case 'running':
      return 'info'
    case 'done':
      return 'success'
    case 'error':
      return 'danger'
    case 'cancelled':
      return 'warn'
    default:
      return 'neutral'
  }
}

export function isLiveJob(job: { status: string; active?: boolean }): boolean {
  return job.status === 'running' && job.active !== false
}

export function parseSpeakerCount(raw: string | undefined): number | null | undefined {
  const trimmed = (raw ?? '').trim()
  if (trimmed === '') return null
  const value = Number(trimmed)
  if (!Number.isInteger(value) || value < 1) return undefined
  return value
}

export function formatSpeakerSetting(job: Job): string {
  if (job.num_speakers != null) return `${job.num_speakers}`
  if (job.min_speakers != null || job.max_speakers != null) {
    return `${job.min_speakers ?? '—'}–${job.max_speakers ?? '—'}`
  }
  return 'авто'
}

export function splitPart(target: { speakerId?: string; newName?: string }) {
  return target.newName
    ? { new_name: target.newName }
    : { speaker_id: target.speakerId ?? null }
}
