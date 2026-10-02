// Общие типы API и утилиты для SPA транскрибера.

export type ConfigInfo = {
  input_dir: string
  output_dir: string
  export_formats: string[]
  llm_enabled: boolean
  glossary_enabled: boolean
  voices_dir: string
  /** Провайдер LLM: llama (локальный) или openai (внешний API). */
  llm_provider: string
  /** true — включён внешний провайдер: текст уходит за пределы машины. */
  llm_external: boolean
}

export type FileItem = {
  name: string
  path: string
  size: number
  duration: number | null
  /** Файл успешно обработан и убран из основного списка (#16). */
  processed: boolean
}

export type StageTime = {
  stage: string
  seconds: number
  cached: boolean
}

/** «Здоровье» задачи: ok — идёт нормально, slow — замедление, stalled — нет активности. */
export type HealthStatus = 'ok' | 'slow' | 'stalled'

export type JobHealth = {
  status: HealthStatus
  /** Сколько секунд назад последний раз обновлялось состояние задачи. */
  last_update_seconds: number | null
  /** Причина замедления/зависания по-русски (пусто для ok) — tooltip (#36). */
  reason: string
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
  min_speakers: number | null
  max_speakers: number | null
  stage_started_at: string | null
  stage_times: StageTime[]
  /** Время последнего изменения задачи (ISO) — для оценки «здоровья». */
  updated_at: string | null
  /** Задача мягко удалена: скрыта из обычного списка, артефакты сохранены (#30). */
  deleted: boolean
  /** Момент мягкого удаления (ISO) или null. */
  deleted_at: string | null
  total_seconds: number | null
  stage_elapsed: number | null
  /** Обрабатывается ли задача текущим воркером (false — осиротевшая running). */
  active: boolean
  /** Сводный процент прогона с учётом весов стадий. */
  progress_percent: number
  /** Ожидаемый остаток всего прогона в секундах (null — нет данных/истории). */
  eta_seconds: number | null
  /** Остаток по стадиям (секунды) или null. */
  eta_by_stage: Record<string, number> | null
  /** «Здоровье» задачи (null вне статуса running). */
  health: JobHealth | null
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
  /** Текст изменён вручную в веб-интерфейсе (#26). */
  edited: boolean
  /** Исходный текст до первой ручной правки (для сброса). */
  original_text: string | null
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

/** Качество образца голоса: метрики, флаги и готовые предупреждения (#29). */
export type VoiceQuality = {
  duration: number
  speech_seconds: number
  trimmed_seconds: number
  speech_ratio: number
  rms_dbfs: number | null
  peak: number
  too_short: boolean
  clipped: boolean
  low_energy: boolean
  mostly_non_speech: boolean
  ok: boolean
  warnings: string[]
}

export type VoiceInfo = {
  name: string
  filename: string
  duration: number
  size: number
  quality?: VoiceQuality
}

/** Группа образцов одного человека (имя → список образцов). */
export type VoiceGroup = { name: string; count: number; samples: VoiceInfo[] }

/** Один вариант прослушивания говорящего: окно исходного аудио [start, end). */
export type SampleVariant = { start: number; end: number; duration: number; score: number }

export type SpeakerVariants = { speaker_id: string; variants: SampleVariant[] }

/** Окно исходного аудио для сохранения в библиотеку как образец. */
export type LibraryWindow = { start: number; end: number }

/** Одно изменение говорящего реплики при переназначении окна (#40/#41). */
export type SpeakerChange = {
  index: number
  before_speaker_id: string | null
  after_speaker_id: string | null
  before_extra_ids: string[]
  after_extra_ids: string[]
}

/** Тело ``POST /api/jobs/{id}/speakers/{sid}/reassign`` (#40/#41). */
export type ReassignRequest = {
  start: number
  end: number
  /** Перенести на существующего говорящего (#40). */
  target_speaker_id?: string | null
  /** Создать нового говорящего с этим именем и назначить ему окно (#41). */
  new_name?: string | null
  /** Не замещать основного целиком, а добавить целевого сов-говорящим (#41). */
  split?: boolean
}

/** Ответ ``POST .../reassign``: результат, изменения и созданный говорящий. */
export type ReassignResponse = {
  result: TranscriptResult
  changes: SpeakerChange[]
  target_speaker_id: string
  created_speaker: { id: string; display_name: string } | null
  speaker_id: string
}

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
  /** Длительность исходной записи в секундах (#43). */
  duration?: number | null
  /** Сводный процент прогона, ETA и «здоровье» (см. #15/#24). */
  progress_percent?: number
  eta_seconds?: number | null
  eta_by_stage?: Record<string, number> | null
  health?: JobHealth | null
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
  /** Системные уведомления о завершении/ошибке/отмене задачи (#35). */
  notifications: boolean
  asr_backend: string
  device: string
  whisper_cpp_model: string
  whisper_cpp_binary: string
  llm_model: string
  llm_binary: string
  /** Провайдер LLM: llama (локальный) или openai (внешний API). */
  llm_provider: string
  /** Базовый URL и имя модели внешнего OpenAI-совместимого API. */
  llm_base_url: string
  llm_model_name: string
  pyannote_local_model: string
  /** GigaAM v3 (onnx-asr): имя модели, локальный каталог снимка и квантизация. */
  gigaam_model: string
  gigaam_model_path: string
  gigaam_quantization: string
  /** Резать длинное аудио встроенным VAD onnx-asr. */
  gigaam_vad: boolean
  input_dir: string
  output_dir: string
  glossary_db_path: string
  voices_dir_resolved: string
  hf_token_set: boolean
  hf_token_masked: string | null
  /** Секрет API-ключа внешней LLM: наружу отдаётся лишь флаг и маска. */
  llm_api_key_set: boolean
  llm_api_key_masked: string | null
}

export const ASR_BACKENDS = ['faster-whisper', 'whisper-cpp', 'gigaam'] as const
export const DEVICES = ['auto', 'cpu', 'cuda'] as const
export const LLM_PROVIDERS = ['llama', 'openai'] as const

export type ModelKind = 'whisper-cpp' | 'llm' | 'pyannote' | 'gigaam'
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

export type LlmCheckStatus = 'ok' | 'no_url' | 'error'

export type LlmCheckResult = {
  status: LlmCheckStatus
  message: string
  models: string[]
}

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

/** Тело ``POST /api/glossary/quick`` — добавление термина из выделения (#17). */
export type GlossaryQuickRequest = {
  term: string
  canonical?: string | null
  variant?: string | null
  note?: string | null
  source?: string | null
}

/** Тело ``PATCH /api/jobs/{id}/transcript`` — ручная правка текста реплик (#26). */
export type TranscriptEditsRequest = {
  edits?: { index: number; text: string }[]
  resets?: number[]
}

/** Одна замена термина при применении глоссария (#32). */
export type GlossaryReplacement = { before: string; after: string }

/** Отчёт по одной реплике при применении глоссария (#32). */
export type GlossaryApplyDetail = {
  index: number
  before: string
  after: string
  replacements: GlossaryReplacement[]
}

/** Ответ ``POST /api/jobs/{id}/apply-glossary`` (#32). */
export type GlossaryApplyResponse = {
  result: TranscriptResult
  replacements: number
  details: GlossaryApplyDetail[]
  skipped_edited: number
  terms: number
  error: string | null
}

/** Предложение редакторской правки текущей стенограммы (#51). */
export type TextSuggestion = {
  id: string
  index: number
  start: number
  end: number
  before: string
  after: string
  kind: string
  reason: string
}

/** Тело ``POST /api/jobs/{id}/correct-text`` (#51). */
export type CorrectTextRequest = {
  dry_run?: boolean
  selection?: string[] | null
  fix_common?: boolean
  check_spelling?: boolean
  respect_edited?: boolean
}

/** Ответ ``POST /api/jobs/{id}/correct-text`` (#51). */
export type CorrectTextResponse = {
  result: TranscriptResult
  suggestions: TextSuggestion[]
  applied: TextSuggestion[]
  applied_count: number
  skipped_edited: number
  error: string | null
}

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

/**
 * Длительность записи без десятых: `M:SS`, для часа и больше — `H:MM:SS`
 * (#43). Отличается от {@link formatTime}, который показывает десятые доли.
 */
export function formatClock(seconds: number | null): string {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return '—'
  const total = Math.round(seconds)
  const hours = Math.floor(total / 3600)
  const minutes = Math.floor((total % 3600) / 60)
  const secs = total % 60
  if (hours) {
    return `${hours}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`
  }
  return `${minutes}:${String(secs).padStart(2, '0')}`
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

//: Ключ сортировки говорящих: числовой суффикс id (``SPEAKER_02`` → 2), затем
//: сам id. Говорящие без числа уходят в конец, чтобы порядок был стабильным и
//: предсказуемым (00, 01, 02, …), а не зависел от порядка диаризации.
function speakerSortKey(id: string): [number, string] {
  const match = id.match(/(\d+)(?!.*\d)/)
  return [match ? Number(match[1]) : Number.MAX_SAFE_INTEGER, id]
}

export function sortSpeakers(speakers: SpeakerInfo[]): SpeakerInfo[] {
  return [...speakers].sort((a, b) => {
    const [aIndex, aId] = speakerSortKey(a.id)
    const [bIndex, bId] = speakerSortKey(b.id)
    return aIndex !== bIndex ? aIndex - bIndex : aId.localeCompare(bId)
  })
}

/**
 * Полный список говорящих результата — объединение `result.speakers` и id,
 * на которые ссылаются реплики (основные и дополнительные). Гарантирует, что
 * говорящий, упомянутый только как участник наложения (`extra_speaker_ids`),
 * не потеряется, даже если его нет в `result.speakers` (старые результаты).
 * Порядок — по возрастанию id.
 */
export function collectSpeakers(result: TranscriptResult): SpeakerInfo[] {
  const byId = new Map<string, SpeakerInfo>()
  for (const speaker of result.speakers) byId.set(speaker.id, speaker)
  for (const entry of result.entries) {
    for (const id of [entry.speaker_id, ...entry.extra_speaker_ids]) {
      if (id && !byId.has(id)) {
        byId.set(id, { id, display_name: id, has_sample: false })
      }
    }
  }
  return sortSpeakers([...byId.values()])
}

/**
 * Человекочитаемый итог применения имён по голосу: сколько сопоставлено и
 * лучшие недобранные пары (как в TUI). `total` — число говорящих результата.
 */
export function describeApply(response: ApplyNamesResponse, total: number): string {
  if (response.error) return response.error
  const matched = Object.entries(response.matched)
  const parts =
    matched.length > 0
      ? [
          `Сопоставлено ${matched.length} из ${total}: ` +
            matched.map(([sid, name]) => `${sid} → ${name}`).join(', '),
        ]
      : [`Имена по голосу не сопоставлены (0 из ${total}, ниже порога)`]
  const candidates = Object.entries(response.best_candidates)
    .sort((a, b) => b[1].score - a[1].score)
    .slice(0, 3)
    .map(
      ([sid, candidate]) =>
        `${sid} ≈ «${candidate.name}» ${candidate.score.toFixed(2)} < ${response.threshold.toFixed(2)}`,
    )
  if (candidates.length > 0) parts.push(`не добрали: ${candidates.join('; ')}`)
  return parts.join(' · ')
}
