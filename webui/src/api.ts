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
  /** Примерная оценка времени обработки до запуска (#107). */
  estimate?: EstimateInfo | null
}

/** Примерная оценка времени обработки файла до запуска (#107). */
export type EstimateInfo = {
  /** Полное время обработки, сек (null — длительность/стадии неизвестны). */
  seconds: number | null
  /** Разбивка по стадиям, сек, или null. */
  by_stage: Record<string, number> | null
  /** true — оценка целиком по свежей истории (уверенная), false — приблизительная. */
  exact: boolean
  /** Есть ли свежая история прогонов (для пояснения «примерно»). */
  has_history: boolean
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
  /** Планируемые стадии конвейера в порядке выполнения (по конфигурации задачи). */
  planned_stages: string[]
  /** Стадия, на которой произошёл сбой (только при статусе ``error``). */
  failed_stage: string | null
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

/** Пословная метка времени реплики (#45), приходит при включённой стадии. */
export type WordTimestamp = {
  text: string
  start: number
  end: number
  probability: number | null
}

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
  /** Пословные таймкоды реплики (#45), если стадия включена. */
  words?: WordTimestamp[]
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

/** Тип совпадения дубликатов (#39): побайтово, по аудио или по эмбеддингу. */
export type VoiceDuplicateKind = 'exact' | 'audio' | 'embedding'

/** Участник группы дубликатов. */
export type VoiceDuplicateMember = {
  name: string
  filename: string
  duration: number
  size: number
}

/** Группа дубликатов с подсказкой «кого оставить» (#39). */
export type VoiceDuplicateGroup = {
  kind: VoiceDuplicateKind
  score: number
  names: string[]
  keep: string
  members: VoiceDuplicateMember[]
}

/** Отчёт аудита библиотеки голосов (``POST /api/voices/dedup``). */
export type VoiceDedupReport = {
  groups: VoiceDuplicateGroup[]
  scanned: number
  embeddings: boolean
  near_threshold: number
  embedding_threshold: number
  error: string | null
}

/** Похожий образец, найденный при добавлении нового (#39). */
export type VoiceSimilar = {
  name: string
  filename: string
  kind: string
  score: number
}

export type VoiceInfo = {
  name: string
  filename: string
  duration: number
  size: number
  quality?: VoiceQuality
  /** Предупреждение о похожих образцах при добавлении (#39). */
  similar?: VoiceSimilar[]
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

/** Цель принудительного назначения говорящего (#59): существующий или новый. */
export type AssignSpeakerTarget = { speakerId?: string; newName?: string }

/** Ответ ``POST /api/jobs/{id}/transcript/assign-speaker`` (#59). */
export type AssignSpeakerResponse = {
  result: TranscriptResult
  changes: SpeakerChange[]
  target_speaker_id: string
  created_speaker: { id: string; display_name: string } | null
  indexes: number[]
}

/** Говорящий одной части разрезаемой реплики (#78): существующий или новый. */
export type SplitPart = { speaker_id?: string | null; new_name?: string | null }

/** Тело ``POST /api/jobs/{id}/transcript/split`` (#78). */
export type SplitEntryRequest = {
  index: number
  /** Момент разреза в секундах аудио (строго внутри реплики). */
  boundary: number
  first: SplitPart
  second: SplitPart
}

/** Ответ ``POST .../transcript/split`` (#78). */
export type SplitEntryResponse = {
  result: TranscriptResult
  index: number
  boundary: number
  first_speaker_id: string
  second_speaker_id: string
  created_speakers: { id: string; display_name: string }[]
}

/** Тело ``POST /api/jobs/{id}/transcript/extra-speaker`` (#78). */
export type ExtraSpeakerRequest = {
  indexes: number[]
  target_speaker_id?: string | null
  new_name?: string | null
  /** Убрать целевого из участников наложения вместо добавления. */
  remove?: boolean
}

/** Ответ ``POST .../transcript/extra-speaker`` (#78). */
export type ExtraSpeakerResponse = {
  result: TranscriptResult
  changes: SpeakerChange[]
  target_speaker_id: string
  created_speaker: { id: string; display_name: string } | null
  indexes: number[]
  removed: boolean
}

/** Класс устройства распознавания речи (#72). */
export type AsrDeviceClass = 'gpu' | 'cpu' | 'unknown'

/** Устройство ASR для индикатора в UI (#72). */
export type AsrDeviceInfo = {
  backend: string
  device: AsrDeviceClass
  /** Готовая подпись: «whisper.cpp · GPU Vulkan0 (AMD Radeon RX 590)». */
  label: string
  accelerator: string | null
  name: string | null
  /** Пометка, что денойз и диаризация всегда идут на CPU. */
  note: string
  /** Сырые строки проб (для подсказки). */
  details: string[]
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
  /** Монотонный номер события (на задачу): дедуп повторно отданной истории. */
  seq?: number
  stage_times?: StageTime[]
  /** Планируемые стадии конвейера в порядке выполнения. */
  planned_stages?: string[]
  /** Стадия, на которой произошёл сбой (при статусе ``error``). */
  failed_stage?: string | null
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
  /** Сводить кластеры с одинаковым уверенным именем в одного говорящего. */
  merge_same_name_speakers: boolean
  normalize_text: boolean
  clean_artifacts: boolean
  /** Автоисправление опечаток (стадия correction, pymorphy3). */
  enable_correction: boolean
  protocol_auto: boolean
  /** Пословные таймстемпы (#45): слова с временами в результате. */
  word_timestamps: boolean
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
  /** Движок диаризации: auto | pyannote | nemo-speech | hybrid (#62). */
  diarization_engine: string
  /** NeMo-Speech.cpp: бинарник, каталог библиотек, модель и устройство (#62). */
  nemo_speech_binary: string
  nemo_speech_lib_path: string
  nemo_speech_model: string
  nemo_speech_device: string
  /** Оценщик числа говорящих и маршрутизация auto (#64). */
  diarization_estimate_enabled: boolean
  diarization_estimate_seconds: number
  diarization_estimate_threshold: number
  diarization_estimate_model: string
  diarization_route_max_speakers: number
  /** Гибридная диаризация (оконный EEND + глобальная склейка) (#64). */
  diarization_hybrid_enabled: boolean
  diarization_hybrid_window_seconds: number
  diarization_hybrid_overlap_seconds: number
  diarization_hybrid_min_speaker_seconds: number
  /** Linkage и порог пороговой ветки кластеризации гибрида (#88). */
  diarization_hybrid_linkage: string
  diarization_hybrid_threshold: number
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
export const DIARIZATION_ENGINES = ['auto', 'pyannote', 'nemo-speech', 'hybrid'] as const
export const NEMO_SPEECH_DEVICES = ['auto', 'vulkan', 'cpu'] as const

/** Найденный автодетектом бинарник nemo-speech (путь + каталог lib/ + проба). */
export type NemoSpeechCandidate = {
  binary: string
  lib_path: string | null
  source: string
  version: string | null
  devices: string[]
  has_vulkan: boolean
}

export type NemoSpeechDetectResponse = {
  candidates: NemoSpeechCandidate[]
  current: { binary: string; lib_path: string; model: string }
  recommended: string | null
  found: boolean
}

export type NemoSpeechDownloadStatus = 'idle' | 'downloading' | 'done' | 'error'

/** Состояние фоновой загрузки модели Sortformer. */
export type NemoSpeechDownloadState = {
  status: NemoSpeechDownloadStatus
  message: string
  fraction: number | null
  bytes_done: number
  total: number
  path: string | null
  error: string | null
}

/** Статус локальной модели Sortformer (кэш `~/.cache/nemo-speech`). */
export type NemoSpeechModelStatus = {
  model: string
  repo: string
  present: boolean
  path: string | null
  size: number
  files: string[]
  source: string
  download: NemoSpeechDownloadState
  binary: { configured: string; available: boolean }
}

/** Событие SSE загрузки модели nemo-speech (``.../model/events``). */
export type NemoSpeechModelEvent = NemoSpeechDownloadState & {
  /** Монотонный номер события (SSE `id`) — защита от повторов истории. */
  seq?: number
}

export type ModelKind = 'whisper-cpp' | 'llm' | 'pyannote' | 'gigaam' | 'sherpa'
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
  /** Монотонный номер события на сервере (SSE `id`); защита от повторов истории. */
  seq?: number
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
  /** Ключ ресурса в едином реестре внешних ресурсов (#98). */
  asset_key?: string
  /** Вид ресурса: pip-пакет или скачиваемый бинарник. */
  kind?: 'pip' | 'binary'
  /** Есть ли готовый артефакт под текущую ОС/архитектуру (бинарники). */
  downloadable?: boolean
  /** Платформа выбранного артефакта. */
  platform?: string
  /** Описание скачиваемого артефакта (размер, вариант, хеш). */
  artifact?: AssetArtifact | null
  /** Ключ настройки пути бинарника (``WHISPER_CPP_BINARY`` и т.п.). */
  setting_key?: string
  /** Ключ настройки каталога библиотек. */
  lib_setting_key?: string
  /** Текущий путь бинарника из настроек. */
  installed_path?: string
  optional?: boolean
  /** Ключ в реестре устанавливаемых пакетов (#66) — если это пакет, а не бинарник. */
  dep_key?: string
  /** Spec установки из allowlist (для пакетов). */
  spec?: string
  /** Есть ли установщик (uv/pip), чтобы показать кнопку «Установить». */
  installable?: boolean
}

/** Один разрешённый артефакт (URL + sha256) под ОС/архитектуру. */
export type AssetArtifact = {
  os: string
  arch: string
  url: string
  sha256: string
  size: number
  archive: string
  variant: string
}

/** Элемент единого реестра внешних ресурсов (#98). */
export type AssetInfo = {
  key: string
  label: string
  kind: 'pip' | 'binary'
  needed_for: string
  check_id: string
  optional: boolean
  note: string
  status: DependencyStatus
  message: string
  error: string | null
  bytes_done: number
  total: number
  fraction: number | null
  installed: boolean
  downloadable: boolean
  /** Установлен в служебный каталог `web-data/bin` (можно удалить из UI). */
  managed: boolean
  path: string
  platform: string
  artifact: AssetArtifact | null
  settings_field: string
  lib_settings_field: string
  env_key: string
  lib_env_key: string
  spec: string
  module: string
}

export type AssetsResponse = {
  assets: AssetInfo[]
  installer: string | null
  bin_dir: string
}

/** Статус фоновой установки опционального пакета (#66). */
export type DependencyStatus = 'idle' | 'running' | 'done' | 'error'

export type DependencyInfo = {
  key: string
  spec: string
  label: string
  check_id: string
  needed_for: string
  /** Модуль уже импортируем (``find_spec``). */
  installed: boolean
  /** Есть ли чем ставить (uv/pip). */
  installable: boolean
  status: DependencyStatus
  message: string
  error: string | null
}

export type DepsResponse = {
  deps: DependencyInfo[]
  /** Выбранный установщик: ``uv``, ``pip`` или ``null``. */
  installer: string | null
}

/** Событие SSE установки пакета/бинарника (``/api/assets/events``). */
export type DependencyEvent = {
  /** Монотонный номер события (SSE `id`) — защита от повторов истории. */
  seq?: number
  key: string
  status: DependencyStatus
  message: string
  error: string | null
  bytes_done?: number
  total?: number
  fraction?: number | null
  path?: string
}

/** Сводка готовности: что установлено и чего не хватает (#98). */
export type ReadinessItem = {
  id?: string
  key?: string
  label?: string
  title?: string
  present?: boolean
  installed?: boolean
  available?: boolean
  needed?: boolean
  size?: number
  expected_size?: number
  installed_path?: string
  path?: string
  platform?: string
}

export type ReadinessSummary = {
  models: ReadinessItem[]
  dependencies: ReadinessItem[]
  binaries: ReadinessItem[]
  missing_models: string[]
  missing_dependencies: string[]
  missing_binaries: string[]
}

export type SetupPlan = {
  hardware: { options: HardwareOption[]; current: string }
  steps: SetupStep[]
  required_models: string[]
  missing_models: string[]
  binaries: BinaryRequirement[]
  readiness?: ReadinessSummary
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

/** Пользовательский шаблон промпта резюме (#97). */
export type SummaryPrompt = {
  id: number
  name: string
  body: string
  builtin: boolean
  created_at: string | null
  updated_at: string | null
}

/** Ответ ``GET /api/summary-prompts`` (#97). */
export type SummaryPromptsResponse = {
  prompts: SummaryPrompt[]
  active_id: number | null
}

export const EXPORT_FORMATS = ['txt', 'docx', 'json', 'srt', 'vtt', 'md', 'pdf'] as const

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

/**
 * Компактная оценка времени обработки (#107): «45 с», «12 мин», «1 ч 20 мин».
 * Значения приблизительные, поэтому вызывающий код добавляет «≈».
 */
export function formatEstimate(seconds: number | null): string {
  if (seconds == null || !Number.isFinite(seconds) || seconds <= 0) return '—'
  if (seconds < 60) return `${Math.round(seconds)} с`
  const minutes = Math.round(seconds / 60)
  if (minutes < 60) return `${minutes} мин`
  const hours = Math.floor(minutes / 60)
  const rest = minutes % 60
  return rest ? `${hours} ч ${rest} мин` : `${hours} ч`
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

// --- Шаблоны промпта резюме (#97) -------------------------------------------

/** Список шаблонов промпта резюме с id активного. */
export async function fetchSummaryPrompts(): Promise<SummaryPromptsResponse> {
  return api<SummaryPromptsResponse>('/api/summary-prompts')
}

/** Создаёт пользовательский шаблон промпта резюме. */
export async function createSummaryPrompt(name: string, body: string): Promise<SummaryPrompt> {
  return api<SummaryPrompt>('/api/summary-prompts', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, body }),
  })
}

/** Обновляет имя/тело существующего шаблона. */
export async function updateSummaryPrompt(
  id: number,
  patch: { name?: string; body?: string },
): Promise<SummaryPrompt> {
  return api<SummaryPrompt>(`/api/summary-prompts/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patch),
  })
}

/** Удаляет шаблон; возвращает новый ``active_id`` (может измениться). */
export async function deleteSummaryPrompt(
  id: number,
): Promise<{ deleted: number; active_id: number | null }> {
  return api<{ deleted: number; active_id: number | null }>(`/api/summary-prompts/${id}`, {
    method: 'DELETE',
  })
}

/** Делает шаблон активным (применяется к задачам по умолчанию). */
export async function activateSummaryPrompt(
  id: number,
): Promise<{ active_id: number | null; prompt: SummaryPrompt }> {
  return api<{ active_id: number | null; prompt: SummaryPrompt }>(
    `/api/summary-prompts/${id}/activate`,
    { method: 'POST' },
  )
}

// --- Постадийный кэш обработки (#100) --------------------------------------

/** Состояние общего постадийного кэша конвейера (``GET /api/cache``). */
export type CacheInfo = {
  /** Каталог кэша (``web-data/cache``). */
  directory: string
  /** Число файлов кэша. */
  files: number
  /** Суммарный размер кэша в байтах. */
  bytes: number
  /** Задачи, которые воркер ведёт прямо сейчас (в очереди или в работе). */
  active_jobs: string[]
}

/** Ответ ``POST /api/cache/clear`` — сколько файлов удалено. */
export type CacheClearResponse = {
  removed: number
  directory: string
  active_jobs: string[]
  /** Очистка выполнена принудительно, несмотря на активные задачи. */
  forced: boolean
}

/** Читает состояние постадийного кэша. */
export async function fetchCacheInfo(): Promise<CacheInfo> {
  return api<CacheInfo>('/api/cache')
}

/**
 * Чистит постадийный кэш. ``force=true`` разрешает очистку во время активного
 * прогона (сервер иначе отвечает 409).
 */
export async function clearCache(force = false): Promise<CacheClearResponse> {
  return api<CacheClearResponse>('/api/cache/clear', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ force }),
  })
}

// --- Ключ OpenAI-совместимого API (#110) -----------------------------------

/** Статус ключа OpenAI-совместимого API (``GET /api/api-key``). */
export type ApiKeyStatus = {
  /** Действующий ключ задан (в секретах или в config.env). */
  set: boolean
  /** Откуда взят действующий ключ: сохранён в секретах или из config.env. */
  source: 'secrets' | 'env' | null
  /** Ключ сохранён именно в web-data/secrets.json. */
  secret_set: boolean
  /** Действующий ключ в открытом виде (сервер локальный) — для показа/копирования. */
  key: string | null
  /** Маска ключа для компактного показа. */
  masked: string | null
}

/** Читает статус ключа OpenAI-совместимого API. */
export async function fetchApiKey(): Promise<ApiKeyStatus> {
  return api<ApiKeyStatus>('/api/api-key')
}

/**
 * Генерирует новый случайный ключ (или сохраняет ``key``, если передан).
 * Генерация по запросу без тела.
 */
export async function generateApiKey(key?: string): Promise<ApiKeyStatus> {
  return api<ApiKeyStatus>('/api/api-key', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(key === undefined ? {} : { key }),
  })
}

/** Очищает сохранённый ключ OpenAI-совместимого API. */
export async function clearApiKey(): Promise<ApiKeyStatus> {
  return api<ApiKeyStatus>('/api/api-key', { method: 'DELETE' })
}

// --- Чат по стенограмме (#54/#96) ------------------------------------------

/** Ссылка ответа LLM на реплику стенограммы (таймкод + говорящий + текст). */
export type ChatCitation = {
  index: number
  start: number
  end: number
  speaker: string
  text: string
}

/** Сообщение чата по стенограмме (вопрос пользователя или ответ LLM). */
export type ChatMessage = {
  id: number
  role: 'user' | 'assistant'
  content: string
  citations: ChatCitation[]
  created_at: string | null
}

export type ChatHistoryResponse = { messages: ChatMessage[] }

/** Событие SSE-потока ответа чата (``POST /api/jobs/{id}/chat``). */
export type ChatEvent =
  | { type: 'start' }
  | { type: 'token'; text: string }
  | { type: 'done'; content: string; citations: ChatCitation[]; message?: ChatMessage }
  | { type: 'error'; message: string }

/** История чата по задаче. */
export async function fetchChatHistory(jobId: string): Promise<ChatHistoryResponse> {
  return api<ChatHistoryResponse>(`/api/jobs/${jobId}/chat`)
}

/** Очищает историю чата по задаче; возвращает число удалённых сообщений. */
export async function clearChatHistory(jobId: string): Promise<{ cleared: number }> {
  return api<{ cleared: number }>(`/api/jobs/${jobId}/chat`, { method: 'DELETE' })
}

/**
 * Отправляет вопрос и читает поток SSE ответа вручную (``fetch`` + reader):
 * ``EventSource`` умеет только GET, а вопрос уходит телом POST. Кадры вида
 * ``data: {...}`` разбираются и передаются в `onEvent` по мере поступления.
 */
export async function streamChat(
  jobId: string,
  message: string,
  onEvent: (event: ChatEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`/api/jobs/${jobId}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
    signal,
  })
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
  if (!response.body) throw new Error('Сервер не вернул поток ответа')
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let boundary = buffer.indexOf('\n\n')
    while (boundary >= 0) {
      const frame = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)
      for (const line of frame.split('\n')) {
        if (!line.startsWith('data:')) continue
        const payload = line.slice(5).trim()
        if (!payload) continue
        try {
          onEvent(JSON.parse(payload) as ChatEvent)
        } catch {
          // повреждённый кадр пропускаем — поток продолжается
        }
      }
      boundary = buffer.indexOf('\n\n')
    }
  }
}
