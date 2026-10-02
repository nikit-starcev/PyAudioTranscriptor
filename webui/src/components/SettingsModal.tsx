import { useCallback, useEffect, useState } from 'react'

import {
  api,
  ASR_BACKENDS,
  DEVICES,
  DIARIZATION_ENGINES,
  errorMessage,
  EXPORT_FORMATS,
  HF_TOKEN_URL,
  LLM_PROVIDERS,
  NEMO_SPEECH_DEVICES,
  PYANNOTE_MODEL_URL,
  type HfCheckResult,
  type LlmCheckResult,
  type WebSettings,
} from '../api'

type Props = {
  open: boolean
  onClose: () => void
  onSaved?: (settings: WebSettings) => void
}

type ToggleKey =
  | 'glossary_enabled'
  | 'llm_enabled'
  | 'llm_summary'
  | 'denoise'
  | 'mark_overlap'
  | 'normalize_text'
  | 'clean_artifacts'
  | 'protocol_auto'

const TOGGLES: { key: ToggleKey; label: string; hint: string }[] = [
  {
    key: 'glossary_enabled',
    label: 'Использовать глоссарий',
    hint: 'Применять термины из БД во время распознавания',
  },
  { key: 'llm_enabled', label: 'LLM-постобработка', hint: 'Правка терминов локальной LLM' },
  { key: 'llm_summary', label: 'Резюме встречи', hint: 'Считать резюме при формировании протокола' },
  { key: 'denoise', label: 'Шумоподавление', hint: 'DeepFilterNet перед распознаванием' },
  { key: 'mark_overlap', label: 'Помечать наложение речи', hint: 'Отмечать реплики поверх друг друга' },
  { key: 'normalize_text', label: 'Нормализация текста', hint: 'Пробелы, пунктуация, многоточия' },
  { key: 'clean_artifacts', label: 'Очистка артефактов', hint: 'Удалять [СМЕХ], [BLANK_AUDIO] и т.п.' },
  {
    key: 'protocol_auto',
    label: 'Протокол сразу после обработки',
    hint: 'Автоматически считать резюме и экспортировать',
  },
]

/** Подсказка под селектором движка диаризации (#62/#64). */
const ENGINE_HINTS: Record<string, string> = {
  auto:
    '≤ 4 говорящих → nemo-speech (Vulkan, быстро); > 4 → hybrid (оконный EEND + склейка); ' +
    'если оценка не удалась или нет sherpa-onnx — pyannote (безопасно).',
  pyannote:
    'Точно и без лимита говорящих, но медленно (особенно на CPU). Нужны модель pyannote ' +
    'и torch; для скачивания gated-модели — токен Hugging Face.',
  'nemo-speech':
    'Быстро на GPU через Vulkan (NeMo-Speech.cpp), жёсткий лимит 4 говорящих. Нужны ' +
    'бинарник nemo-speech и модель Sortformer.',
  hybrid:
    'Оконный EEND (nemo-speech, ≤ 4 в окне) + глобальная склейка говорящих по ' +
    'эмбеддингам (sherpa-onnx, 3D-Speaker CAM++). Обходит лимит 4; нужны sherpa-onnx, ' +
    'модель эмбеддингов и бинарник nemo-speech.',
}

/** Класс полей нового раздела «Диаризация» (как у существующих input/select). */
const INPUT_CLASS =
  'mt-1 w-full rounded-md border border-slate-300 px-2 py-1 text-sm focus:border-blue-400 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100'

/** Числовые поля раздела «Диаризация» — редактируются как текст ради дробей. */
type NumericKey =
  | 'diarization_estimate_seconds'
  | 'diarization_estimate_threshold'
  | 'diarization_route_max_speakers'
  | 'diarization_hybrid_window_seconds'
  | 'diarization_hybrid_overlap_seconds'
  | 'diarization_hybrid_min_speaker_seconds'

const NUMERIC_KEYS: NumericKey[] = [
  'diarization_estimate_seconds',
  'diarization_estimate_threshold',
  'diarization_route_max_speakers',
  'diarization_hybrid_window_seconds',
  'diarization_hybrid_overlap_seconds',
  'diarization_hybrid_min_speaker_seconds',
]

/** Текстовые черновики числовых полей (чтобы «0.» не теряло точку). */
function numericDrafts(settings: WebSettings): Record<NumericKey, string> {
  return Object.fromEntries(
    NUMERIC_KEYS.map((key) => [key, String(settings[key])]),
  ) as Record<NumericKey, string>
}

/** Разбирает число из черновика; пустое/битое значение — прежнее. */
function parseNumber(raw: string, fallback: number): number {
  const trimmed = raw.trim().replace(',', '.')
  if (!trimmed) return fallback
  const parsed = Number(trimmed)
  return Number.isFinite(parsed) ? parsed : fallback
}

function SettingsModal({ open, onClose, onSaved }: Props) {
  const [settings, setSettings] = useState<WebSettings | null>(null)
  const [numericDraft, setNumericDraft] = useState<Record<NumericKey, string> | null>(null)
  const [loading, setLoading] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [status, setStatus] = useState<string | null>(null)
  const [hfToken, setHfToken] = useState('')
  const [hfTokenTouched, setHfTokenTouched] = useState(false)
  const [hfCheck, setHfCheck] = useState<HfCheckResult | null>(null)
  const [hfChecking, setHfChecking] = useState(false)
  const [llmApiKey, setLlmApiKey] = useState('')
  const [llmApiKeyTouched, setLlmApiKeyTouched] = useState(false)
  const [llmCheck, setLlmCheck] = useState<LlmCheckResult | null>(null)
  const [llmChecking, setLlmChecking] = useState(false)

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      const next = await api<WebSettings>('/api/settings')
      setSettings(next)
      setNumericDraft(numericDrafts(next))
      setHfToken('')
      setHfTokenTouched(false)
      setHfCheck(null)
      setLlmApiKey('')
      setLlmApiKeyTouched(false)
      setLlmCheck(null)
      setError(null)
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (open) void refresh()
  }, [open, refresh])

  useEffect(() => {
    if (!open) return
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])

  if (!open) return null

  const numbers = numericDraft ?? (settings ? numericDrafts(settings) : null)

  const update = (patch: Partial<WebSettings>) =>
    setSettings((current) => (current ? { ...current, ...patch } : current))

  const updateNumber = (key: NumericKey, raw: string) =>
    setNumericDraft((current) => ({ ...(current ?? numericDrafts(settings!)), [key]: raw }))

  const toggleFormat = (format: string) => {
    if (!settings) return
    const formats = settings.export_formats.includes(format)
      ? settings.export_formats.filter((item) => item !== format)
      : [...settings.export_formats, format]
    update({ export_formats: formats })
  }

  const save = async () => {
    if (!settings) return
    // Внешний провайдер отправляет текст за пределы машины — не включаем молча.
    if (
      settings.llm_enabled &&
      settings.llm_provider === 'openai' &&
      !window.confirm(
        'Внешний провайдер LLM: текст стенограммы будет отправлен на внешний сервер ' +
          'за пределы вашей машины. Проект заявлен как «100% локально». Продолжить?',
      )
    ) {
      return
    }
    setBusy(true)
    setError(null)
    setStatus(null)
    try {
      const numbers = numericDraft ?? numericDrafts(settings)
      const payload = {
        glossary_enabled: settings.glossary_enabled,
        glossary_db: settings.glossary_db,
        voices_dir: settings.voices_dir,
        export_formats: settings.export_formats,
        llm_enabled: settings.llm_enabled,
        llm_summary: settings.llm_summary,
        denoise: settings.denoise,
        mark_overlap: settings.mark_overlap,
        normalize_text: settings.normalize_text,
        clean_artifacts: settings.clean_artifacts,
        protocol_auto: settings.protocol_auto,
        notifications: settings.notifications,
        asr_backend: settings.asr_backend,
        device: settings.device,
        whisper_cpp_model: settings.whisper_cpp_model,
        whisper_cpp_binary: settings.whisper_cpp_binary,
        llm_model: settings.llm_model,
        llm_binary: settings.llm_binary,
        llm_provider: settings.llm_provider,
        llm_base_url: settings.llm_base_url,
        llm_model_name: settings.llm_model_name,
        pyannote_local_model: settings.pyannote_local_model,
        diarization_engine: settings.diarization_engine,
        nemo_speech_binary: settings.nemo_speech_binary,
        nemo_speech_lib_path: settings.nemo_speech_lib_path,
        nemo_speech_model: settings.nemo_speech_model,
        nemo_speech_device: settings.nemo_speech_device,
        diarization_estimate_enabled: settings.diarization_estimate_enabled,
        diarization_estimate_seconds: parseNumber(
          numbers.diarization_estimate_seconds,
          settings.diarization_estimate_seconds,
        ),
        diarization_estimate_threshold: parseNumber(
          numbers.diarization_estimate_threshold,
          settings.diarization_estimate_threshold,
        ),
        diarization_estimate_model: settings.diarization_estimate_model,
        diarization_route_max_speakers: Math.trunc(
          parseNumber(
            numbers.diarization_route_max_speakers,
            settings.diarization_route_max_speakers,
          ),
        ),
        diarization_hybrid_enabled: settings.diarization_hybrid_enabled,
        diarization_hybrid_window_seconds: parseNumber(
          numbers.diarization_hybrid_window_seconds,
          settings.diarization_hybrid_window_seconds,
        ),
        diarization_hybrid_overlap_seconds: parseNumber(
          numbers.diarization_hybrid_overlap_seconds,
          settings.diarization_hybrid_overlap_seconds,
        ),
        diarization_hybrid_min_speaker_seconds: parseNumber(
          numbers.diarization_hybrid_min_speaker_seconds,
          settings.diarization_hybrid_min_speaker_seconds,
        ),
        gigaam_model: settings.gigaam_model,
        gigaam_model_path: settings.gigaam_model_path,
        gigaam_quantization: settings.gigaam_quantization,
        gigaam_vad: settings.gigaam_vad,
        ...(hfTokenTouched ? { hf_token: hfToken } : {}),
        ...(llmApiKeyTouched ? { llm_api_key: llmApiKey } : {}),
      }
      const saved = await api<WebSettings>('/api/settings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      setSettings(saved)
      setNumericDraft(numericDrafts(saved))
      setHfToken('')
      setHfTokenTouched(false)
      setHfCheck(null)
      setLlmApiKey('')
      setLlmApiKeyTouched(false)
      setLlmCheck(null)
      setStatus('Настройки сохранены')
      onSaved?.(saved)
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setBusy(false)
    }
  }

  const checkLlm = async () => {
    if (!settings) return
    setLlmChecking(true)
    setLlmCheck(null)
    try {
      const apiKey = llmApiKey.trim()
      const result = await api<LlmCheckResult>('/api/llm/check', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          base_url: settings.llm_base_url,
          model_name: settings.llm_model_name,
          ...(apiKey ? { api_key: apiKey } : {}),
        }),
      })
      setLlmCheck(result)
    } catch (cause) {
      setLlmCheck({ status: 'error', message: errorMessage(cause), models: [] })
    } finally {
      setLlmChecking(false)
    }
  }

  const checkAccess = async () => {
    setHfChecking(true)
    setHfCheck(null)
    try {
      const token = hfToken.trim()
      const result = await api<HfCheckResult>('/api/doctor/hf-check', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(token ? { token } : {}),
      })
      setHfCheck(result)
    } catch (cause) {
      setHfCheck({ status: 'error', message: errorMessage(cause), account: null })
    } finally {
      setHfChecking(false)
    }
  }

  const hfResultStyle = (status: HfCheckResult['status']): string => {
    if (status === 'ok') {
      return 'bg-emerald-50 text-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-300'
    }
    if (status === 'error') {
      return 'bg-red-50 text-red-700 dark:bg-red-950/50 dark:text-red-300'
    }
    return 'bg-amber-50 text-amber-700 dark:bg-amber-950/40 dark:text-amber-300'
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-slate-900/40 p-4 sm:items-center dark:bg-black/60"
      onClick={onClose}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Настройки"
        className="flex max-h-[85vh] w-full max-w-2xl flex-col overflow-hidden rounded-lg bg-white shadow-xl dark:bg-slate-900"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3 dark:border-slate-800">
          <h2 className="font-medium">Настройки</h2>
          <button
            type="button"
            onClick={onClose}
            className="rounded-md border border-slate-300 px-2 py-1 text-xs hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800"
          >
            Закрыть
          </button>
        </div>

        <div className="space-y-4 overflow-y-auto px-4 py-3">
          {error && (
            <p className="rounded-md bg-red-50 px-3 py-1.5 text-xs text-red-700 dark:bg-red-950/50 dark:text-red-300">
              {error}
            </p>
          )}
          {status && (
            <p
              role="status"
              className="rounded-md bg-emerald-50 px-3 py-1.5 text-xs text-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-300"
            >
              {status}
            </p>
          )}

          {loading || !settings ? (
            <p className="py-6 text-center text-sm text-slate-400 dark:text-slate-500">Загрузка…</p>
          ) : (
            <>
              <fieldset className="space-y-2 rounded-md border border-slate-200 p-3 dark:border-slate-800">
                <legend className="px-1 text-xs font-medium text-slate-500 dark:text-slate-400">
                  Режимы обработки
                </legend>
                {TOGGLES.map((toggle) => (
                  <label key={toggle.key} className="flex items-start gap-2 text-sm">
                    <input
                      type="checkbox"
                      className="mt-0.5"
                      checked={settings[toggle.key]}
                      onChange={(event) =>
                        update({ [toggle.key]: event.target.checked } as Partial<WebSettings>)
                      }
                    />
                    <span>
                      {toggle.label}
                      <span className="block text-xs text-slate-400 dark:text-slate-500">
                        {toggle.hint}
                      </span>
                    </span>
                  </label>
                ))}
              </fieldset>

              <fieldset className="space-y-2 rounded-md border border-slate-200 p-3 dark:border-slate-800">
                <legend className="px-1 text-xs font-medium text-slate-500 dark:text-slate-400">
                  Уведомления
                </legend>
                <label className="flex items-start gap-2 text-sm">
                  <input
                    type="checkbox"
                    className="mt-0.5"
                    checked={settings.notifications}
                    onChange={(event) => update({ notifications: event.target.checked })}
                  />
                  <span>
                    Системные уведомления
                    <span className="block text-xs text-slate-400 dark:text-slate-500">
                      Показывать уведомление о завершении, ошибке или отмене задачи
                    </span>
                  </span>
                </label>
              </fieldset>

              <fieldset className="space-y-2 rounded-md border border-slate-200 p-3 dark:border-slate-800">
                <legend className="px-1 text-xs font-medium text-slate-500 dark:text-slate-400">
                  Форматы экспорта
                </legend>
                <div className="flex flex-wrap gap-4">
                  {EXPORT_FORMATS.map((format) => (
                    <label key={format} className="flex items-center gap-2 text-sm">
                      <input
                        type="checkbox"
                        checked={settings.export_formats.includes(format)}
                        onChange={() => toggleFormat(format)}
                      />
                      {format}
                    </label>
                  ))}
                </div>
              </fieldset>

              <fieldset className="space-y-3 rounded-md border border-slate-200 p-3 dark:border-slate-800">
                <legend className="px-1 text-xs font-medium text-slate-500 dark:text-slate-400">
                  Пути
                </legend>
                <label className="block text-sm">
                  <span className="text-slate-600 dark:text-slate-300">БД глоссария</span>
                  <input
                    value={settings.glossary_db}
                    onChange={(event) => update({ glossary_db: event.target.value })}
                    placeholder="glossary.db"
                    className="mt-1 w-full rounded-md border border-slate-300 px-2 py-1 text-sm focus:border-blue-400 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
                  />
                  <span className="mt-0.5 block text-xs text-slate-400 dark:text-slate-500">
                    Итог: <code>{settings.glossary_db_path}</code>
                  </span>
                </label>
                <label className="block text-sm">
                  <span className="text-slate-600 dark:text-slate-300">
                    Каталог образцов голоса
                  </span>
                  <input
                    value={settings.voices_dir}
                    onChange={(event) => update({ voices_dir: event.target.value })}
                    placeholder="voices"
                    className="mt-1 w-full rounded-md border border-slate-300 px-2 py-1 text-sm focus:border-blue-400 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
                  />
                  <span className="mt-0.5 block text-xs text-slate-400 dark:text-slate-500">
                    Итог: <code>{settings.voices_dir_resolved}</code>
                  </span>
                </label>
                <div className="grid gap-1 text-xs text-slate-400 sm:grid-cols-2 dark:text-slate-500">
                  <span>
                    Загрузки: <code>{settings.input_dir}</code>
                  </span>
                  <span>
                    Результаты: <code>{settings.output_dir}</code>
                  </span>
                </div>
              </fieldset>

              <fieldset className="space-y-3 rounded-md border border-slate-200 p-3 dark:border-slate-800">
                <legend className="px-1 text-xs font-medium text-slate-500 dark:text-slate-400">
                  Распознавание: бэкенд и модели
                </legend>
                <div className="grid gap-3 sm:grid-cols-2">
                  <label className="block text-sm">
                    <span className="text-slate-600 dark:text-slate-300">Бэкенд (ASR_BACKEND)</span>
                    <select
                      value={settings.asr_backend}
                      onChange={(event) => update({ asr_backend: event.target.value })}
                      className="mt-1 w-full rounded-md border border-slate-300 px-2 py-1 text-sm focus:border-blue-400 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
                    >
                      {ASR_BACKENDS.map((backend) => (
                        <option key={backend} value={backend}>
                          {backend === 'gigaam' ? 'gigaam (GigaAM v3, RU, onnx-asr)' : backend}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="block text-sm">
                    <span className="text-slate-600 dark:text-slate-300">Устройство (DEVICE)</span>
                    <select
                      value={settings.device}
                      onChange={(event) => update({ device: event.target.value })}
                      className="mt-1 w-full rounded-md border border-slate-300 px-2 py-1 text-sm focus:border-blue-400 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
                    >
                      {DEVICES.map((device) => (
                        <option key={device} value={device}>
                          {device}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>
                <label className="block text-sm">
                  <span className="text-slate-600 dark:text-slate-300">
                    Модель whisper.cpp (ggml)
                  </span>
                  <input
                    value={settings.whisper_cpp_model}
                    onChange={(event) => update({ whisper_cpp_model: event.target.value })}
                    placeholder="whisper-models/ggml-large-v3-turbo.bin"
                    className="mt-1 w-full rounded-md border border-slate-300 px-2 py-1 text-sm focus:border-blue-400 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
                  />
                </label>
                <label className="block text-sm">
                  <span className="text-slate-600 dark:text-slate-300">Бинарник whisper-cli</span>
                  <input
                    value={settings.whisper_cpp_binary}
                    onChange={(event) => update({ whisper_cpp_binary: event.target.value })}
                    placeholder="whisper-cli"
                    className="mt-1 w-full rounded-md border border-slate-300 px-2 py-1 text-sm focus:border-blue-400 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
                  />
                </label>
                {settings.asr_backend === 'gigaam' && (
                  <div className="space-y-3 rounded-md border border-blue-200 bg-blue-50/50 p-3 dark:border-blue-900 dark:bg-blue-950/30">
                    <p className="text-xs text-slate-600 dark:text-slate-300">
                      Бэкенд gigaam требует пакет <code>onnx-asr[cpu,hub]</code>:
                      <code className="ml-1">pip install 'onnx-asr[cpu,hub]'</code>. Модель
                      GigaAM v3 (RU) скачивается на шаге «Модели» или подтягивается с Hugging
                      Face при первом запуске.
                    </p>
                    <label className="block text-sm">
                      <span className="text-slate-600 dark:text-slate-300">
                        Модель GigaAM (onnx-asr)
                      </span>
                      <input
                        value={settings.gigaam_model}
                        onChange={(event) => update({ gigaam_model: event.target.value })}
                        placeholder="gigaam-v3-e2e-rnnt"
                        className="mt-1 w-full rounded-md border border-slate-300 px-2 py-1 text-sm focus:border-blue-400 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
                      />
                    </label>
                    <label className="block text-sm">
                      <span className="text-slate-600 dark:text-slate-300">
                        Каталог модели GigaAM
                      </span>
                      <input
                        value={settings.gigaam_model_path}
                        onChange={(event) => update({ gigaam_model_path: event.target.value })}
                        placeholder="gigaam-models/gigaam-v3-onnx"
                        className="mt-1 w-full rounded-md border border-slate-300 px-2 py-1 text-sm focus:border-blue-400 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
                      />
                      <span className="mt-0.5 block text-xs text-slate-400 dark:text-slate-500">
                        Если указан существующий каталог — модель грузится офлайн, без сети.
                      </span>
                    </label>
                    <div className="grid gap-3 sm:grid-cols-2">
                      <label className="block text-sm">
                        <span className="text-slate-600 dark:text-slate-300">Квантизация</span>
                        <select
                          value={settings.gigaam_quantization}
                          onChange={(event) => update({ gigaam_quantization: event.target.value })}
                          className="mt-1 w-full rounded-md border border-slate-300 px-2 py-1 text-sm focus:border-blue-400 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
                        >
                          <option value="">fp32 (по умолчанию)</option>
                          <option value="int8">int8 (меньше и быстрее)</option>
                        </select>
                      </label>
                      <label className="flex items-start gap-2 text-sm sm:mt-6">
                        <input
                          type="checkbox"
                          className="mt-0.5"
                          checked={settings.gigaam_vad}
                          onChange={(event) => update({ gigaam_vad: event.target.checked })}
                        />
                        <span>
                          VAD-сегментация
                          <span className="block text-xs text-slate-400 dark:text-slate-500">
                            Резать длинное аудио встроенным VAD onnx-asr
                          </span>
                        </span>
                      </label>
                    </div>
                  </div>
                )}
                <label className="block text-sm">
                  <span className="text-slate-600 dark:text-slate-300">Модель LLM (GGUF)</span>
                  <input
                    value={settings.llm_model}
                    onChange={(event) => update({ llm_model: event.target.value })}
                    placeholder="llama-models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
                    className="mt-1 w-full rounded-md border border-slate-300 px-2 py-1 text-sm focus:border-blue-400 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
                  />
                </label>
                <label className="block text-sm">
                  <span className="text-slate-600 dark:text-slate-300">Бинарник llama-server</span>
                  <input
                    value={settings.llm_binary}
                    onChange={(event) => update({ llm_binary: event.target.value })}
                    placeholder="llama-server"
                    className="mt-1 w-full rounded-md border border-slate-300 px-2 py-1 text-sm focus:border-blue-400 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
                  />
                </label>
                <label className="block text-sm">
                  <span className="text-slate-600 dark:text-slate-300">
                    Локальная модель диаризации (pyannote)
                  </span>
                  <input
                    value={settings.pyannote_local_model}
                    onChange={(event) => update({ pyannote_local_model: event.target.value })}
                    placeholder="pyannote-models/speaker-diarization-community-1"
                    className="mt-1 w-full rounded-md border border-slate-300 px-2 py-1 text-sm focus:border-blue-400 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
                  />
                  <span className="mt-0.5 block text-xs text-slate-400 dark:text-slate-500">
                    Если указан существующий каталог — pyannote грузится офлайн, токен HF не нужен.
                  </span>
                </label>
              </fieldset>

              <fieldset className="space-y-3 rounded-md border border-slate-200 p-3 dark:border-slate-800">
                <legend className="px-1 text-xs font-medium text-slate-500 dark:text-slate-400">
                  LLM-постобработка (провайдер)
                </legend>
                <label className="block text-sm">
                  <span className="text-slate-600 dark:text-slate-300">Провайдер LLM</span>
                  <select
                    value={settings.llm_provider}
                    onChange={(event) => {
                      update({ llm_provider: event.target.value })
                      setLlmCheck(null)
                    }}
                    className="mt-1 w-full rounded-md border border-slate-300 px-2 py-1 text-sm focus:border-blue-400 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
                  >
                    {LLM_PROVIDERS.map((provider) => (
                      <option key={provider} value={provider}>
                        {provider === 'llama' ? 'llama (локально, llama.cpp)' : 'openai (внешний API)'}
                      </option>
                    ))}
                  </select>
                </label>

                {settings.llm_provider === 'llama' ? (
                  <p className="rounded-md bg-slate-50 px-3 py-1.5 text-xs text-slate-500 dark:bg-slate-800/60 dark:text-slate-400">
                    Локальный llama.cpp: модель GGUF и бинарник задаются выше, в блоке
                    «Распознавание: бэкенд и модели». Текст не покидает машину.
                  </p>
                ) : (
                  <>
                    <p
                      role="alert"
                      className="rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-800 dark:border-amber-700 dark:bg-amber-950/40 dark:text-amber-200"
                    >
                      <strong>Внимание: приватность.</strong> При внешнем провайдере текст
                      стенограммы отправляется на сторонний сервер и покидает вашу машину.
                      Проект заявлен как «100% локально» — включайте осознанно.
                    </p>
                    <label className="block text-sm">
                      <span className="text-slate-600 dark:text-slate-300">
                        Базовый URL (LLM_BASE_URL)
                      </span>
                      <input
                        value={settings.llm_base_url}
                        onChange={(event) => update({ llm_base_url: event.target.value })}
                        placeholder="https://api.openai.com/v1"
                        className="mt-1 w-full rounded-md border border-slate-300 px-2 py-1 text-sm focus:border-blue-400 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
                      />
                      <span className="mt-0.5 block text-xs text-slate-400 dark:text-slate-500">
                        OpenAI, Ollama, vLLM, LM Studio, OpenRouter. Префикс /v1 подставляется
                        автоматически, если не указан.
                      </span>
                    </label>
                    <label className="block text-sm">
                      <span className="text-slate-600 dark:text-slate-300">
                        Имя модели (LLM_MODEL_NAME)
                      </span>
                      <input
                        value={settings.llm_model_name}
                        onChange={(event) => update({ llm_model_name: event.target.value })}
                        placeholder="gpt-4o-mini"
                        className="mt-1 w-full rounded-md border border-slate-300 px-2 py-1 text-sm focus:border-blue-400 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
                      />
                    </label>
                    <label className="block text-sm">
                      <span className="text-slate-600 dark:text-slate-300">
                        API-ключ (LLM_API_KEY)
                      </span>
                      <input
                        type="password"
                        autoComplete="off"
                        value={llmApiKey}
                        onChange={(event) => {
                          setLlmApiKey(event.target.value)
                          setLlmApiKeyTouched(true)
                          setLlmCheck(null)
                        }}
                        placeholder={settings.llm_api_key_set ? 'сохранён' : 'sk-...'}
                        className="mt-1 w-full rounded-md border border-slate-300 px-2 py-1 text-sm focus:border-blue-400 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
                      />
                      <span className="mt-0.5 block text-xs text-slate-400 dark:text-slate-500">
                        {settings.llm_api_key_set
                          ? `Ключ сохранён (${settings.llm_api_key_masked ?? '…'}). Введите новый, чтобы заменить, или очистите поле и сохраните, чтобы удалить.`
                          : 'Необязателен для локальных серверов. Хранится в отдельном файле с правами 0600.'}
                      </span>
                    </label>
                    <div className="flex flex-wrap items-center gap-3">
                      <button
                        type="button"
                        onClick={() => void checkLlm()}
                        disabled={llmChecking}
                        className="rounded-md border border-slate-300 px-3 py-1 text-xs hover:bg-slate-100 disabled:opacity-40 dark:border-slate-600 dark:hover:bg-slate-800"
                      >
                        {llmChecking ? 'Проверка…' : 'Проверить доступность'}
                      </button>
                    </div>
                    {llmCheck && (
                      <p
                        role="status"
                        className={`rounded-md px-3 py-1.5 text-xs ${
                          llmCheck.status === 'ok'
                            ? 'bg-emerald-50 text-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-300'
                            : 'bg-red-50 text-red-700 dark:bg-red-950/50 dark:text-red-300'
                        }`}
                      >
                        {llmCheck.message}
                      </p>
                    )}
                  </>
                )}
              </fieldset>

              <fieldset className="space-y-3 rounded-md border border-slate-200 p-3 dark:border-slate-800">
                <legend className="px-1 text-xs font-medium text-slate-500 dark:text-slate-400">
                  Hugging Face (диаризация)
                </legend>
                <label className="block text-sm">
                  <span className="text-slate-600 dark:text-slate-300">Токен доступа (HF)</span>
                  <input
                    type="password"
                    autoComplete="off"
                    value={hfToken}
                    onChange={(event) => {
                      setHfToken(event.target.value)
                      setHfTokenTouched(true)
                      setHfCheck(null)
                    }}
                    placeholder={settings.hf_token_set ? 'сохранён' : 'hf_...'}
                    className="mt-1 w-full rounded-md border border-slate-300 px-2 py-1 text-sm focus:border-blue-400 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
                  />
                  <span className="mt-0.5 block text-xs text-slate-400 dark:text-slate-500">
                    {settings.hf_token_set
                      ? `Токен сохранён (${settings.hf_token_masked ?? '…'}). Введите новый, чтобы заменить, или очистите поле и сохраните, чтобы удалить.`
                      : 'Токен ещё не сохранён. Он хранится локально в отдельном файле с правами 0600.'}
                  </span>
                </label>
                <div className="flex flex-wrap items-center gap-3">
                  <button
                    type="button"
                    onClick={() => void checkAccess()}
                    disabled={hfChecking}
                    className="rounded-md border border-slate-300 px-3 py-1 text-xs hover:bg-slate-100 disabled:opacity-40 dark:border-slate-600 dark:hover:bg-slate-800"
                  >
                    {hfChecking ? 'Проверка…' : 'Проверить доступ'}
                  </button>
                  <span className="flex flex-wrap gap-x-3 text-xs">
                    <a
                      href={HF_TOKEN_URL}
                      target="_blank"
                      rel="noreferrer noopener"
                      className="text-blue-600 underline hover:text-blue-500 dark:text-blue-400 dark:hover:text-blue-300"
                    >
                      получить токен
                    </a>
                    <a
                      href={PYANNOTE_MODEL_URL}
                      target="_blank"
                      rel="noreferrer noopener"
                      className="text-blue-600 underline hover:text-blue-500 dark:text-blue-400 dark:hover:text-blue-300"
                    >
                      принять условия модели pyannote
                    </a>
                  </span>
                </div>
                {hfCheck && (
                  <p
                    role="status"
                    className={`rounded-md px-3 py-1.5 text-xs ${hfResultStyle(hfCheck.status)}`}
                  >
                    {hfCheck.message}
                  </p>
                )}
              </fieldset>

              <fieldset className="space-y-3 rounded-md border border-slate-200 p-3 dark:border-slate-800">
                <legend className="px-1 text-xs font-medium text-slate-500 dark:text-slate-400">
                  Диаризация
                </legend>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  Кто и когда говорил. Можно выбрать движок вручную или довериться
                  авто-выбору по числу говорящих.
                </p>

                <label className="block text-sm">
                  <span className="text-slate-600 dark:text-slate-300">
                    Движок (DIARIZATION_ENGINE)
                  </span>
                  <select
                    value={settings.diarization_engine}
                    onChange={(event) => update({ diarization_engine: event.target.value })}
                    className={INPUT_CLASS}
                  >
                    {DIARIZATION_ENGINES.map((engine) => (
                      <option key={engine} value={engine}>
                        {engine}
                      </option>
                    ))}
                  </select>
                </label>
                <p className="rounded-md bg-slate-50 px-3 py-1.5 text-xs text-slate-500 dark:bg-slate-800/60 dark:text-slate-400">
                  {ENGINE_HINTS[settings.diarization_engine] ?? ''}
                </p>

                <div className="space-y-2 rounded-md border border-slate-200 p-3 dark:border-slate-800">
                  <p className="text-xs font-medium text-slate-500 dark:text-slate-400">
                    Оценщик числа говорящих (маршрутизация auto)
                  </p>
                  <label className="flex items-start gap-2 text-sm">
                    <input
                      type="checkbox"
                      className="mt-0.5"
                      checked={settings.diarization_estimate_enabled}
                      onChange={(event) =>
                        update({ diarization_estimate_enabled: event.target.checked })
                      }
                    />
                    <span>
                      Включить оценщик
                      <span className="block text-xs text-slate-400 dark:text-slate-500">
                        sherpa-onnx: silero VAD + эмбеддинги CAM++. Без него auto
                        уходит на pyannote.
                      </span>
                    </span>
                  </label>
                  <div className="grid gap-3 sm:grid-cols-3">
                    <label className="block text-sm">
                      <span className="text-slate-600 dark:text-slate-300">
                        Секунд речи (DIARIZATION_ESTIMATE_SECONDS)
                      </span>
                      <input
                        inputMode="decimal"
                        value={numbers?.diarization_estimate_seconds ?? ''}
                        onChange={(event) =>
                          updateNumber('diarization_estimate_seconds', event.target.value)
                        }
                        placeholder="30"
                        className={INPUT_CLASS}
                      />
                    </label>
                    <label className="block text-sm">
                      <span className="text-slate-600 dark:text-slate-300">
                        Порог (DIARIZATION_ESTIMATE_THRESHOLD)
                      </span>
                      <input
                        inputMode="decimal"
                        value={numbers?.diarization_estimate_threshold ?? ''}
                        onChange={(event) =>
                          updateNumber('diarization_estimate_threshold', event.target.value)
                        }
                        placeholder="0.7"
                        className={INPUT_CLASS}
                      />
                    </label>
                    <label className="block text-sm">
                      <span className="text-slate-600 dark:text-slate-300">
                        Cap N (DIARIZATION_ROUTE_MAX_SPEAKERS)
                      </span>
                      <input
                        inputMode="numeric"
                        value={numbers?.diarization_route_max_speakers ?? ''}
                        onChange={(event) =>
                          updateNumber('diarization_route_max_speakers', event.target.value)
                        }
                        placeholder="4"
                        className={INPUT_CLASS}
                      />
                    </label>
                  </div>
                  <label className="block text-sm">
                    <span className="text-slate-600 dark:text-slate-300">
                      Модель эмбеддингов CAM++ (имя/путь, DIARIZATION_ESTIMATE_MODEL)
                    </span>
                    <input
                      value={settings.diarization_estimate_model}
                      onChange={(event) =>
                        update({ diarization_estimate_model: event.target.value })
                      }
                      placeholder="3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
                      className={INPUT_CLASS}
                    />
                    <span className="mt-0.5 block text-xs text-slate-400 dark:text-slate-500">
                      Имя модели скачивается из релиза sherpa-onnx в кэш; можно указать
                      путь к локальному .onnx. Модель есть в каталоге «Модели».
                    </span>
                  </label>
                </div>

                <div className="space-y-2 rounded-md border border-slate-200 p-3 dark:border-slate-800">
                  <p className="text-xs font-medium text-slate-500 dark:text-slate-400">
                    Гибрид (обход лимита 4 говорящих)
                  </p>
                  <label className="flex items-start gap-2 text-sm">
                    <input
                      type="checkbox"
                      className="mt-0.5"
                      checked={settings.diarization_hybrid_enabled}
                      onChange={(event) =>
                        update({ diarization_hybrid_enabled: event.target.checked })
                      }
                    />
                    <span>
                      Включить гибридную диаризацию
                      <span className="block text-xs text-slate-400 dark:text-slate-500">
                        Оконный nemo-speech (≤ 4 в окне) + глобальная склейка
                        говорящих по эмбеддингам CAM++.
                      </span>
                    </span>
                  </label>
                  <div className="grid gap-3 sm:grid-cols-3">
                    <label className="block text-sm">
                      <span className="text-slate-600 dark:text-slate-300">
                        Окно, с (DIARIZATION_HYBRID_WINDOW_SECONDS)
                      </span>
                      <input
                        inputMode="decimal"
                        value={numbers?.diarization_hybrid_window_seconds ?? ''}
                        onChange={(event) =>
                          updateNumber('diarization_hybrid_window_seconds', event.target.value)
                        }
                        placeholder="90"
                        className={INPUT_CLASS}
                      />
                    </label>
                    <label className="block text-sm">
                      <span className="text-slate-600 dark:text-slate-300">
                        Перекрытие, с (DIARIZATION_HYBRID_OVERLAP_SECONDS)
                      </span>
                      <input
                        inputMode="decimal"
                        value={numbers?.diarization_hybrid_overlap_seconds ?? ''}
                        onChange={(event) =>
                          updateNumber('diarization_hybrid_overlap_seconds', event.target.value)
                        }
                        placeholder="2"
                        className={INPUT_CLASS}
                      />
                    </label>
                    <label className="block text-sm">
                      <span className="text-slate-600 dark:text-slate-300">
                        Мин. речь, с (DIARIZATION_HYBRID_MIN_SPEAKER_SECONDS)
                      </span>
                      <input
                        inputMode="decimal"
                        value={numbers?.diarization_hybrid_min_speaker_seconds ?? ''}
                        onChange={(event) =>
                          updateNumber(
                            'diarization_hybrid_min_speaker_seconds',
                            event.target.value,
                          )
                        }
                        placeholder="1.5"
                        className={INPUT_CLASS}
                      />
                    </label>
                  </div>
                  <span className="block text-xs text-slate-400 dark:text-slate-500">
                    Перекрытие должно быть меньше окна. Требуются sherpa-onnx и модель
                    эмбеддингов (см. выше/каталог моделей).
                  </span>
                </div>

                <div className="space-y-3 rounded-md border border-slate-200 p-3 dark:border-slate-800">
                  <p className="text-xs font-medium text-slate-500 dark:text-slate-400">
                    nemo-speech (NeMo-Speech.cpp)
                  </p>
                  <label className="block text-sm">
                    <span className="text-slate-600 dark:text-slate-300">
                      Бинарник nemo-speech (NEMO_SPEECH_BINARY)
                    </span>
                    <input
                      value={settings.nemo_speech_binary}
                      onChange={(event) => update({ nemo_speech_binary: event.target.value })}
                      placeholder="nemo-speech"
                      className={INPUT_CLASS}
                    />
                  </label>
                  <label className="block text-sm">
                    <span className="text-slate-600 dark:text-slate-300">
                      Каталог библиотек lib/ (NEMO_SPEECH_LIB_PATH)
                    </span>
                    <input
                      value={settings.nemo_speech_lib_path}
                      onChange={(event) => update({ nemo_speech_lib_path: event.target.value })}
                      placeholder="nemo-speech/lib"
                      className={INPUT_CLASS}
                    />
                    <span className="mt-0.5 block text-xs text-slate-400 dark:text-slate-500">
                      Необязательно: если библиотеки лежат рядом с бинарником, путь не нужен.
                    </span>
                  </label>
                  <div className="grid gap-3 sm:grid-cols-2">
                    <label className="block text-sm">
                      <span className="text-slate-600 dark:text-slate-300">
                        Модель Sortformer (NEMO_SPEECH_MODEL)
                      </span>
                      <input
                        value={settings.nemo_speech_model}
                        onChange={(event) => update({ nemo_speech_model: event.target.value })}
                        placeholder="nvidia/diar_streaming_sortformer_4spk-v2"
                        className={INPUT_CLASS}
                      />
                      <span className="mt-0.5 block text-xs text-slate-400 dark:text-slate-500">
                        Имя каталога, HF-репозиторий или путь к .gguf. Модель тянется
                        командой <code>nemo-speech pull …</code>.
                      </span>
                    </label>
                    <label className="block text-sm">
                      <span className="text-slate-600 dark:text-slate-300">
                        Устройство (NEMO_SPEECH_DEVICE)
                      </span>
                      <select
                        value={settings.nemo_speech_device}
                        onChange={(event) =>
                          update({ nemo_speech_device: event.target.value })
                        }
                        className={INPUT_CLASS}
                      >
                        {NEMO_SPEECH_DEVICES.map((device) => (
                          <option key={device} value={device}>
                            {device}
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>
                </div>
              </fieldset>
            </>
          )}
        </div>

        <div className="flex justify-end gap-2 border-t border-slate-200 px-4 py-3 dark:border-slate-800">
          <button
            type="button"
            onClick={onClose}
            className="rounded-md border border-slate-300 px-3 py-1.5 text-sm hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800"
          >
            Отмена
          </button>
          <button
            type="button"
            onClick={() => void save()}
            disabled={busy || !settings}
            className="rounded-md bg-slate-800 px-3 py-1.5 text-sm text-white hover:bg-slate-700 disabled:opacity-40 dark:bg-slate-200 dark:text-slate-900 dark:hover:bg-white"
          >
            Сохранить
          </button>
        </div>
      </div>
    </div>
  )
}

export default SettingsModal
