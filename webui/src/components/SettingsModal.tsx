import { useCallback, useEffect, useState } from 'react'

import {
  api,
  ASR_BACKENDS,
  DEVICES,
  errorMessage,
  EXPORT_FORMATS,
  HF_TOKEN_URL,
  LLM_PROVIDERS,
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

function SettingsModal({ open, onClose, onSaved }: Props) {
  const [settings, setSettings] = useState<WebSettings | null>(null)
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
      setSettings(await api<WebSettings>('/api/settings'))
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

  const update = (patch: Partial<WebSettings>) =>
    setSettings((current) => (current ? { ...current, ...patch } : current))

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
