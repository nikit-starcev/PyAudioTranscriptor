import { useCallback, useEffect, useRef, useState } from 'react'

import {
  api,
  ASR_BACKENDS,
  DEVICES,
  DIARIZATION_ENGINES,
  errorMessage,
  EXPORT_FORMATS,
  formatBytes,
  HF_TOKEN_URL,
  LLM_PROVIDERS,
  NEMO_SPEECH_DEVICES,
  PYANNOTE_MODEL_URL,
  type HfCheckResult,
  type LlmCheckResult,
  type NemoSpeechCandidate,
  type NemoSpeechDetectResponse,
  type NemoSpeechModelEvent,
  type NemoSpeechModelStatus,
  type WebSettings,
} from '../api'
import {
  Alert,
  Button,
  Card,
  CardContent,
  CardHeader,
  Checkbox,
  Field,
  Input,
  ProgressBar,
  Select,
  Spinner,
} from './ui'

type Props = {
  onSaved?: (settings: WebSettings) => void
}

type ToggleKey =
  | 'glossary_enabled'
  | 'llm_enabled'
  | 'llm_summary'
  | 'denoise'
  | 'mark_overlap'
  | 'merge_same_name_speakers'
  | 'normalize_text'
  | 'clean_artifacts'
  | 'enable_correction'
  | 'protocol_auto'
  | 'word_timestamps'

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
  {
    key: 'merge_same_name_speakers',
    label: 'Сводить одинаковые имена',
    hint:
      'Кластеры с одинаковым уверенным именем (enrollment) — в одного говорящего; ' +
      'безымянные «Спикер N» не сливаются',
  },
  { key: 'normalize_text', label: 'Нормализация текста', hint: 'Пробелы, пунктуация, многоточия' },
  { key: 'clean_artifacts', label: 'Очистка артефактов', hint: 'Удалять [СМЕХ], [BLANK_AUDIO] и т.п.' },
  {
    key: 'enable_correction',
    label: 'Автоисправление опечаток',
    hint:
      'Правит только неизвестные словоформы (pymorphy3), осторожно; ' +
      'включает стадию «correction» в прогон',
  },
  {
    key: 'protocol_auto',
    label: 'Считать резюме автоматически по завершении',
    hint:
      'Автоматически считать резюме и экспортировать протокол по итогам прогона; ' +
      'при выключении протокол формируется только кнопкой «Сформировать протокол»',
  },
  {
    key: 'word_timestamps',
    label: 'Пословные таймстемпы',
    hint:
      'Слова с временами (start/end) в результате — из токенов ASR; ' +
      'доступны в JSON/API (поле words). По умолчанию включено',
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

/** Допустимые linkage гибридной кластеризации (DIARIZATION_HYBRID_LINKAGE). */
const HYBRID_LINKAGES = ['ward', 'complete', 'average'] as const

/** Числовые поля раздела «Диаризация» — редактируются как текст ради дробей. */
type NumericKey =
  | 'diarization_estimate_seconds'
  | 'diarization_estimate_threshold'
  | 'diarization_route_max_speakers'
  | 'diarization_hybrid_window_seconds'
  | 'diarization_hybrid_overlap_seconds'
  | 'diarization_hybrid_min_speaker_seconds'
  | 'diarization_hybrid_threshold'

const NUMERIC_KEYS: NumericKey[] = [
  'diarization_estimate_seconds',
  'diarization_estimate_threshold',
  'diarization_route_max_speakers',
  'diarization_hybrid_window_seconds',
  'diarization_hybrid_overlap_seconds',
  'diarization_hybrid_min_speaker_seconds',
  'diarization_hybrid_threshold',
]

/** Текстовые черновики числовых полей (чтобы «0.» не теряло точку). */
function numericDrafts(settings: WebSettings): Record<NumericKey, string> {
  return Object.fromEntries(NUMERIC_KEYS.map((key) => [key, String(settings[key])])) as Record<
    NumericKey,
    string
  >
}

/** Разбирает число из черновика; пустое/битое значение — прежнее. */
function parseNumber(raw: string, fallback: number): number {
  const trimmed = raw.trim().replace(',', '.')
  if (!trimmed) return fallback
  const parsed = Number(trimmed)
  return Number.isFinite(parsed) ? parsed : fallback
}

function ToggleRow({
  label,
  hint,
  checked,
  onChange,
}: {
  label: string
  hint: string
  checked: boolean
  onChange: (value: boolean) => void
}) {
  return (
    <label className="flex items-start gap-2 text-sm">
      <Checkbox
        className="mt-0.5"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
      />
      <span>
        {label}
        <span className="block text-xs text-muted">{hint}</span>
      </span>
    </label>
  )
}

function SettingsPanel({ onSaved }: Props) {
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
  const [nemoDetect, setNemoDetect] = useState<NemoSpeechDetectResponse | null>(null)
  const [nemoDetecting, setNemoDetecting] = useState(false)
  const [nemoModel, setNemoModel] = useState<NemoSpeechModelStatus | null>(null)
  const [nemoDownload, setNemoDownload] = useState<NemoSpeechModelEvent | null>(null)
  const [nemoBusy, setNemoBusy] = useState(false)
  const lastNemoSeq = useRef(-1)

  const refreshNemoModel = useCallback(async () => {
    try {
      setNemoModel(await api<NemoSpeechModelStatus>('/api/diarization/nemo-speech/model'))
    } catch {
      // Статус модели не критичен для остальных настроек — молча пропускаем.
    }
  }, [])

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
    void refresh()
  }, [refresh])

  // Статус nemo-speech обновляем при открытии и подписываемся на SSE загрузки.
  // Соединение создаётся/закрывается вместе с панелью; при завершении
  // (done/error) перечитываем статус модели.
  useEffect(() => {
    setNemoDetect(null)
    setNemoDownload(null)
    lastNemoSeq.current = -1
    void refreshNemoModel()
    const source = new EventSource('/api/diarization/nemo-speech/model/events')
    source.onmessage = (message) => {
      const event = JSON.parse(message.data) as NemoSpeechModelEvent
      // Дедуп по ``seq``: сервер может повторно отдать накопленную историю при
      // (пере)подключении SSE — старые события не должны откатывать прогресс.
      const seq = event.seq
      if (typeof seq === 'number') {
        if (seq <= lastNemoSeq.current) return
        lastNemoSeq.current = seq
      }
      setNemoDownload(event)
      if (event.status === 'done' || event.status === 'error') {
        void refreshNemoModel()
      }
    }
    source.onerror = () => {
      // Соединение переподключится само; ошибку не показываем.
    }
    return () => source.close()
  }, [refreshNemoModel])

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
      const nums = numericDraft ?? numericDrafts(settings)
      const payload = {
        glossary_enabled: settings.glossary_enabled,
        glossary_db: settings.glossary_db,
        voices_dir: settings.voices_dir,
        export_formats: settings.export_formats,
        llm_enabled: settings.llm_enabled,
        llm_summary: settings.llm_summary,
        denoise: settings.denoise,
        mark_overlap: settings.mark_overlap,
        merge_same_name_speakers: settings.merge_same_name_speakers,
        normalize_text: settings.normalize_text,
        clean_artifacts: settings.clean_artifacts,
        enable_correction: settings.enable_correction,
        protocol_auto: settings.protocol_auto,
        word_timestamps: settings.word_timestamps,
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
          nums.diarization_estimate_seconds,
          settings.diarization_estimate_seconds,
        ),
        diarization_estimate_threshold: parseNumber(
          nums.diarization_estimate_threshold,
          settings.diarization_estimate_threshold,
        ),
        diarization_estimate_model: settings.diarization_estimate_model,
        diarization_route_max_speakers: Math.trunc(
          parseNumber(nums.diarization_route_max_speakers, settings.diarization_route_max_speakers),
        ),
        diarization_hybrid_enabled: settings.diarization_hybrid_enabled,
        diarization_hybrid_window_seconds: parseNumber(
          nums.diarization_hybrid_window_seconds,
          settings.diarization_hybrid_window_seconds,
        ),
        diarization_hybrid_overlap_seconds: parseNumber(
          nums.diarization_hybrid_overlap_seconds,
          settings.diarization_hybrid_overlap_seconds,
        ),
        diarization_hybrid_min_speaker_seconds: parseNumber(
          nums.diarization_hybrid_min_speaker_seconds,
          settings.diarization_hybrid_min_speaker_seconds,
        ),
        diarization_hybrid_linkage: settings.diarization_hybrid_linkage,
        diarization_hybrid_threshold: parseNumber(
          nums.diarization_hybrid_threshold,
          settings.diarization_hybrid_threshold,
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
      // Путь к бинарнику мог измениться — обновляем доступность nemo-speech.
      void refreshNemoModel()
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

  const findNemoSpeech = async () => {
    setNemoDetecting(true)
    setError(null)
    try {
      const result = await api<NemoSpeechDetectResponse>('/api/diarization/nemo-speech/detect')
      setNemoDetect(result)
      const best = result.candidates[0]
      if (best) applyNemoCandidate(best)
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setNemoDetecting(false)
    }
  }

  const applyNemoCandidate = (candidate: NemoSpeechCandidate) => {
    update({
      nemo_speech_binary: candidate.binary,
      ...(candidate.lib_path ? { nemo_speech_lib_path: candidate.lib_path } : {}),
    })
  }

  const downloadNemoModel = async () => {
    setNemoBusy(true)
    setError(null)
    try {
      await api('/api/diarization/nemo-speech/model/download', { method: 'POST' })
      await refreshNemoModel()
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setNemoBusy(false)
    }
  }

  const nemoAvailable = nemoModel?.binary.available ?? false
  const nemoDownloadState = nemoDownload ?? nemoModel?.download ?? null

  if (loading || !settings) {
    return (
      <div className="flex items-center justify-center py-10">
        <Spinner size={24} label="Загрузка настроек" />
      </div>
    )
  }

  return (
    <div className="space-y-5">
      {error && (
        <Alert tone="danger" live onDismiss={() => setError(null)}>
          {error}
        </Alert>
      )}
      {status && (
        <Alert tone="success" live onDismiss={() => setStatus(null)}>
          {status}
        </Alert>
      )}

      <div className="flex justify-end">
        <Button variant="primary" loading={busy} onClick={() => void save()}>
          Сохранить
        </Button>
      </div>

      <Card>
        <CardHeader title="Режимы обработки" />
        <CardContent className="space-y-3">
          {TOGGLES.map((toggle) => (
            <ToggleRow
              key={toggle.key}
              label={toggle.label}
              hint={toggle.hint}
              checked={settings[toggle.key]}
              onChange={(value) =>
                update({ [toggle.key]: value } as Partial<WebSettings>)
              }
            />
          ))}
        </CardContent>
      </Card>

      <Card>
        <CardHeader title="Уведомления" />
        <CardContent>
          <ToggleRow
            label="Системные уведомления"
            hint="Показывать уведомление о завершении, ошибке или отмене задачи"
            checked={settings.notifications}
            onChange={(value) => update({ notifications: value })}
          />
        </CardContent>
      </Card>

      <Card>
        <CardHeader title="Форматы экспорта" />
        <CardContent>
          <div className="flex flex-wrap gap-4">
            {EXPORT_FORMATS.map((format) => (
              <Checkbox
                key={format}
                label={format}
                checked={settings.export_formats.includes(format)}
                onChange={() => toggleFormat(format)}
              />
            ))}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader title="Пути" />
        <CardContent className="space-y-4">
          <Field label="БД глоссария" htmlFor="settings-glossary-db" hint={<>Итог: <code>{settings.glossary_db_path}</code></>}>
            <Input
              id="settings-glossary-db"
              value={settings.glossary_db}
              onChange={(event) => update({ glossary_db: event.target.value })}
              placeholder="glossary.db"
            />
          </Field>
          <Field
            label="Каталог образцов голоса"
            htmlFor="settings-voices-dir"
            hint={<>Итог: <code>{settings.voices_dir_resolved}</code></>}
          >
            <Input
              id="settings-voices-dir"
              value={settings.voices_dir}
              onChange={(event) => update({ voices_dir: event.target.value })}
              placeholder="voices"
            />
          </Field>
          <div className="grid gap-1 text-xs text-muted sm:grid-cols-2">
            <span>
              Загрузки: <code>{settings.input_dir}</code>
            </span>
            <span>
              Результаты: <code>{settings.output_dir}</code>
            </span>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader title="Распознавание: бэкенд и модели" />
        <CardContent className="space-y-4">
          <div className="grid gap-4 sm:grid-cols-2">
            <Field label="Бэкенд (ASR_BACKEND)" htmlFor="settings-asr-backend">
              <Select
                id="settings-asr-backend"
                value={settings.asr_backend}
                onChange={(event) => update({ asr_backend: event.target.value })}
              >
                {ASR_BACKENDS.map((backend) => (
                  <option key={backend} value={backend}>
                    {backend === 'gigaam' ? 'gigaam (GigaAM v3, RU, onnx-asr)' : backend}
                  </option>
                ))}
              </Select>
            </Field>
            <Field label="Устройство (DEVICE)" htmlFor="settings-device">
              <Select
                id="settings-device"
                value={settings.device}
                onChange={(event) => update({ device: event.target.value })}
              >
                {DEVICES.map((device) => (
                  <option key={device} value={device}>
                    {device}
                  </option>
                ))}
              </Select>
            </Field>
          </div>
          <Field label="Модель whisper.cpp (ggml)" htmlFor="settings-whisper-model">
            <Input
              id="settings-whisper-model"
              value={settings.whisper_cpp_model}
              onChange={(event) => update({ whisper_cpp_model: event.target.value })}
              placeholder="whisper-models/ggml-large-v3-turbo.bin"
            />
          </Field>
          <Field label="Бинарник whisper-cli" htmlFor="settings-whisper-binary">
            <Input
              id="settings-whisper-binary"
              value={settings.whisper_cpp_binary}
              onChange={(event) => update({ whisper_cpp_binary: event.target.value })}
              placeholder="whisper-cli"
            />
          </Field>
          {settings.asr_backend === 'gigaam' && (
            <div className="space-y-4 rounded-md border border-info/40 bg-info-soft p-3">
              <p className="text-xs text-info-soft-fg">
                Бэкенд gigaam требует пакет <code>onnx-asr[cpu,hub]</code>:{' '}
                <code>pip install 'onnx-asr[cpu,hub]'</code>. Модель GigaAM v3 (RU) скачивается на
                шаге «Модели» или подтягивается с Hugging Face при первом запуске.
              </p>
              <Field label="Модель GigaAM (onnx-asr)" htmlFor="settings-gigaam-model">
                <Input
                  id="settings-gigaam-model"
                  value={settings.gigaam_model}
                  onChange={(event) => update({ gigaam_model: event.target.value })}
                  placeholder="gigaam-v3-e2e-rnnt"
                />
              </Field>
              <Field
                label="Каталог модели GigaAM"
                htmlFor="settings-gigaam-path"
                hint="Если указан существующий каталог — модель грузится офлайн, без сети."
              >
                <Input
                  id="settings-gigaam-path"
                  value={settings.gigaam_model_path}
                  onChange={(event) => update({ gigaam_model_path: event.target.value })}
                  placeholder="gigaam-models/gigaam-v3-onnx"
                />
              </Field>
              <div className="grid gap-4 sm:grid-cols-2">
                <Field label="Квантизация" htmlFor="settings-gigaam-quant">
                  <Select
                    id="settings-gigaam-quant"
                    value={settings.gigaam_quantization}
                    onChange={(event) => update({ gigaam_quantization: event.target.value })}
                  >
                    <option value="">fp32 (по умолчанию)</option>
                    <option value="int8">int8 (меньше и быстрее)</option>
                  </Select>
                </Field>
                <div className="sm:pt-6">
                  <ToggleRow
                    label="VAD-сегментация"
                    hint="Резать длинное аудио встроенным VAD onnx-asr"
                    checked={settings.gigaam_vad}
                    onChange={(value) => update({ gigaam_vad: value })}
                  />
                </div>
              </div>
            </div>
          )}
          <Field label="Модель LLM (GGUF)" htmlFor="settings-llm-model">
            <Input
              id="settings-llm-model"
              value={settings.llm_model}
              onChange={(event) => update({ llm_model: event.target.value })}
              placeholder="llama-models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
            />
          </Field>
          <Field label="Бинарник llama-server" htmlFor="settings-llm-binary">
            <Input
              id="settings-llm-binary"
              value={settings.llm_binary}
              onChange={(event) => update({ llm_binary: event.target.value })}
              placeholder="llama-server"
            />
          </Field>
          <Field
            label="Локальная модель диаризации (pyannote)"
            htmlFor="settings-pyannote-model"
            hint="Если указан существующий каталог — pyannote грузится офлайн, токен HF не нужен."
          >
            <Input
              id="settings-pyannote-model"
              value={settings.pyannote_local_model}
              onChange={(event) => update({ pyannote_local_model: event.target.value })}
              placeholder="pyannote-models/speaker-diarization-community-1"
            />
          </Field>
        </CardContent>
      </Card>

      <Card>
        <CardHeader title="LLM-постобработка (провайдер)" />
        <CardContent className="space-y-4">
          <Field label="Провайдер LLM" htmlFor="settings-llm-provider">
            <Select
              id="settings-llm-provider"
              value={settings.llm_provider}
              onChange={(event) => {
                update({ llm_provider: event.target.value })
                setLlmCheck(null)
              }}
            >
              {LLM_PROVIDERS.map((provider) => (
                <option key={provider} value={provider}>
                  {provider === 'llama' ? 'llama (локально, llama.cpp)' : 'openai (внешний API)'}
                </option>
              ))}
            </Select>
          </Field>

          {settings.llm_provider === 'llama' ? (
            <p className="rounded-md bg-surface-2 px-3 py-1.5 text-xs text-muted">
              Локальный llama.cpp: модель GGUF и бинарник задаются выше, в блоке «Распознавание:
              бэкенд и модели». Текст не покидает машину.
            </p>
          ) : (
            <>
              <Alert tone="warn" title="Внимание: приватность.">
                При внешнем провайдере текст стенограммы отправляется на сторонний сервер и
                покидает вашу машину. Проект заявлен как «100% локально» — включайте осознанно.
              </Alert>
              <Field
                label="Базовый URL (LLM_BASE_URL)"
                htmlFor="settings-llm-base-url"
                hint="OpenAI, Ollama, vLLM, LM Studio, OpenRouter. Префикс /v1 подставляется автоматически, если не указан."
              >
                <Input
                  id="settings-llm-base-url"
                  value={settings.llm_base_url}
                  onChange={(event) => update({ llm_base_url: event.target.value })}
                  placeholder="https://api.openai.com/v1"
                />
              </Field>
              <Field label="Имя модели (LLM_MODEL_NAME)" htmlFor="settings-llm-model-name">
                <Input
                  id="settings-llm-model-name"
                  value={settings.llm_model_name}
                  onChange={(event) => update({ llm_model_name: event.target.value })}
                  placeholder="gpt-4o-mini"
                />
              </Field>
              <Field
                label="API-ключ (LLM_API_KEY)"
                htmlFor="settings-llm-api-key"
                hint={
                  settings.llm_api_key_set
                    ? `Ключ сохранён (${settings.llm_api_key_masked ?? '…'}). Введите новый, чтобы заменить, или очистите поле и сохраните, чтобы удалить.`
                    : 'Необязателен для локальных серверов. Хранится в отдельном файле с правами 0600.'
                }
              >
                <Input
                  id="settings-llm-api-key"
                  type="password"
                  autoComplete="off"
                  value={llmApiKey}
                  onChange={(event) => {
                    setLlmApiKey(event.target.value)
                    setLlmApiKeyTouched(true)
                    setLlmCheck(null)
                  }}
                  placeholder={settings.llm_api_key_set ? 'сохранён' : 'sk-...'}
                />
              </Field>
              <Button
                variant="secondary"
                size="sm"
                loading={llmChecking}
                onClick={() => void checkLlm()}
              >
                Проверить доступность
              </Button>
              {llmCheck && (
                <Alert tone={llmCheck.status === 'ok' ? 'success' : 'danger'} live>
                  {llmCheck.message}
                </Alert>
              )}
            </>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader title="Hugging Face (диаризация)" />
        <CardContent className="space-y-4">
          <Field
            label="Токен доступа (HF)"
            htmlFor="settings-hf-token"
            hint={
              settings.hf_token_set
                ? `Токен сохранён (${settings.hf_token_masked ?? '…'}). Введите новый, чтобы заменить, или очистите поле и сохраните, чтобы удалить.`
                : 'Токен ещё не сохранён. Он хранится локально в отдельном файле с правами 0600.'
            }
          >
            <Input
              id="settings-hf-token"
              type="password"
              autoComplete="off"
              value={hfToken}
              onChange={(event) => {
                setHfToken(event.target.value)
                setHfTokenTouched(true)
                setHfCheck(null)
              }}
              placeholder={settings.hf_token_set ? 'сохранён' : 'hf_...'}
            />
          </Field>
          <div className="flex flex-wrap items-center gap-3">
            <Button
              variant="secondary"
              size="sm"
              loading={hfChecking}
              onClick={() => void checkAccess()}
            >
              Проверить доступ
            </Button>
            <span className="flex flex-wrap gap-x-3 text-xs">
              <a
                href={HF_TOKEN_URL}
                target="_blank"
                rel="noreferrer noopener"
                className="font-medium text-primary underline hover:opacity-80"
              >
                получить токен
              </a>
              <a
                href={PYANNOTE_MODEL_URL}
                target="_blank"
                rel="noreferrer noopener"
                className="font-medium text-primary underline hover:opacity-80"
              >
                принять условия модели pyannote
              </a>
            </span>
          </div>
          {hfCheck && (
            <Alert tone={hfCheck.status === 'ok' ? 'success' : hfCheck.status === 'error' ? 'danger' : 'warn'} live>
              {hfCheck.message}
            </Alert>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader
          title="Диаризация"
          description="Кто и когда говорил. Можно выбрать движок вручную или довериться авто-выбору по числу говорящих."
        />
        <CardContent className="space-y-4">
          <Field label="Движок (DIARIZATION_ENGINE)" htmlFor="settings-diar-engine">
            <Select
              id="settings-diar-engine"
              value={settings.diarization_engine}
              onChange={(event) => update({ diarization_engine: event.target.value })}
            >
              {DIARIZATION_ENGINES.map((engine) => (
                <option key={engine} value={engine}>
                  {engine}
                </option>
              ))}
            </Select>
          </Field>
          <p className="rounded-md bg-surface-2 px-3 py-1.5 text-xs text-muted">
            {ENGINE_HINTS[settings.diarization_engine] ?? ''}
          </p>

          <div className="space-y-4 rounded-md border border-border p-3">
            <p className="text-xs font-medium text-muted">
              Оценщик числа говорящих (маршрутизация auto)
            </p>
            <ToggleRow
              label="Включить оценщик"
              hint="sherpa-onnx: silero VAD + эмбеддинги CAM++. Без него auto уходит на pyannote."
              checked={settings.diarization_estimate_enabled}
              onChange={(value) => update({ diarization_estimate_enabled: value })}
            />
            <div className="grid gap-4 sm:grid-cols-3">
              <Field label="Секунд речи (DIARIZATION_ESTIMATE_SECONDS)" htmlFor="settings-diar-estimate-seconds">
                <Input
                  id="settings-diar-estimate-seconds"
                  inputMode="decimal"
                  value={numbers?.diarization_estimate_seconds ?? ''}
                  onChange={(event) => updateNumber('diarization_estimate_seconds', event.target.value)}
                  placeholder="30"
                />
              </Field>
              <Field label="Порог (DIARIZATION_ESTIMATE_THRESHOLD)" htmlFor="settings-diar-estimate-threshold">
                <Input
                  id="settings-diar-estimate-threshold"
                  inputMode="decimal"
                  value={numbers?.diarization_estimate_threshold ?? ''}
                  onChange={(event) =>
                    updateNumber('diarization_estimate_threshold', event.target.value)
                  }
                  placeholder="0.7"
                />
              </Field>
              <Field label="Cap N (DIARIZATION_ROUTE_MAX_SPEAKERS)" htmlFor="settings-diar-route-max">
                <Input
                  id="settings-diar-route-max"
                  inputMode="numeric"
                  value={numbers?.diarization_route_max_speakers ?? ''}
                  onChange={(event) =>
                    updateNumber('diarization_route_max_speakers', event.target.value)
                  }
                  placeholder="4"
                />
              </Field>
            </div>
            <Field
              label="Модель эмбеддингов CAM++ (имя/путь, DIARIZATION_ESTIMATE_MODEL)"
              htmlFor="settings-diar-estimate-model"
              hint="Имя модели скачивается из релиза sherpa-onnx в кэш; можно указать путь к локальному .onnx. Модель есть в каталоге «Модели»."
            >
              <Input
                id="settings-diar-estimate-model"
                value={settings.diarization_estimate_model}
                onChange={(event) => update({ diarization_estimate_model: event.target.value })}
                placeholder="3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
              />
            </Field>
          </div>

          <div className="space-y-4 rounded-md border border-border p-3">
            <p className="text-xs font-medium text-muted">Гибрид (обход лимита 4 говорящих)</p>
            <ToggleRow
              label="Включить гибридную диаризацию"
              hint="Оконный nemo-speech (≤ 4 в окне) + глобальная склейка говорящих по эмбеддингам CAM++."
              checked={settings.diarization_hybrid_enabled}
              onChange={(value) => update({ diarization_hybrid_enabled: value })}
            />
            <div className="grid gap-4 sm:grid-cols-3">
              <Field label="Окно, с (DIARIZATION_HYBRID_WINDOW_SECONDS)" htmlFor="settings-hybrid-window">
                <Input
                  id="settings-hybrid-window"
                  inputMode="decimal"
                  value={numbers?.diarization_hybrid_window_seconds ?? ''}
                  onChange={(event) =>
                    updateNumber('diarization_hybrid_window_seconds', event.target.value)
                  }
                  placeholder="90"
                />
              </Field>
              <Field label="Перекрытие, с (DIARIZATION_HYBRID_OVERLAP_SECONDS)" htmlFor="settings-hybrid-overlap">
                <Input
                  id="settings-hybrid-overlap"
                  inputMode="decimal"
                  value={numbers?.diarization_hybrid_overlap_seconds ?? ''}
                  onChange={(event) =>
                    updateNumber('diarization_hybrid_overlap_seconds', event.target.value)
                  }
                  placeholder="2"
                />
              </Field>
              <Field label="Мин. речь, с (DIARIZATION_HYBRID_MIN_SPEAKER_SECONDS)" htmlFor="settings-hybrid-min">
                <Input
                  id="settings-hybrid-min"
                  inputMode="decimal"
                  value={numbers?.diarization_hybrid_min_speaker_seconds ?? ''}
                  onChange={(event) =>
                    updateNumber('diarization_hybrid_min_speaker_seconds', event.target.value)
                  }
                  placeholder="1.5"
                />
              </Field>
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <Field label="Linkage (DIARIZATION_HYBRID_LINKAGE)" htmlFor="settings-hybrid-linkage">
                <Select
                  id="settings-hybrid-linkage"
                  value={settings.diarization_hybrid_linkage}
                  onChange={(event) => update({ diarization_hybrid_linkage: event.target.value })}
                >
                  {HYBRID_LINKAGES.map((linkage) => (
                    <option key={linkage} value={linkage}>
                      {linkage}
                    </option>
                  ))}
                </Select>
              </Field>
              <Field label="Порог кластеризации (DIARIZATION_HYBRID_THRESHOLD)" htmlFor="settings-hybrid-threshold">
                <Input
                  id="settings-hybrid-threshold"
                  inputMode="decimal"
                  value={numbers?.diarization_hybrid_threshold ?? ''}
                  onChange={(event) =>
                    updateNumber('diarization_hybrid_threshold', event.target.value)
                  }
                  placeholder="0.8"
                />
              </Field>
            </div>
            <p className="text-xs text-muted">
              Перекрытие должно быть меньше окна. Требуются sherpa-onnx и модель эмбеддингов (см.
              выше/каталог моделей). Порог — в единицах евклидова расстояния при linkage=ward.
            </p>
          </div>

          <div className="space-y-4 rounded-md border border-border p-3">
            <p className="text-xs font-medium text-muted">nemo-speech (NeMo-Speech.cpp)</p>
            <Field
              label="Бинарник nemo-speech (NEMO_SPEECH_BINARY)"
              htmlFor="settings-nemo-binary"
              hint="Автопоиск в PATH, ~/.local/bin, /usr/local/bin и рядом с проектом."
            >
              <div className="flex gap-2">
                <Input
                  id="settings-nemo-binary"
                  value={settings.nemo_speech_binary}
                  onChange={(event) => update({ nemo_speech_binary: event.target.value })}
                  placeholder="nemo-speech"
                />
                <Button
                  variant="secondary"
                  className="shrink-0"
                  loading={nemoDetecting}
                  onClick={() => void findNemoSpeech()}
                >
                  Найти
                </Button>
              </div>
            </Field>
            {nemoDetect && (
              <div className="space-y-1 rounded-md bg-surface-2 p-2 text-xs">
                {nemoDetect.found ? (
                  <>
                    <p className="text-muted">
                      Найдено вариантов: {nemoDetect.candidates.length}. Выберите путь:
                    </p>
                    <ul className="space-y-1">
                      {nemoDetect.candidates.map((candidate) => (
                        <li key={candidate.binary} className="flex items-start justify-between gap-2">
                          <span className="min-w-0 break-all text-text">
                            <code>{candidate.binary}</code>
                            {candidate.version && (
                              <span className="text-muted"> · v{candidate.version}</span>
                            )}
                            {candidate.lib_path && (
                              <span className="block text-muted">lib: {candidate.lib_path}</span>
                            )}
                            {candidate.devices.length > 0 && (
                              <span className="block text-muted">{candidate.devices[0]}</span>
                            )}
                            <span className="block text-muted">источник: {candidate.source}</span>
                          </span>
                          <Button
                            variant="secondary"
                            size="sm"
                            className="shrink-0"
                            onClick={() => applyNemoCandidate(candidate)}
                          >
                            Выбрать
                          </Button>
                        </li>
                      ))}
                    </ul>
                  </>
                ) : (
                  <p className="text-warn">
                    Бинарник не найден. Укажите путь вручную или соберите NeMo-Speech.cpp.
                  </p>
                )}
              </div>
            )}
            <Field
              label="Каталог библиотек lib/ (NEMO_SPEECH_LIB_PATH)"
              htmlFor="settings-nemo-lib"
              hint="Необязательно: если библиотеки лежат рядом с бинарником, путь не нужен."
            >
              <Input
                id="settings-nemo-lib"
                value={settings.nemo_speech_lib_path}
                onChange={(event) => update({ nemo_speech_lib_path: event.target.value })}
                placeholder="nemo-speech/lib"
              />
            </Field>
            <div className="grid gap-4 sm:grid-cols-2">
              <Field
                label="Модель Sortformer (NEMO_SPEECH_MODEL)"
                htmlFor="settings-nemo-model"
                hint={
                  <>
                    Имя каталога, HF-репозиторий или путь к .gguf. Модель тянется командой{' '}
                    <code>nemo-speech pull …</code>.
                  </>
                }
              >
                <Input
                  id="settings-nemo-model"
                  value={settings.nemo_speech_model}
                  onChange={(event) => update({ nemo_speech_model: event.target.value })}
                  placeholder="nvidia/diar_streaming_sortformer_4spk-v2"
                />
              </Field>
              <Field label="Устройство (NEMO_SPEECH_DEVICE)" htmlFor="settings-nemo-device">
                <Select
                  id="settings-nemo-device"
                  value={settings.nemo_speech_device}
                  onChange={(event) => update({ nemo_speech_device: event.target.value })}
                >
                  {NEMO_SPEECH_DEVICES.map((device) => (
                    <option key={device} value={device}>
                      {device}
                    </option>
                  ))}
                </Select>
              </Field>
            </div>
            <div className="space-y-2 rounded-md border border-border p-2">
              <div className="flex flex-wrap items-center gap-2">
                <Button
                  variant="secondary"
                  size="sm"
                  loading={nemoBusy || nemoDownloadState?.status === 'downloading'}
                  disabled={!nemoAvailable}
                  onClick={() => void downloadNemoModel()}
                >
                  {nemoDownloadState?.status === 'downloading'
                    ? 'Скачивание…'
                    : 'Скачать модель Sortformer'}
                </Button>
                {nemoModel && (
                  <span className="text-xs text-muted">
                    {nemoModel.present
                      ? `В кэше${nemoModel.size ? `: ${formatBytes(nemoModel.size)}` : ''}`
                      : 'Модель не найдена в кэше'}
                  </span>
                )}
              </div>
              {!nemoAvailable && (
                <p className="text-xs text-warn">
                  Сначала укажите или найдите nemo-speech (кнопка «Найти» выше) и сохраните
                  настройки — затем можно скачать модель.
                </p>
              )}
              {nemoDownloadState && nemoDownloadState.status !== 'idle' && (
                <div className="space-y-1">
                  {nemoDownloadState.status === 'downloading' && (
                    <ProgressBar
                      size="sm"
                      value={(nemoDownloadState.fraction ?? 0) * 100}
                      indeterminate={nemoDownloadState.fraction == null}
                      label="Скачивание модели Sortformer"
                    />
                  )}
                  <p
                    className={
                      nemoDownloadState.status === 'error'
                        ? 'text-xs text-danger'
                        : nemoDownloadState.status === 'done'
                          ? 'text-xs text-success'
                          : 'text-xs text-muted'
                    }
                  >
                    {nemoDownloadState.error ?? nemoDownloadState.message}
                    {nemoDownloadState.status === 'downloading' &&
                      nemoDownloadState.bytes_done > 0 && (
                        <span className="tabular-nums">
                          {' '}
                          · {formatBytes(nemoDownloadState.bytes_done)}
                          {nemoDownloadState.total > 0
                            ? ` / ${formatBytes(nemoDownloadState.total)}`
                            : ''}
                        </span>
                      )}
                  </p>
                  {nemoDownloadState.status === 'done' && nemoDownloadState.path && (
                    <p className="break-all text-[11px] text-muted">{nemoDownloadState.path}</p>
                  )}
                </div>
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="flex justify-end">
        <Button variant="primary" loading={busy} onClick={() => void save()}>
          Сохранить
        </Button>
      </div>
    </div>
  )
}

export default SettingsPanel
