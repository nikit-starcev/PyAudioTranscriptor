import { useCallback, useEffect, useState } from 'react'

import {
  api,
  errorMessage,
  EXPORT_FORMATS,
  HF_TOKEN_URL,
  PYANNOTE_MODEL_URL,
  type HfCheckResult,
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

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      setSettings(await api<WebSettings>('/api/settings'))
      setHfToken('')
      setHfTokenTouched(false)
      setHfCheck(null)
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
        ...(hfTokenTouched ? { hf_token: hfToken } : {}),
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
      setStatus('Настройки сохранены')
      onSaved?.(saved)
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setBusy(false)
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
