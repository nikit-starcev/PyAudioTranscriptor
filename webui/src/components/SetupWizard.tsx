import { useCallback, useEffect, useRef, useState } from 'react'

import {
  api,
  errorMessage,
  HF_TOKEN_URL,
  PYANNOTE_MODEL_URL,
  type DoctorReport,
  type HfCheckResult,
  type ModelsResponse,
  type ReadinessItem,
  type SetupPlan,
  type WebSettings,
} from '../api'
import BinaryRequirements from './BinaryRequirements'
import ModelsPanel from './ModelsPanel'

type Props = {
  open: boolean
  onClose: () => void
  report: DoctorReport | null
  onRecheck: () => void
  onChanged?: () => void
}

const STEP_ORDER = ['hardware', 'hf_token', 'models', 'binaries', 'readiness']

function formatSize(size?: number): string {
  if (!size || size <= 0) return ''
  const units = ['Б', 'КБ', 'МБ', 'ГБ']
  let value = size
  let index = 0
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024
    index += 1
  }
  return index === 0 ? `${Math.round(value)} ${units[index]}` : `${value.toFixed(1)} ${units[index]}`
}

function readinessOk(item: ReadinessItem): boolean {
  return Boolean(item.present ?? item.installed ?? item.available)
}

function readinessLabel(item: ReadinessItem): string {
  return item.title ?? item.label ?? item.id ?? item.key ?? ''
}

/** Строка сводки готовности с иконкой, размером и путём. */
function ReadinessGroup({
  title,
  items,
  missing,
}: {
  title: string
  items: ReadinessItem[]
  missing: string[]
}) {
  if (items.length === 0) return null
  return (
    <div className="rounded-md border border-slate-200 p-2 dark:border-slate-800">
      <p className="mb-1 text-xs font-medium text-slate-500 dark:text-slate-400">
        {title}
        {missing.length > 0 ? ` — не хватает: ${missing.length}` : ' — всё на месте'}
      </p>
      <ul className="space-y-0.5">
        {items.map((item, idx) => {
          const ok = readinessOk(item)
          const path = item.installed_path || item.path || ''
          const size = item.expected_size || item.size || 0
          return (
            <li
              key={item.id ?? item.key ?? idx}
              className="flex flex-wrap items-baseline gap-x-2 text-xs text-slate-600 dark:text-slate-300"
            >
              <span className={ok ? 'text-emerald-600 dark:text-emerald-400' : 'text-amber-600 dark:text-amber-400'}>
                {ok ? '✓' : '✗'}
              </span>
              <span>{readinessLabel(item)}</span>
              {size > 0 && <span className="text-slate-400 dark:text-slate-500">~{formatSize(size)}</span>}
              {path && ok && (
                <span className="truncate font-mono text-[10px] text-slate-400 dark:text-slate-500" title={path}>
                  {path}
                </span>
              )}
            </li>
          )
        })}
      </ul>
    </div>
  )
}

function hfStyle(status: HfCheckResult['status']): string {
  if (status === 'ok') return 'bg-emerald-50 text-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-300'
  if (status === 'error') return 'bg-red-50 text-red-700 dark:bg-red-950/50 dark:text-red-300'
  return 'bg-amber-50 text-amber-700 dark:bg-amber-950/40 dark:text-amber-300'
}

function SetupWizard({ open, onClose, report, onRecheck, onChanged }: Props) {
  const [plan, setPlan] = useState<SetupPlan | null>(null)
  const [settings, setSettings] = useState<WebSettings | null>(null)
  const [index, setIndex] = useState(0)
  const [loading, setLoading] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [status, setStatus] = useState<string | null>(null)
  const [hardware, setHardware] = useState<string | null>(null)
  const [hfToken, setHfToken] = useState('')
  const [hfTokenTouched, setHfTokenTouched] = useState(false)
  const [hfCheck, setHfCheck] = useState<HfCheckResult | null>(null)
  const [hfChecking, setHfChecking] = useState(false)

  // Дедупликация refresh: параллельные вызовы схлопываются, а если запрос
  // пришёл во время выполнения — выполняется ещё один проход после него.
  const refreshBusy = useRef(false)
  const refreshQueued = useRef(false)

  const refresh = useCallback(async () => {
    if (refreshBusy.current) {
      refreshQueued.current = true
      return
    }
    refreshBusy.current = true
    try {
      do {
        refreshQueued.current = false
        setLoading(true)
        setError(null)
        try {
          const nextPlan = await api<SetupPlan>('/api/setup')
          const nextSettings = await api<WebSettings>('/api/settings')
          setPlan(nextPlan)
          setSettings(nextSettings)
          setHardware((current) => current ?? nextPlan.hardware.current)
        } catch (cause) {
          setError(errorMessage(cause))
        }
      } while (refreshQueued.current)
    } finally {
      refreshBusy.current = false
      setLoading(false)
    }
  }, [])

  // Обработчик изменения моделей для панели: стабильная идентичность, чтобы
  // не пересоздавать её на каждый рендер (и не рвать SSE-соединение, #65).
  const handleModelsChanged = useCallback(() => {
    void refresh()
    onChanged?.()
  }, [refresh, onChanged])

  // Стабильный обработчик для шага «Бинарники/Пакеты»: после автоустановки
  // пакета пересобираем план (и доктор через onChanged), не пересоздавая SSE.
  const handleDepsChanged = useCallback(() => {
    void refresh()
    onChanged?.()
  }, [refresh, onChanged])

  useEffect(() => {
    if (open) {
      setIndex(0)
      setStatus(null)
      setHfToken('')
      setHfTokenTouched(false)
      setHfCheck(null)
      void refresh()
    }
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

  const currentStep = STEP_ORDER[index]
  const steps = plan?.steps ?? []

  const savePatch = async (patch: Record<string, unknown>, message: string) => {
    await api<WebSettings>('/api/settings', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    })
    setStatus(message)
    await refresh()
    onChanged?.()
  }

  const chooseHardware = async (optionId: string) => {
    const option = plan?.hardware.options.find((item) => item.id === optionId)
    setHardware(optionId)
    if (!option) return
    setBusy(true)
    setError(null)
    try {
      await savePatch(
        { asr_backend: option.asr_backend, device: option.device },
        `Выбрано: ${option.label}`,
      )
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setBusy(false)
    }
  }

  const saveToken = async () => {
    setBusy(true)
    setError(null)
    try {
      await savePatch({ hf_token: hfToken.trim() }, 'Токен сохранён')
      setHfToken('')
      setHfTokenTouched(false)
      setHfCheck(null)
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
      setHfCheck(
        await api<HfCheckResult>('/api/doctor/hf-check', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(token ? { token } : {}),
        }),
      )
    } catch (cause) {
      setHfCheck({ status: 'error', message: errorMessage(cause), account: null })
    } finally {
      setHfChecking(false)
    }
  }

  const applyModelPaths = async () => {
    setBusy(true)
    setError(null)
    try {
      const models = await api<ModelsResponse>('/api/models')
      const patch: Record<string, string> = {}
      for (const model of models.models) {
        if (model.status.present && model.setting_key) {
          patch[model.setting_key] = model.primary_path
        }
      }
      if (Object.keys(patch).length === 0) {
        setStatus('Нет загруженных моделей для привязки')
      } else {
        await savePatch(patch, 'Пути моделей сохранены в настройках')
      }
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setBusy(false)
    }
  }

  const recheck = async () => {
    setBusy(true)
    try {
      onRecheck()
    } finally {
      setBusy(false)
    }
  }

  const stepBadge = (stepId: string) => {
    const step = steps.find((item) => item.id === stepId)
    if (!step) return 'text-slate-300 dark:text-slate-600'
    if (step.status === 'ok') return 'text-emerald-600 dark:text-emerald-400'
    return 'text-amber-600 dark:text-amber-400'
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-slate-900/40 p-4 sm:items-center dark:bg-black/60"
      onClick={onClose}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Мастер настройки"
        className="flex max-h-[90vh] w-full max-w-3xl flex-col overflow-hidden rounded-lg bg-white shadow-xl dark:bg-slate-900"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3 dark:border-slate-800">
          <h2 className="font-medium">Мастер первого запуска</h2>
          <button
            type="button"
            onClick={onClose}
            className="rounded-md border border-slate-300 px-2 py-1 text-xs hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800"
          >
            Закрыть
          </button>
        </div>

        <ol className="flex flex-wrap gap-x-4 gap-y-1 border-b border-slate-200 px-4 py-2 text-xs dark:border-slate-800">
          {steps.map((step, stepIndex) => (
            <li key={step.id}>
              <button
                type="button"
                onClick={() => setIndex(stepIndex)}
                className="flex items-center gap-1 hover:underline"
              >
                <span className={stepBadge(step.id)}>{step.status === 'ok' ? '✓' : '●'}</span>
                <span className={stepIndex === index ? 'font-medium' : ''}>{step.title}</span>
              </button>
            </li>
          ))}
        </ol>

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

          {!plan || !settings ? (
            <p className="py-6 text-center text-sm text-slate-400 dark:text-slate-500">Загрузка…</p>
          ) : (
            <>
              {loading && (
                <p className="text-center text-xs text-slate-400 dark:text-slate-500" role="status">
                  Обновление…
                </p>
              )}
              {currentStep === 'hardware' && (
                <div className="space-y-3">
                  <p className="text-sm text-slate-600 dark:text-slate-300">
                    На чём считать распознавание? Выбор сохранится в настройках
                    (<code>ASR_BACKEND</code>, <code>DEVICE</code>).
                  </p>
                  <div className="space-y-2">
                    {plan.hardware.options.map((option) => (
                      <label
                        key={option.id}
                        className={`flex cursor-pointer items-start gap-3 rounded-md border p-3 text-sm ${
                          hardware === option.id
                            ? 'border-blue-400 bg-blue-50 dark:border-blue-700 dark:bg-blue-950/40'
                            : 'border-slate-200 dark:border-slate-800'
                        }`}
                      >
                        <input
                          type="radio"
                          name="wizard-hardware"
                          className="mt-0.5"
                          checked={hardware === option.id}
                          onChange={() => void chooseHardware(option.id)}
                        />
                        <span>
                          <span className="font-medium">{option.label}</span>
                          <span className="block text-xs text-slate-500 dark:text-slate-400">
                            {option.note}
                          </span>
                          <span className="block text-xs text-slate-400 dark:text-slate-500">
                            {option.asr_backend} · {option.device}
                          </span>
                        </span>
                      </label>
                    ))}
                  </div>
                </div>
              )}

              {currentStep === 'hf_token' && (
                <div className="space-y-3">
                  <p className="text-sm text-slate-600 dark:text-slate-300">
                    Токен нужен для скачивания gated-модели pyannote. Если модель уже
                    скачана и путь указан в настройках — можно пропустить.
                  </p>
                  <label className="block text-sm">
                    <span className="text-slate-600 dark:text-slate-300">Токен Hugging Face</span>
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
                        ? `Токен сохранён (${settings.hf_token_masked ?? '…'}).`
                        : 'Токен ещё не сохранён.'}
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
                    <button
                      type="button"
                      onClick={() => void saveToken()}
                      disabled={busy || !hfTokenTouched}
                      className="rounded-md bg-slate-800 px-3 py-1 text-xs text-white hover:bg-slate-700 disabled:opacity-40 dark:bg-slate-200 dark:text-slate-900 dark:hover:bg-white"
                    >
                      Сохранить токен
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
                    <p role="status" className={`rounded-md px-3 py-1.5 text-xs ${hfStyle(hfCheck.status)}`}>
                      {hfCheck.message}
                    </p>
                  )}
                </div>
              )}

              {currentStep === 'models' && (
                <div className="space-y-3">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <p className="text-sm text-slate-600 dark:text-slate-300">
                      Нужные для выбранного режима модели отмечены галочками. Скачанные можно
                      привязать к настройкам одной кнопкой.
                    </p>
                    <button
                      type="button"
                      onClick={() => void applyModelPaths()}
                      disabled={busy}
                      className="rounded-md border border-slate-300 px-3 py-1 text-xs hover:bg-slate-100 disabled:opacity-40 dark:border-slate-600 dark:hover:bg-slate-800"
                    >
                      Применить пути моделей
                    </button>
                  </div>
                  <ModelsPanel
                    requiredIds={plan.required_models}
                    onChanged={handleModelsChanged}
                  />
                </div>
              )}

              {currentStep === 'binaries' && (
                <BinaryRequirements
                  requirements={plan.binaries}
                  onChanged={handleDepsChanged}
                />
              )}

              {currentStep === 'readiness' && (
                <div className="space-y-3">
                  <p className="text-sm text-slate-600 dark:text-slate-300">
                    Финальная проверка окружения. Если критичных проблем нет — можно запускать
                    обработку.
                  </p>
                  {report ? (
                    <div
                      className={`rounded-md border px-3 py-2 text-sm ${
                        report.summary.critical_failures === 0
                          ? 'border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900 dark:bg-emerald-950/40 dark:text-emerald-300'
                          : 'border-amber-300 bg-amber-50 text-amber-800 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-200'
                      }`}
                    >
                      {report.summary.critical_failures === 0
                        ? '✓ Готово — критичных проблем нет'
                        : `Осталось критичных проблем: ${report.summary.critical_failures}`}
                      <span className="ml-2 text-xs">
                        (проверок {report.checks.length}, предупреждений {report.summary.warn})
                      </span>
                    </div>
                  ) : (
                    <p className="text-sm text-slate-400 dark:text-slate-500">Отчёт недоступен.</p>
                  )}
                  {plan.readiness && (
                    <div className="space-y-2 text-sm">
                      <p className="text-xs font-medium uppercase tracking-wide text-slate-400 dark:text-slate-500">
                        Что установлено
                      </p>
                      <ReadinessGroup
                        title="Модели"
                        items={plan.readiness.models}
                        missing={plan.readiness.missing_models}
                      />
                      <ReadinessGroup
                        title="Пакеты"
                        items={plan.readiness.dependencies}
                        missing={plan.readiness.missing_dependencies}
                      />
                      <ReadinessGroup
                        title="Бинарники"
                        items={plan.readiness.binaries}
                        missing={plan.readiness.missing_binaries}
                      />
                    </div>
                  )}
                  <button
                    type="button"
                    onClick={() => void recheck()}
                    disabled={busy}
                    className="rounded-md bg-slate-800 px-3 py-1.5 text-sm text-white hover:bg-slate-700 disabled:opacity-40 dark:bg-slate-200 dark:text-slate-900 dark:hover:bg-white"
                  >
                    Проверить готовность
                  </button>
                </div>
              )}
            </>
          )}
        </div>

        <div className="flex items-center justify-between gap-2 border-t border-slate-200 px-4 py-3 dark:border-slate-800">
          <button
            type="button"
            onClick={() => setIndex((value) => Math.max(0, value - 1))}
            disabled={index === 0}
            className="rounded-md border border-slate-300 px-3 py-1.5 text-sm hover:bg-slate-100 disabled:opacity-40 dark:border-slate-700 dark:hover:bg-slate-800"
          >
            Назад
          </button>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={onClose}
              className="rounded-md border border-slate-300 px-3 py-1.5 text-sm hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800"
            >
              Закрыть
            </button>
            {index < STEP_ORDER.length - 1 ? (
              <button
                type="button"
                onClick={() => setIndex((value) => Math.min(STEP_ORDER.length - 1, value + 1))}
                className="rounded-md bg-slate-800 px-3 py-1.5 text-sm text-white hover:bg-slate-700 dark:bg-slate-200 dark:text-slate-900 dark:hover:bg-white"
              >
                Далее
              </button>
            ) : (
              <button
                type="button"
                onClick={onClose}
                className="rounded-md bg-emerald-600 px-3 py-1.5 text-sm text-white hover:bg-emerald-500"
              >
                Готово
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default SetupWizard
