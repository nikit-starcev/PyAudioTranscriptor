import { useCallback, useEffect, useRef, useState } from 'react'
import { Check, Circle } from 'lucide-react'

import {
  api,
  errorMessage,
  formatBytes,
  HF_TOKEN_URL,
  PYANNOTE_MODEL_URL,
  type DoctorReport,
  type HfCheckResult,
  type ModelsResponse,
  type ReadinessItem,
  type SetupPlan,
  type WebSettings,
} from '../api'
import {
  Alert,
  Badge,
  Button,
  Card,
  CardContent,
  CardFooter,
  CardHeader,
  Field,
  Input,
  Spinner,
  cn,
} from './ui'
import BinaryRequirements from './BinaryRequirements'
import ModelsPanel from './ModelsPanel'

type Props = {
  report: DoctorReport | null
  onRecheck: () => void
  onDone: () => void
  onChanged?: () => void
}

const STEP_ORDER = ['hardware', 'hf_token', 'models', 'binaries', 'readiness']

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
    <div className="rounded-md border border-border p-2">
      <p className="mb-1 text-xs font-medium text-muted">
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
              className="flex flex-wrap items-baseline gap-x-2 text-xs text-muted"
            >
              <span aria-hidden className={ok ? 'text-success' : 'text-warn'}>
                {ok ? <Check className="h-3 w-3" /> : <Circle className="h-2 w-2" />}
              </span>
              <span className="text-text">{readinessLabel(item)}</span>
              {size > 0 && <span className="tabular-nums">~{formatBytes(size)}</span>}
              {path && ok && (
                <span className="truncate font-mono text-[10px]" title={path}>
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

function SetupWizard({ report, onRecheck, onDone, onChanged }: Props) {
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
    setIndex(0)
    setStatus(null)
    setHfToken('')
    setHfTokenTouched(false)
    setHfCheck(null)
    void refresh()
  }, [refresh])

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

  return (
    <Card>
      <CardHeader
        title="Шаги настройки"
        description="Переключайтесь между шагами и выполняйте их по порядку"
        actions={
          loading ? <Spinner size={16} label="Обновление" /> : undefined
        }
      />
      <CardContent className="space-y-4">
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

        <ol className="flex flex-wrap gap-2">
          {steps.map((step, stepIndex) => {
            const active = stepIndex === index
            const ok = step.status === 'ok'
            return (
              <li key={step.id}>
                <button
                  type="button"
                  onClick={() => setIndex(stepIndex)}
                  aria-current={active ? 'step' : undefined}
                  className={cn(
                    'inline-flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-medium transition-colors',
                    'focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring',
                    active
                      ? 'bg-primary-soft text-primary-soft-fg'
                      : 'text-muted hover:bg-surface-2 hover:text-text',
                  )}
                >
                  <span aria-hidden className={ok ? 'text-success' : 'text-warn'}>
                    {ok ? <Check className="h-3 w-3" /> : <Circle className="h-2 w-2" />}
                  </span>
                  {step.title}
                </button>
              </li>
            )
          })}
        </ol>

        {!plan || !settings ? (
          <div className="flex items-center justify-center py-6">
            <Spinner size={20} label="Загрузка плана настройки" />
          </div>
        ) : (
          <>
            {currentStep === 'hardware' && (
              <div className="space-y-3">
                <p className="text-sm text-muted">
                  На чём считать распознавание? Выбор сохранится в настройках (
                  <code>ASR_BACKEND</code>, <code>DEVICE</code>).
                </p>
                <div className="space-y-2">
                  {plan.hardware.options.map((option) => (
                    <label
                      key={option.id}
                      className={cn(
                        'flex cursor-pointer items-start gap-3 rounded-md border p-3 text-sm',
                        hardware === option.id
                          ? 'border-primary bg-primary-soft'
                          : 'border-border hover:bg-surface-2',
                      )}
                    >
                      <input
                        type="radio"
                        name="wizard-hardware"
                        className="mt-0.5 h-4 w-4 shrink-0 accent-primary focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"
                        checked={hardware === option.id}
                        onChange={() => void chooseHardware(option.id)}
                      />
                      <span>
                        <span className="font-medium text-text">{option.label}</span>
                        <span className="block text-xs text-muted">{option.note}</span>
                        <span className="block text-xs text-muted">
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
                <p className="text-sm text-muted">
                  Токен нужен для скачивания gated-модели pyannote. Если модель уже скачана и путь
                  указан в настройках — можно пропустить.
                </p>
                <Field
                  label="Токен Hugging Face"
                  htmlFor="wizard-hf-token"
                  hint={
                    settings.hf_token_set
                      ? `Токен сохранён (${settings.hf_token_masked ?? '…'}).`
                      : 'Токен ещё не сохранён.'
                  }
                >
                  <Input
                    id="wizard-hf-token"
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
                  <Button
                    variant="primary"
                    size="sm"
                    loading={busy}
                    disabled={!hfTokenTouched}
                    onClick={() => void saveToken()}
                  >
                    Сохранить токен
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
                  <Alert
                    tone={
                      hfCheck.status === 'ok'
                        ? 'success'
                        : hfCheck.status === 'error'
                          ? 'danger'
                          : 'warn'
                    }
                    live
                  >
                    {hfCheck.message}
                  </Alert>
                )}
              </div>
            )}

            {currentStep === 'models' && (
              <div className="space-y-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <p className="text-sm text-muted">
                    Нужные для выбранного режима модели отмечены галочками. Скачанные можно
                    привязать к настройкам одной кнопкой.
                  </p>
                  <Button
                    variant="secondary"
                    size="sm"
                    loading={busy}
                    onClick={() => void applyModelPaths()}
                  >
                    Применить пути моделей
                  </Button>
                </div>
                <ModelsPanel requiredIds={plan.required_models} onChanged={handleModelsChanged} />
              </div>
            )}

            {currentStep === 'binaries' && (
              <BinaryRequirements requirements={plan.binaries} onChanged={handleDepsChanged} />
            )}

            {currentStep === 'readiness' && (
              <div className="space-y-3">
                <p className="text-sm text-muted">
                  Финальная проверка окружения. Если критичных проблем нет — можно запускать
                  обработку.
                </p>
                {report ? (
                  <Alert tone={report.summary.critical_failures === 0 ? 'success' : 'warn'}>
                    {report.summary.critical_failures === 0
                      ? 'Готово — критичных проблем нет'
                      : `Осталось критичных проблем: ${report.summary.critical_failures}`}
                    <span className="ml-2 text-xs">
                      (проверок {report.checks.length}, предупреждений {report.summary.warn})
                    </span>
                  </Alert>
                ) : (
                  <p className="text-sm text-muted">Отчёт недоступен.</p>
                )}
                {plan.readiness && (
                  <div className="space-y-2 text-sm">
                    <p className="text-xs font-medium uppercase tracking-wide text-muted">
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
                <div className="flex items-center gap-2">
                  <Button variant="primary" size="sm" loading={busy} onClick={onRecheck}>
                    Проверить готовность
                  </Button>
                  {report && (
                    <Badge tone={report.summary.critical_failures === 0 ? 'success' : 'warn'}>
                      {report.summary.critical_failures === 0
                        ? 'критичных проблем нет'
                        : `критичных: ${report.summary.critical_failures}`}
                    </Badge>
                  )}
                </div>
              </div>
            )}
          </>
        )}
      </CardContent>
      <CardFooter className="justify-between">
        <Button
          variant="secondary"
          onClick={() => setIndex((value) => Math.max(0, value - 1))}
          disabled={index === 0}
        >
          Назад
        </Button>
        <div className="flex items-center gap-2">
          <Button variant="ghost" onClick={onDone}>
            Закрыть
          </Button>
          {index < STEP_ORDER.length - 1 ? (
            <Button
              variant="primary"
              onClick={() => setIndex((value) => Math.min(STEP_ORDER.length - 1, value + 1))}
            >
              Далее
            </Button>
          ) : (
            <Button variant="primary" onClick={onDone}>
              Готово
            </Button>
          )}
        </div>
      </CardFooter>
    </Card>
  )
}

export default SetupWizard
