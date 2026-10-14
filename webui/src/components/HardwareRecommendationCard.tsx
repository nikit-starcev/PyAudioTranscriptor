import { useCallback, useEffect, useState } from 'react'

import { api, errorMessage, type HardwareReport, type WebSettings } from '../api'
import { Alert, Badge, Button, Spinner } from './ui'

type Props = {
  /** Заголовок блока. */
  title?: string
  /** Вызывается после успешного применения рекомендации (сохранённые настройки). */
  onApplied?: (settings: WebSettings) => void
}

/**
 * Блок «Определённое железо и рекомендация» (#146).
 *
 * Самостоятельно запрашивает ``GET /api/hardware`` и показывает найденную
 * конфигурацию и рекомендацию с обоснованием «нашли / предлагаем / потому что».
 * Применение — только по подтверждению пользователя: ``PUT /api/settings`` с
 * профилем и точным набором полей рекомендации. Ничего не меняет молча.
 */
function HardwareRecommendationCard({ title = 'Подбор настроек под железо', onApplied }: Props) {
  const [report, setReport] = useState<HardwareReport | null>(null)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [status, setStatus] = useState<string | null>(null)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      setReport(await api<HardwareReport>('/api/hardware'))
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  const apply = async () => {
    const rec = report?.recommendation
    if (!rec) return
    if (
      !window.confirm(
        'Применить рекомендованные настройки? Текущие значения бэкенда, устройства, ' +
          'диаризации и стадий будут заменены. Индивидуальные поля можно изменить и ' +
          'сохранить после.',
      )
    ) {
      return
    }
    setBusy(true)
    setError(null)
    setStatus(null)
    try {
      const saved = await api<WebSettings>('/api/settings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ preset: rec.profile, ...rec.apply }),
      })
      setStatus('Рекомендация применена')
      onApplied?.(saved)
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setBusy(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center gap-3 text-sm text-muted">
        <Spinner size={16} label="Определение конфигурации железа" />
        <span>Определяем конфигурацию железа…</span>
      </div>
    )
  }

  const hw = report?.hardware
  const rec = report?.recommendation

  return (
    <div className="space-y-3 rounded-md border border-border p-3">
      <div className="flex flex-wrap items-center gap-2">
        <p className="text-sm font-medium text-text">{title}</p>
        {rec && <Badge tone="info">{rec.label}</Badge>}
        <span className="ml-auto">
          <Button variant="ghost" size="sm" loading={false} onClick={() => void load()}>
            Определить снова
          </Button>
        </span>
      </div>

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

      {hw && rec ? (
        <div className="space-y-2 text-sm">
          <p className="text-muted">{rec.found}</p>
          <p className="text-text">{rec.proposed}</p>
          <p className="text-xs text-muted">Потому что: {rec.why}</p>
          {rec.whisper_model_id && !rec.whisper_model_available && (
            <p className="text-xs text-warn">
              Модель {rec.whisper_model_id} ещё не скачана — добавьте её на шаге «Модели».
            </p>
          )}
          {hw.notes.length > 0 && (
            <p className="text-xs text-muted">Оговорки: {hw.notes.join('; ')}</p>
          )}
          <div>
            <Button variant="primary" size="sm" loading={busy} onClick={() => void apply()}>
              Применить рекомендацию
            </Button>
          </div>
        </div>
      ) : (
        <p className="text-sm text-muted">Конфигурация недоступна.</p>
      )}
    </div>
  )
}

export default HardwareRecommendationCard
