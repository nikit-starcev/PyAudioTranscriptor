import { useEffect, useRef, useState } from 'react'

import { formatClock, type AsrDeviceInfo, type HealthStatus, type JobEvent } from '../api'
import { Badge, type BadgeTone } from './ui'
import { STAGE_KEYS, stageLabel } from './StageTimes'

const HEALTH_LABELS: Record<HealthStatus, string> = {
  ok: 'В норме',
  slow: 'Замедлено',
  stalled: 'Нет активности',
}

const HEALTH_TONES: Record<HealthStatus, BadgeTone> = {
  ok: 'success',
  slow: 'warn',
  stalled: 'danger',
}

/** ETA человекачитаемо: секунды до минуты, дальше — минуты. */
function formatEta(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return '—'
  if (seconds < 60) return `${Math.ceil(seconds)} с`
  return `${Math.max(1, Math.round(seconds / 60))} мин`
}

/** Давность последнего обновления: «только что» / секунды / минуты / часы. */
function formatAgo(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return '—'
  if (seconds < 5) return 'только что'
  if (seconds < 60) return `${Math.floor(seconds)} с назад`
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes} мин назад`
  return `${Math.floor(minutes / 60)} ч назад`
}

type Props = {
  /** Последнее SSE-событие прогресса (или снимок из API). */
  progress: JobEvent | null
  /** Идёт ли обработка прямо сейчас (для локального «тиканья»). */
  running: boolean
  /** Устройство ASR для индикатора на стадии распознавания (#72). */
  asrDevice?: AsrDeviceInfo | null
}

/**
 * Сводка над стадиями: общий процент прогона, бейдж здоровья, ETA и давность
 * последнего обновления (#15/#24).
 *
 * Между событиями SSE счётчики «тикают» локально: ETA уменьшается, а возраст
 * обновления растёт — так «зависшая» задача заметна без новых событий.
 */
export default function ProgressSummary({ progress, running, asrDevice }: Props) {
  const [now, setNow] = useState(() => Date.now())
  const anchor = useRef<{ at: number; eta: number | null; last: number | null }>({
    at: Date.now(),
    eta: null,
    last: null,
  })

  useEffect(() => {
    anchor.current = {
      at: Date.now(),
      eta: progress?.eta_seconds ?? null,
      last: progress?.health?.last_update_seconds ?? null,
    }
    setNow(Date.now())
  }, [progress])

  useEffect(() => {
    if (!running) return
    const id = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(id)
  }, [running])

  if (!progress) return null

  const since = Math.max((now - anchor.current.at) / 1000, 0)
  const eta = anchor.current.eta != null ? Math.max(anchor.current.eta - since, 0) : null
  const lastUpdate = anchor.current.last != null ? anchor.current.last + since : null
  const health = progress.health?.status ?? null
  const healthReason = progress.health?.reason?.trim()
  const percent =
    progress.progress_percent ?? (progress.fraction != null ? progress.fraction * 100 : 0)

  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm">
      <span className="tabular-nums text-text">
        Общий прогресс: <span className="font-medium">{Math.round(percent)}%</span>
      </span>
      {health && (
        <span title={healthReason || 'Состояние задачи по активности и темпу прогресса'}>
          <Badge tone={HEALTH_TONES[health] ?? 'neutral'}>
            {HEALTH_LABELS[health] ?? health}
          </Badge>
        </span>
      )}
      {progress.duration != null && (
        <span className="tabular-nums text-muted">
          длительность записи: {formatClock(progress.duration)}
        </span>
      )}
      {running && progress.stage && STAGE_KEYS.includes(progress.stage) && (
        <span className="text-muted">
          идёт: <span className="font-medium text-text">{stageLabel(progress.stage)}</span>
        </span>
      )}
      {asrDevice && progress.stage === 'asr' && (
        <span
          title={`${asrDevice.label} · ${asrDevice.note}${
            asrDevice.details.length ? ` · ${asrDevice.details.join('; ')}` : ''
          }`}
        >
          <Badge
            tone={
              asrDevice.device === 'gpu'
                ? 'success'
                : asrDevice.device === 'unknown'
                  ? 'warn'
                  : 'neutral'
            }
          >
            Устройство ASR: {asrDevice.label}
          </Badge>
        </span>
      )}
      {running && eta != null && (
        <span className="tabular-nums text-muted">≈ осталось {formatEta(eta)}</span>
      )}
      {lastUpdate != null && (
        <span className="text-xs text-muted">последнее обновление {formatAgo(lastUpdate)}</span>
      )}
    </div>
  )
}
