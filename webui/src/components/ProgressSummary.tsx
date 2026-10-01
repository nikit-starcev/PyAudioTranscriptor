import { useEffect, useRef, useState } from 'react'

import type { HealthStatus, JobEvent } from '../api'

const HEALTH_LABELS: Record<HealthStatus, string> = {
  ok: 'В норме',
  slow: 'Замедлено',
  stalled: 'Нет активности',
}

const HEALTH_STYLES: Record<HealthStatus, string> = {
  ok: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300',
  slow: 'bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300',
  stalled: 'bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300',
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
}

/**
 * Сводка над стадиями: общий процент прогона, бейдж здоровья, ETA и давность
 * последнего обновления (#15/#24).
 *
 * Между событиями SSE счётчики «тикают» локально: ETA уменьшается, а возраст
 * обновления растёт — так «зависшая» задача заметна без новых событий.
 */
export default function ProgressSummary({ progress, running }: Props) {
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
  const percent =
    progress.progress_percent ?? (progress.fraction != null ? progress.fraction * 100 : 0)

  return (
    <div className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-sm">
      <span className="tabular-nums text-slate-700 dark:text-slate-200">
        Общий прогресс: <span className="font-medium">{Math.round(percent)}%</span>
      </span>
      {health && (
        <span
          className={`rounded-full px-2 py-0.5 text-xs ${
            HEALTH_STYLES[health] ??
            'bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300'
          }`}
          title="Состояние задачи по активности и темпу прогресса"
        >
          {HEALTH_LABELS[health] ?? health}
        </span>
      )}
      {running && eta != null && (
        <span className="tabular-nums text-slate-600 dark:text-slate-300">
          ≈ осталось {formatEta(eta)}
        </span>
      )}
      {lastUpdate != null && (
        <span className="text-xs text-slate-400 dark:text-slate-500">
          последнее обновление {formatAgo(lastUpdate)}
        </span>
      )}
    </div>
  )
}
