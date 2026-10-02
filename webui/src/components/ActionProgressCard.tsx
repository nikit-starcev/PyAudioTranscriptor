// Карточка прогресса длительного действия (#58): этапы, секундомер, результат.
//
// Не блокирует интерфейс: это плавающая панель без модальной подложки, которую
// можно закрыть. Показывается, пока действие идёт, и после завершения — с
// итоговым текстом (успех/ошибка) и кнопкой закрытия.

import { useEffect, useState } from 'react'

import { formatClock } from '../api'
import { ACTION_STEPS, ACTION_TITLES, type ActionRun } from '../actionProgress'

type Props = {
  run: ActionRun | null
  onClose: () => void
}

type StepState = 'done' | 'active' | 'error' | 'pending'

function ProgressStep({
  label,
  state,
  message,
}: {
  label: string
  state: StepState
  message: string | null
}) {
  const marker =
    state === 'done' ? '✓' : state === 'error' ? '✕' : state === 'active' ? '◐' : '·'
  const markerClass =
    state === 'done'
      ? 'text-emerald-500 dark:text-emerald-400'
      : state === 'error'
        ? 'text-red-500 dark:text-red-400'
        : state === 'active'
          ? 'animate-pulse text-blue-500 dark:text-blue-400'
          : 'text-slate-300 dark:text-slate-600'
  const labelClass =
    state === 'pending'
      ? 'text-slate-400 dark:text-slate-500'
      : 'text-slate-700 dark:text-slate-200'
  return (
    <li className="flex items-start gap-2">
      <span aria-hidden="true" className={`mt-0.5 w-3 shrink-0 text-center ${markerClass}`}>
        {marker}
      </span>
      <span className="min-w-0">
        <span className={`block text-sm ${labelClass}`}>{label}</span>
        {state === 'active' && message && (
          <span className="block break-words text-xs text-slate-500 dark:text-slate-400">
            {message}
          </span>
        )}
      </span>
    </li>
  )
}

function ActionProgressCard({ run, onClose }: Props) {
  const [now, setNow] = useState(() => Date.now())
  const running = run?.status === 'running'

  useEffect(() => {
    if (!running) return
    const timer = window.setInterval(() => setNow(Date.now()), 200)
    return () => window.clearInterval(timer)
  }, [running])

  if (!run) return null

  const steps = ACTION_STEPS[run.kind]
  const activeIndex = steps.findIndex((step) => step.id === run.stage)
  const elapsedSeconds =
    ((run.finishedAt ?? (running ? now : run.startedAt)) - run.startedAt) / 1000
  const percent =
    run.status === 'done'
      ? 100
      : run.fraction != null
        ? Math.round(Math.max(0, Math.min(1, run.fraction)) * 100)
        : null

  const stepState = (index: number): StepState => {
    if (run.status === 'done') return 'done'
    if (run.status === 'error' && index === Math.max(activeIndex, 0)) return 'error'
    if (activeIndex >= 0 && index < activeIndex) return 'done'
    if (run.status === 'running' && index === activeIndex) return 'active'
    return 'pending'
  }

  return (
    <div
      role="status"
      aria-live="polite"
      className="fixed bottom-4 right-4 z-50 w-80 max-w-[calc(100vw-2rem)] rounded-lg border border-slate-200 bg-white/95 p-3 shadow-lg backdrop-blur dark:border-slate-700 dark:bg-slate-900/95"
    >
      <div className="mb-2 flex items-start gap-2">
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-medium text-slate-800 dark:text-slate-100">
            {ACTION_TITLES[run.kind]}
          </p>
          <p className="text-xs text-slate-400 dark:text-slate-500">
            Затрачено: <span className="tabular-nums">{formatClock(elapsedSeconds)}</span>
          </p>
        </div>
        <button
          type="button"
          onClick={onClose}
          aria-label="Закрыть окно прогресса"
          className="shrink-0 rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-700 dark:hover:bg-slate-800 dark:hover:text-slate-200"
        >
          ✕
        </button>
      </div>

      <div className="mb-2 h-1.5 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-700">
        <div
          className={`h-full rounded-full transition-all duration-300 ${
            run.status === 'error' ? 'bg-red-500' : 'bg-blue-500 dark:bg-blue-400'
          } ${percent == null && running ? 'w-1/3 animate-pulse' : ''}`}
          style={percent != null ? { width: `${percent}%` } : undefined}
        />
      </div>

      <ul className="space-y-1">
        {steps.map((step, index) => (
          <ProgressStep
            key={step.id}
            label={step.label}
            state={stepState(index)}
            message={run.message}
          />
        ))}
      </ul>

      {run.status !== 'running' && (
        <p
          className={`mt-2 rounded-md px-2 py-1.5 text-xs ${
            run.status === 'error'
              ? 'bg-red-50 text-red-700 dark:bg-red-950/50 dark:text-red-300'
              : 'bg-emerald-50 text-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-300'
          }`}
        >
          {run.result ?? (run.status === 'done' ? 'Готово' : 'Ошибка')}
        </p>
      )}
    </div>
  )
}

export default ActionProgressCard
