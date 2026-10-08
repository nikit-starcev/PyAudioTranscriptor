// Карточка прогресса длительного действия (#58): этапы, секундомер, результат.
//
// Не блокирует интерфейс: это плавающая панель без модальной подложки, которую
// можно закрыть. Показывается, пока действие идёт, и после завершения — с
// итоговым текстом (успех/ошибка) и кнопкой закрытия.

import { useEffect, useState } from 'react'
import { Loader2, X } from 'lucide-react'

import { formatClock } from '../api'
import { ACTION_STEPS, ACTION_TITLES, type ActionRun } from '../actionProgress'
import { Card, IconButton, ProgressBar, cn } from './ui'

type Props = {
  run: ActionRun | null
  onClose: () => void
}

type StepState = 'done' | 'active' | 'error' | 'pending'

function StepMarker({ state }: { state: StepState }) {
  if (state === 'active') {
    return <Loader2 aria-hidden className="h-3.5 w-3.5 animate-spin text-primary" />
  }
  const glyph = state === 'done' ? '✓' : state === 'error' ? '✕' : '·'
  return (
    <span
      aria-hidden
      className={cn(
        'w-3.5 text-center text-xs',
        state === 'done' ? 'text-success' : state === 'error' ? 'text-danger' : 'text-muted',
      )}
    >
      {glyph}
    </span>
  )
}

function ProgressStep({
  label,
  state,
  message,
}: {
  label: string
  state: StepState
  message: string | null
}) {
  return (
    <li className="flex items-start gap-2">
      <span className="mt-0.5 shrink-0">
        <StepMarker state={state} />
      </span>
      <span className="min-w-0">
        <span className={cn('block text-sm', state === 'pending' ? 'text-muted' : 'text-text')}>
          {label}
        </span>
        {state === 'active' && message && (
          <span className="block break-words text-xs text-muted">{message}</span>
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
    <Card
      role="status"
      aria-live="polite"
      className="fixed bottom-4 right-4 z-50 w-80 max-w-[calc(100vw-2rem)] p-3 shadow-lg backdrop-blur"
    >
      <div className="mb-2 flex items-start gap-2">
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-medium text-text">{ACTION_TITLES[run.kind]}</p>
          <p className="text-xs text-muted">
            Затрачено: <span className="tabular-nums">{formatClock(elapsedSeconds)}</span>
          </p>
        </div>
        <IconButton aria-label="Закрыть окно прогресса" size="sm" onClick={onClose}>
          <X aria-hidden className="h-4 w-4" />
        </IconButton>
      </div>

      <ProgressBar
        className="mb-2"
        size="sm"
        value={percent ?? 0}
        tone={run.status === 'error' ? 'danger' : 'primary'}
        indeterminate={percent == null && running}
        label={`Прогресс: ${ACTION_TITLES[run.kind]}`}
      />

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
          className={cn(
            'mt-2 rounded-md px-2 py-1.5 text-xs',
            run.status === 'error'
              ? 'bg-danger-soft text-danger-soft-fg'
              : 'bg-success-soft text-success-soft-fg',
          )}
        >
          {run.result ?? (run.status === 'done' ? 'Готово' : 'Ошибка')}
        </p>
      )}
    </Card>
  )
}

export default ActionProgressCard
