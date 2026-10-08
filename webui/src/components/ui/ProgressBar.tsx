import { cn } from './cn'

export type ProgressTone = 'primary' | 'success' | 'warn' | 'danger'

const TONES: Record<ProgressTone, string> = {
  primary: 'bg-primary',
  success: 'bg-success',
  warn: 'bg-warn',
  danger: 'bg-danger',
}

export type ProgressBarProps = {
  value?: number
  max?: number
  tone?: ProgressTone
  indeterminate?: boolean
  size?: 'sm' | 'md'
  label?: string
  className?: string
}

export function ProgressBar({
  value = 0,
  max = 100,
  tone = 'primary',
  indeterminate = false,
  size = 'md',
  label,
  className,
}: ProgressBarProps) {
  const safeMax = max > 0 ? max : 100
  const percent = Math.min(100, Math.max(0, (value / safeMax) * 100))
  const height = size === 'sm' ? 'h-1.5' : 'h-2'

  return (
    <div
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={safeMax}
      aria-valuenow={indeterminate ? undefined : Math.round(value)}
      className={cn('w-full overflow-hidden rounded-full bg-surface-3', height, className)}
    >
      <div
        className={cn(
          'h-full rounded-full transition-[width] duration-300',
          TONES[tone],
          indeterminate && 'w-1/3 animate-pulse',
        )}
        style={indeterminate ? undefined : { width: `${percent}%` }}
      />
    </div>
  )
}
