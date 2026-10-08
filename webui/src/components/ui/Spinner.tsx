import { Loader2 } from 'lucide-react'

import { cn } from './cn'

export type SpinnerProps = {
  size?: number
  label?: string
  className?: string
}

export function Spinner({ size = 16, label, className }: SpinnerProps) {
  return (
    <span
      role={label ? 'status' : undefined}
      aria-label={label}
      className={cn('inline-flex items-center text-muted', className)}
    >
      <Loader2 aria-hidden className="animate-spin" style={{ width: size, height: size }} />
      {label != null && <span className="sr-only">{label}</span>}
    </span>
  )
}
