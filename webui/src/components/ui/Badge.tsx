import type { ReactNode } from 'react'

import { cn } from './cn'

export type BadgeTone = 'neutral' | 'success' | 'warn' | 'danger' | 'info' | 'primary'

const TONES: Record<BadgeTone, string> = {
  neutral: 'bg-neutral-soft text-neutral-soft-fg',
  success: 'bg-success-soft text-success-soft-fg',
  warn: 'bg-warn-soft text-warn-soft-fg',
  danger: 'bg-danger-soft text-danger-soft-fg',
  info: 'bg-info-soft text-info-soft-fg',
  primary: 'bg-primary-soft text-primary-soft-fg',
}

export type BadgeProps = {
  tone?: BadgeTone
  icon?: ReactNode
  children: ReactNode
  className?: string
}

export function Badge({ tone = 'neutral', icon, children, className }: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 whitespace-nowrap rounded-full px-2 py-0.5 text-xs font-medium',
        TONES[tone],
        className,
      )}
    >
      {icon}
      {children}
    </span>
  )
}
