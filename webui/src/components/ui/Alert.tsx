import type { ReactNode } from 'react'
import { AlertCircle, CheckCircle2, Info, TriangleAlert } from 'lucide-react'

import { cn } from './cn'
import { IconButton } from './IconButton'
import { X } from 'lucide-react'

export type AlertTone = 'info' | 'success' | 'warn' | 'danger'

const TONES: Record<AlertTone, { container: string; icon: typeof Info }> = {
  info: { container: 'border-info/40 bg-info-soft text-info-soft-fg', icon: Info },
  success: { container: 'border-success/40 bg-success-soft text-success-soft-fg', icon: CheckCircle2 },
  warn: { container: 'border-warn/40 bg-warn-soft text-warn-soft-fg', icon: TriangleAlert },
  danger: { container: 'border-danger/40 bg-danger-soft text-danger-soft-fg', icon: AlertCircle },
}

export type AlertProps = {
  tone?: AlertTone
  title?: ReactNode
  children?: ReactNode
  onDismiss?: () => void
  dismissLabel?: string
  live?: boolean
  className?: string
}

export function Alert({
  tone = 'info',
  title,
  children,
  onDismiss,
  dismissLabel = 'Скрыть сообщение',
  live = false,
  className,
}: AlertProps) {
  const { container, icon: Icon } = TONES[tone]
  return (
    <div
      role={live ? 'status' : 'alert'}
      aria-live={live ? 'polite' : undefined}
      className={cn(
        'flex items-start gap-3 rounded-md border px-3 py-2 text-sm',
        container,
        className,
      )}
    >
      <Icon aria-hidden className="mt-0.5 h-4 w-4 shrink-0" />
      <div className="min-w-0 flex-1 space-y-0.5">
        {title != null && <p className="font-medium">{title}</p>}
        {children != null && <div className="break-words">{children}</div>}
      </div>
      {onDismiss != null && (
        <IconButton
          aria-label={dismissLabel}
          size="sm"
          variant="ghost"
          onClick={onDismiss}
          className="-mr-1 -mt-0.5 shrink-0 text-current hover:bg-black/5 dark:hover:bg-white/10"
        >
          <X aria-hidden className="h-3.5 w-3.5" />
        </IconButton>
      )}
    </div>
  )
}
