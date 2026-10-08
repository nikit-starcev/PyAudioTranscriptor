import type { ReactNode } from 'react'

import { cn } from './cn'

export type EmptyStateProps = {
  icon?: ReactNode
  title: ReactNode
  description?: ReactNode
  action?: ReactNode
  className?: string
}

export function EmptyState({ icon, title, description, action, className }: EmptyStateProps) {
  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center gap-3 rounded-lg border border-dashed border-border px-6 py-10 text-center',
        className,
      )}
    >
      {icon != null && <div className="text-muted">{icon}</div>}
      <div className="space-y-1">
        <p className="text-sm font-medium text-text">{title}</p>
        {description != null && <p className="text-xs text-muted">{description}</p>}
      </div>
      {action != null && <div className="mt-1">{action}</div>}
    </div>
  )
}
