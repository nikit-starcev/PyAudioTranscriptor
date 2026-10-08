import type { HTMLAttributes, ReactNode } from 'react'

import { cn } from './cn'

export type CardProps = HTMLAttributes<HTMLDivElement> & {
  padded?: boolean
}

export function Card({ padded = false, className, children, ...rest }: CardProps) {
  return (
    <div
      className={cn(
        'rounded-lg border border-border bg-surface text-text shadow-sm',
        padded && 'p-4',
        className,
      )}
      {...rest}
    >
      {children}
    </div>
  )
}

export type CardHeaderProps = {
  title: ReactNode
  description?: ReactNode
  actions?: ReactNode
  className?: string
  id?: string
}

export function CardHeader({ title, description, actions, className, id }: CardHeaderProps) {
  return (
    <div
      className={cn(
        'flex flex-wrap items-center justify-between gap-3 border-b border-border px-4 py-3',
        className,
      )}
    >
      <div className="min-w-0">
        <h2 id={id} className="truncate text-sm font-semibold text-text">
          {title}
        </h2>
        {description != null && <p className="mt-0.5 text-xs text-muted">{description}</p>}
      </div>
      {actions != null && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </div>
  )
}

export function CardTitle({ className, children, ...rest }: HTMLAttributes<HTMLHeadingElement>) {
  return (
    <h2 className={cn('text-sm font-semibold text-text', className)} {...rest}>
      {children}
    </h2>
  )
}

export function CardContent({ className, children, ...rest }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn('p-4', className)} {...rest}>
      {children}
    </div>
  )
}

export function CardFooter({ className, children, ...rest }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        'flex flex-wrap items-center justify-end gap-2 border-t border-border px-4 py-3',
        className,
      )}
      {...rest}
    >
      {children}
    </div>
  )
}
