import type { ReactNode } from 'react'

import { cn } from './cn'

export type TooltipSide = 'top' | 'bottom'
export type TooltipAlign = 'center' | 'left' | 'right'

export type TooltipProps = {
  label: string
  side?: TooltipSide
  align?: TooltipAlign
  children: ReactNode
  className?: string
}

const ALIGN: Record<TooltipAlign, string> = {
  center: 'left-1/2 -translate-x-1/2',
  left: 'left-0',
  right: 'right-0',
}

export function Tooltip({
  label,
  side = 'top',
  align = 'center',
  children,
  className,
}: TooltipProps) {
  return (
    <span className={cn('group relative inline-flex', className)}>
      {children}
      <span
        role="tooltip"
        className={cn(
          'pointer-events-none absolute z-50 w-max max-w-64 rounded-md',
          ALIGN[align],
          'bg-text px-2 py-1 text-xs font-medium text-bg opacity-0 shadow-md transition-opacity',
          'group-hover:opacity-100 group-focus-within:opacity-100',
          side === 'top' ? 'bottom-full mb-1.5' : 'top-full mt-1.5',
        )}
      >
        {label}
      </span>
    </span>
  )
}
