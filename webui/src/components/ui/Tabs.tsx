import { useRef, type ReactNode } from 'react'

import { cn } from './cn'

export type TabItem = {
  id: string
  label: ReactNode
  icon?: ReactNode
  disabled?: boolean
}

export type TabsProps = {
  items: TabItem[]
  value: string
  onChange: (id: string) => void
  'aria-label': string
  className?: string
}

export function Tabs({ items, value, onChange, className, 'aria-label': ariaLabel }: TabsProps) {
  const listRef = useRef<HTMLDivElement>(null)

  const onKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    const enabled = items.filter((item) => !item.disabled)
    if (enabled.length === 0) return
    const index = enabled.findIndex((item) => item.id === value)
    let next = index
    if (event.key === 'ArrowRight') next = (index + 1) % enabled.length
    else if (event.key === 'ArrowLeft') next = (index - 1 + enabled.length) % enabled.length
    else if (event.key === 'Home') next = 0
    else if (event.key === 'End') next = enabled.length - 1
    else return
    event.preventDefault()
    const target = enabled[next]
    onChange(target.id)
    listRef.current?.querySelector<HTMLButtonElement>(`#tab-${CSS.escape(target.id)}`)?.focus()
  }

  return (
    <div
      ref={listRef}
      role="tablist"
      aria-label={ariaLabel}
      onKeyDown={onKeyDown}
      className={cn('flex flex-wrap items-center gap-1 border-b border-border', className)}
    >
      {items.map((item) => {
        const active = item.id === value
        return (
          <button
            key={item.id}
            id={`tab-${item.id}`}
            role="tab"
            type="button"
            aria-selected={active}
            aria-controls={`panel-${item.id}`}
            tabIndex={active ? 0 : -1}
            disabled={item.disabled}
            onClick={() => onChange(item.id)}
            className={cn(
              '-mb-px inline-flex items-center gap-2 border-b-2 px-3 py-2 text-sm font-medium transition-colors',
              'focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring',
              'disabled:cursor-not-allowed disabled:opacity-50',
              active
                ? 'border-primary text-primary'
                : 'border-transparent text-muted hover:border-border-strong hover:text-text',
            )}
          >
            {item.icon}
            {item.label}
          </button>
        )
      })}
    </div>
  )
}

export type TabPanelProps = {
  value: string
  active: string
  children: ReactNode
  className?: string
}

export function TabPanel({ value, active, children, className }: TabPanelProps) {
  if (value !== active) return null
  return (
    <div
      role="tabpanel"
      id={`panel-${value}`}
      aria-labelledby={`tab-${value}`}
      tabIndex={0}
      className={cn('focus-visible:outline-none', className)}
    >
      {children}
    </div>
  )
}
