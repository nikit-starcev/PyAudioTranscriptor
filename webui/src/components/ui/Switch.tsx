import type { ReactNode } from 'react'

import { cn } from './cn'

export type SwitchProps = {
  checked: boolean
  onChange: (checked: boolean) => void
  label?: ReactNode
  disabled?: boolean
  id?: string
  'aria-label'?: string
  className?: string
}

export function Switch({
  checked,
  onChange,
  label,
  disabled = false,
  id,
  className,
  'aria-label': ariaLabel,
}: SwitchProps) {
  const control = (
    <button
      id={id}
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={ariaLabel}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={cn(
        'relative inline-flex h-5 w-9 shrink-0 items-center rounded-full transition-colors',
        'focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring',
        'disabled:cursor-not-allowed disabled:opacity-50',
        checked ? 'bg-primary' : 'bg-surface-3 border border-border-strong',
        className,
      )}
    >
      <span
        aria-hidden
        className={cn(
          'inline-block h-4 w-4 transform rounded-full bg-white shadow-sm transition-transform',
          checked ? 'translate-x-[18px]' : 'translate-x-[2px]',
        )}
      />
    </button>
  )

  if (label == null) return control

  return (
    <label className="inline-flex cursor-pointer items-center gap-2 text-sm text-text">
      {control}
      <span>{label}</span>
    </label>
  )
}
