import type { InputHTMLAttributes, ReactNode } from 'react'

import { cn } from './cn'

export type CheckboxProps = Omit<InputHTMLAttributes<HTMLInputElement>, 'type'> & {
  label?: ReactNode
}

export function Checkbox({ label, className, id, ...rest }: CheckboxProps) {
  const control = (
    <input
      id={id}
      type="checkbox"
      className={cn(
        'h-4 w-4 shrink-0 rounded border-border-strong bg-surface text-primary accent-primary',
        'focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring',
        className,
      )}
      {...rest}
    />
  )

  if (label == null) return control

  return (
    <label className="inline-flex cursor-pointer items-center gap-2 text-sm text-text">
      {control}
      <span>{label}</span>
    </label>
  )
}
