import type { SelectHTMLAttributes } from 'react'

import { cn } from './cn'
import { FIELD_CONTROL } from './fieldStyles'

export type SelectProps = SelectHTMLAttributes<HTMLSelectElement> & {
  invalid?: boolean
}

export function Select({ className, invalid = false, children, ...rest }: SelectProps) {
  return (
    <select
      aria-invalid={invalid || undefined}
      className={cn(FIELD_CONTROL, 'h-9 cursor-pointer px-3 pr-8', invalid && 'border-danger', className)}
      {...rest}
    >
      {children}
    </select>
  )
}
