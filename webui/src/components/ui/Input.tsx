import type { InputHTMLAttributes } from 'react'

import { cn } from './cn'
import { FIELD_CONTROL } from './fieldStyles'

export type InputProps = InputHTMLAttributes<HTMLInputElement> & {
  invalid?: boolean
}

export function Input({ className, invalid = false, type = 'text', ...rest }: InputProps) {
  return (
    <input
      type={type}
      aria-invalid={invalid || undefined}
      className={cn(FIELD_CONTROL, 'h-9 px-3', invalid && 'border-danger', className)}
      {...rest}
    />
  )
}
