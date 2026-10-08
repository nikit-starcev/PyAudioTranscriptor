import type { TextareaHTMLAttributes } from 'react'

import { cn } from './cn'
import { FIELD_CONTROL } from './fieldStyles'

export type TextareaProps = TextareaHTMLAttributes<HTMLTextAreaElement> & {
  invalid?: boolean
}

export function Textarea({ className, invalid = false, rows = 4, ...rest }: TextareaProps) {
  return (
    <textarea
      rows={rows}
      aria-invalid={invalid || undefined}
      className={cn(FIELD_CONTROL, 'min-h-20 resize-y px-3 py-2', invalid && 'border-danger', className)}
      {...rest}
    />
  )
}
