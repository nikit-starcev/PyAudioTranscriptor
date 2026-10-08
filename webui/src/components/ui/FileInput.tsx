import { useId, useRef } from 'react'
import { Paperclip, X } from 'lucide-react'

import { cn } from './cn'
import { Button } from './Button'
import { IconButton } from './IconButton'

export type FileInputProps = {
  value?: File | null
  onChange: (file: File | null) => void
  accept?: string
  disabled?: boolean
  buttonLabel?: string
  placeholder?: string
  clearLabel?: string
  className?: string
  id?: string
  'aria-label'?: string
}

export function FileInput({
  value = null,
  onChange,
  accept,
  disabled = false,
  buttonLabel = 'Выбрать файл',
  placeholder = 'Файл не выбран',
  clearLabel = 'Убрать выбранный файл',
  className,
  id,
  'aria-label': ariaLabel,
}: FileInputProps) {
  const reactId = useId()
  const inputId = id ?? `file-${reactId.replace(/[^a-zA-Z0-9_-]/g, '')}`
  const inputRef = useRef<HTMLInputElement>(null)

  return (
    <div className={cn('flex flex-wrap items-center gap-2', className)}>
      <input
        ref={inputRef}
        id={inputId}
        type="file"
        accept={accept}
        disabled={disabled}
        aria-label={ariaLabel ?? buttonLabel}
        className="sr-only"
        onChange={(event) => {
          onChange(event.target.files?.[0] ?? null)
          event.target.value = ''
        }}
      />
      <Button
        type="button"
        variant="secondary"
        size="sm"
        disabled={disabled}
        icon={<Paperclip aria-hidden className="h-4 w-4" />}
        onClick={() => inputRef.current?.click()}
      >
        {buttonLabel}
      </Button>
      {value ? (
        <span className="flex min-w-0 items-center gap-1">
          <span className="min-w-0 break-words text-xs text-text" title={value.name}>
            {value.name}
          </span>
          <IconButton
            aria-label={clearLabel}
            size="sm"
            disabled={disabled}
            onClick={() => onChange(null)}
          >
            <X aria-hidden className="h-3.5 w-3.5" />
          </IconButton>
        </span>
      ) : (
        <span className="text-xs text-muted">{placeholder}</span>
      )}
    </div>
  )
}
