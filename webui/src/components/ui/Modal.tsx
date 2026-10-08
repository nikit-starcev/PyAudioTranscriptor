import { useEffect, useId, useRef, type ReactNode } from 'react'
import { X } from 'lucide-react'

import { cn } from './cn'
import { IconButton } from './IconButton'

const FOCUSABLE =
  'a[href],button:not([disabled]),textarea:not([disabled]),input:not([disabled]),select:not([disabled]),[tabindex]:not([tabindex="-1"])'

export type ModalSize = 'sm' | 'md' | 'lg' | 'xl'

const SIZES: Record<ModalSize, string> = {
  sm: 'max-w-sm',
  md: 'max-w-lg',
  lg: 'max-w-2xl',
  xl: 'max-w-4xl',
}

export type ModalProps = {
  open: boolean
  onClose: () => void
  title?: ReactNode
  description?: ReactNode
  children: ReactNode
  footer?: ReactNode
  size?: ModalSize
  'aria-label'?: string
}

export function Modal({
  open,
  onClose,
  title,
  description,
  children,
  footer,
  size = 'md',
  'aria-label': ariaLabel,
}: ModalProps) {
  const panelRef = useRef<HTMLDivElement>(null)
  const previouslyFocused = useRef<HTMLElement | null>(null)
  const titleId = useId()

  useEffect(() => {
    if (!open) return
    previouslyFocused.current = (document.activeElement as HTMLElement | null) ?? null
    const panel = panelRef.current
    const nodes = panel?.querySelectorAll<HTMLElement>(FOCUSABLE)
    const initial = nodes && nodes.length > 0 ? nodes[0] : panel
    initial?.focus?.()

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.stopPropagation()
        onClose()
        return
      }
      if (event.key !== 'Tab') return
      const list = Array.from(panel?.querySelectorAll<HTMLElement>(FOCUSABLE) ?? []).filter(
        (node) => !node.hasAttribute('disabled'),
      )
      if (list.length === 0) return
      const first = list[0]
      const last = list[list.length - 1]
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault()
        first.focus()
      }
    }

    document.addEventListener('keydown', onKeyDown)
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.removeEventListener('keydown', onKeyDown)
      document.body.style.overflow = previousOverflow
      previouslyFocused.current?.focus?.()
    }
  }, [open, onClose])

  if (!open) return null

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-overlay p-4 [overscroll-behavior:contain] sm:items-center"
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose()
      }}
    >
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-label={title == null ? ariaLabel : undefined}
        aria-labelledby={title != null ? titleId : undefined}
        tabIndex={-1}
        className={cn(
          'relative flex max-h-[85vh] w-full flex-col overflow-hidden rounded-lg border border-border bg-surface text-text shadow-lg outline-none',
          SIZES[size],
        )}
      >
        {title != null && (
          <div className="flex items-start justify-between gap-4 border-b border-border px-4 py-3">
            <div className="min-w-0">
              <h2 id={titleId} className="truncate text-base font-semibold">
                {title}
              </h2>
              {description != null && <p className="mt-0.5 text-xs text-muted">{description}</p>}
            </div>
            <IconButton aria-label="Закрыть" onClick={onClose} size="sm">
              <X aria-hidden className="h-4 w-4" />
            </IconButton>
          </div>
        )}
        {title == null && (
          <div className="absolute right-3 top-3 z-10">
            <IconButton aria-label="Закрыть" onClick={onClose} size="sm">
              <X aria-hidden className="h-4 w-4" />
            </IconButton>
          </div>
        )}
        <div className="min-h-0 flex-1 overflow-y-auto [overscroll-behavior:contain] px-4 py-3">
          {children}
        </div>
        {footer != null && (
          <div className="flex flex-wrap items-center justify-end gap-2 border-t border-border px-4 py-3">
            {footer}
          </div>
        )}
      </div>
    </div>
  )
}
