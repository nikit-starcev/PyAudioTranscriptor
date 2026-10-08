import { useCallback, useEffect, useState } from 'react'

import { api, errorMessage, type GlossaryEntry, type GlossaryQuickRequest } from '../api'
import { Alert, Button, Field, Input, Modal } from './ui'

type Props = {
  open: boolean
  /** Выделенный в стенограмме фрагмент — предзаполняет «ошибочную форму» (#31). */
  term: string
  /** Источник по умолчанию — имя записи/файла активной задачи (#31). */
  source?: string
  onClose: () => void
  /** Термин сохранён в глоссарии — можно обновить список. */
  onSaved?: (entry: GlossaryEntry) => void
}

/**
 * Модальное окно быстрого добавления термина в глоссарий из выделения.
 *
 * Выделение подставляется в «ошибочную форму» (как распозналось), а канон
 * (правильное написание) вводит пользователь — пустой канон не сохраняется.
 * Поле «источник» по умолчанию заполнено именем записи активной задачи,
 * чтобы термин привязывался к встрече (#31).
 */
function GlossaryQuickModal({ open, term, source, onClose, onSaved }: Props) {
  const [canonical, setCanonical] = useState('')
  const [variant, setVariant] = useState(term)
  const [note, setNote] = useState('')
  const [sourceName, setSourceName] = useState(source ?? '')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [saved, setSaved] = useState<GlossaryEntry | null>(null)

  useEffect(() => {
    if (!open) return
    setCanonical('')
    setVariant(term)
    setNote('')
    setSourceName(source ?? '')
    setError(null)
    setSaved(null)
  }, [open, term, source])

  const submit = useCallback(async () => {
    const canonicalValue = canonical.trim()
    if (!canonicalValue) {
      setError('Укажите канон (правильное написание)')
      return
    }
    setBusy(true)
    setError(null)
    try {
      const body: GlossaryQuickRequest = {
        term,
        canonical: canonicalValue,
        variant: variant.trim() || null,
        note: note.trim() || null,
        source: sourceName.trim() || null,
      }
      const entry = await api<GlossaryEntry>('/api/glossary/quick', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      setSaved(entry)
      onSaved?.(entry)
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setBusy(false)
    }
  }, [canonical, term, variant, note, sourceName, onSaved])

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="Добавить в глоссарий"
      size="md"
      footer={
        saved ? (
          <Button variant="primary" onClick={onClose}>
            Готово
          </Button>
        ) : (
          <>
            <Button variant="ghost" onClick={onClose}>
              Отмена
            </Button>
            <Button variant="primary" loading={busy} onClick={() => void submit()}>
              Сохранить
            </Button>
          </>
        )
      }
    >
      <div className="space-y-3">
        <p className="rounded-md bg-surface-2 px-3 py-2 text-xs text-muted">
          Выделенный фрагмент:{' '}
          <span className="font-medium text-text">{term}</span>
        </p>

        {error && (
          <Alert tone="danger" live>
            {error}
          </Alert>
        )}
        {saved && (
          <Alert tone="success" live>
            Сохранено: «{saved.canonical}»{saved.variant ? ` ← ${saved.variant}` : ''} (id {saved.id})
          </Alert>
        )}

        <Field label="Канон (правильно)" htmlFor="quick-glossary-canonical" required>
          <Input
            id="quick-glossary-canonical"
            value={canonical}
            onChange={(event) => setCanonical(event.target.value)}
            placeholder="Правильное написание"
            autoFocus
          />
        </Field>

        <Field label="Ошибочная форма (как распозналось)" htmlFor="quick-glossary-variant">
          <Input
            id="quick-glossary-variant"
            value={variant}
            onChange={(event) => setVariant(event.target.value)}
            placeholder="Как распознано (вариант)"
          />
        </Field>

        <div className="grid gap-3 sm:grid-cols-2">
          <Field label="Источник" htmlFor="quick-glossary-source">
            <Input
              id="quick-glossary-source"
              value={sourceName}
              onChange={(event) => setSourceName(event.target.value)}
              placeholder="Например, встреча"
            />
          </Field>
          <Field label="Заметка" htmlFor="quick-glossary-note">
            <Input
              id="quick-glossary-note"
              value={note}
              onChange={(event) => setNote(event.target.value)}
              placeholder="Комментарий"
            />
          </Field>
        </div>
      </div>
    </Modal>
  )
}

export default GlossaryQuickModal
