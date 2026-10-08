import { useCallback, useEffect, useState } from 'react'
import { Check } from 'lucide-react'

import {
  activateSummaryPrompt,
  createSummaryPrompt,
  deleteSummaryPrompt,
  errorMessage,
  fetchSummaryPrompts,
  updateSummaryPrompt,
  type SummaryPrompt,
} from '../api'
import {
  Alert,
  Badge,
  Button,
  Card,
  CardContent,
  CardHeader,
  EmptyState,
  Field,
  Input,
  Spinner,
  Textarea,
} from './ui'

type Props = {
  /** Вызывается при смене активного шаблона (id или null). */
  onChanged?: (activeId: number | null) => void
}

type PromptDraft = { name: string; body: string }

const EMPTY_DRAFT: PromptDraft = { name: '', body: '' }

/**
 * Панель управления пользовательскими шаблонами промпта резюме (#97):
 * список, создание/редактирование/удаление, выбор активного. Активный шаблон
 * применяется к задачам и по кнопке «Сформировать протокол».
 */
function SummaryPromptsPanel({ onChanged }: Props) {
  const [prompts, setPrompts] = useState<SummaryPrompt[]>([])
  const [activeId, setActiveId] = useState<number | null>(null)
  const [loading, setLoading] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [status, setStatus] = useState<string | null>(null)
  const [draft, setDraft] = useState<PromptDraft>(EMPTY_DRAFT)
  const [edit, setEdit] = useState<{ id: number; name: string; body: string } | null>(null)

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      const data = await fetchSummaryPrompts()
      setPrompts(data.prompts)
      setActiveId(data.active_id)
      setError(null)
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const run = async (action: () => Promise<unknown>, message: string) => {
    setBusy(true)
    setError(null)
    setStatus(null)
    try {
      await action()
      setStatus(message)
      await refresh()
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setBusy(false)
    }
  }

  const addPrompt = () =>
    run(async () => {
      const name = draft.name.trim()
      const body = draft.body.trim()
      if (!name) throw new Error('Укажите имя шаблона')
      if (!body) throw new Error('Тело шаблона не может быть пустым')
      await createSummaryPrompt(name, body)
      setDraft(EMPTY_DRAFT)
    }, 'Шаблон добавлен')

  const saveEdit = () => {
    if (!edit) return
    const current = edit
    void run(async () => {
      const body = current.body.trim()
      if (!body) throw new Error('Тело шаблона не может быть пустым')
      await updateSummaryPrompt(current.id, { name: current.name.trim(), body })
      setEdit(null)
    }, 'Шаблон сохранён')
  }

  const removePrompt = (prompt: SummaryPrompt) => {
    if (!window.confirm(`Удалить шаблон «${prompt.name}»?`)) return
    void run(async () => {
      const result = await deleteSummaryPrompt(prompt.id)
      setActiveId(result.active_id)
      onChanged?.(result.active_id)
    }, 'Шаблон удалён')
  }

  const activate = (prompt: SummaryPrompt) => {
    void run(async () => {
      const result = await activateSummaryPrompt(prompt.id)
      setActiveId(result.active_id)
      onChanged?.(result.active_id)
    }, `Активный шаблон: «${prompt.name}»`)
  }

  return (
    <Card>
      <CardHeader
        title="Промпты резюме"
        description="Активный шаблон используется при постановке задачи и по кнопке «Сформировать протокол»"
      />
      <CardContent className="space-y-4">
        {error && (
          <Alert tone="danger" live onDismiss={() => setError(null)}>
            {error}
          </Alert>
        )}
        {status && (
          <Alert tone="success" live onDismiss={() => setStatus(null)}>
            {status}
          </Alert>
        )}

        {loading ? (
          <div className="flex items-center justify-center py-6">
            <Spinner size={20} label="Загрузка шаблонов" />
          </div>
        ) : prompts.length === 0 ? (
          <EmptyState title="Шаблонов пока нет" description="Добавьте свой шаблон ниже" />
        ) : (
          <ul className="divide-y divide-border">
            {prompts.map((prompt) => {
              const isActive = prompt.id === activeId
              const isEditing = edit?.id === prompt.id
              return (
                <li key={prompt.id} className="py-3">
                  <div className="flex flex-wrap items-center gap-2">
                    <label className="flex min-w-0 flex-1 cursor-pointer items-center gap-2 text-sm">
                      <input
                        type="radio"
                        name="active-prompt"
                        checked={isActive}
                        disabled={busy}
                        onChange={() => activate(prompt)}
                        className="h-4 w-4 shrink-0 accent-primary focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"
                      />
                      <span className="min-w-0 break-words font-medium text-text">{prompt.name}</span>
                    </label>
                    {prompt.builtin && <Badge tone="neutral">встроенный</Badge>}
                    {isActive && (
                      <Badge tone="success" icon={<Check aria-hidden className="h-3 w-3" />}>
                        активный
                      </Badge>
                    )}
                    {!isEditing && (
                      <>
                        <Button
                          variant="secondary"
                          size="sm"
                          disabled={busy}
                          onClick={() =>
                            setEdit({ id: prompt.id, name: prompt.name, body: prompt.body })
                          }
                        >
                          Изменить
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          disabled={busy}
                          onClick={() => removePrompt(prompt)}
                        >
                          Удалить
                        </Button>
                      </>
                    )}
                  </div>

                  {isEditing && edit ? (
                    <div className="mt-2 space-y-2 rounded-md border border-border p-3">
                      <Field label="Имя" htmlFor={`edit-prompt-name-${edit.id}`}>
                        <Input
                          id={`edit-prompt-name-${edit.id}`}
                          value={edit.name}
                          onChange={(event) => setEdit({ ...edit, name: event.target.value })}
                        />
                      </Field>
                      <Field label="Текст промпта" htmlFor={`edit-prompt-body-${edit.id}`}>
                        <Textarea
                          id={`edit-prompt-body-${edit.id}`}
                          value={edit.body}
                          className="font-mono"
                          onChange={(event) => setEdit({ ...edit, body: event.target.value })}
                        />
                      </Field>
                      <div className="flex items-center gap-2">
                        <Button variant="primary" size="sm" disabled={busy} onClick={saveEdit}>
                          Сохранить
                        </Button>
                        <Button variant="ghost" size="sm" disabled={busy} onClick={() => setEdit(null)}>
                          Отмена
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <p className="mt-1 line-clamp-3 whitespace-pre-wrap text-xs text-muted">
                      {prompt.body}
                    </p>
                  )}
                </li>
              )
            })}
          </ul>
        )}

        <fieldset className="space-y-3 rounded-md border border-border p-3">
          <legend className="px-1 text-xs font-medium text-muted">Новый шаблон</legend>
          <Field label="Имя" htmlFor="new-prompt-name">
            <Input
              id="new-prompt-name"
              value={draft.name}
              onChange={(event) => setDraft({ ...draft, name: event.target.value })}
              placeholder="Например, «Кратко для email»"
            />
          </Field>
          <Field label="Текст промпта" htmlFor="new-prompt-body" hint="Системная инструкция для модели">
            <Textarea
              id="new-prompt-body"
              value={draft.body}
              className="font-mono"
              onChange={(event) => setDraft({ ...draft, body: event.target.value })}
              placeholder="Ты — ассистент, который…"
            />
          </Field>
          <Button variant="primary" disabled={busy} onClick={addPrompt}>
            Добавить шаблон
          </Button>
        </fieldset>
      </CardContent>
    </Card>
  )
}

export default SummaryPromptsPanel
