import { useCallback, useEffect, useRef, useState } from 'react'
import { Download, HardDrive } from 'lucide-react'

import {
  api,
  errorMessage,
  formatBytes,
  type ModelEvent,
  type ModelInfo,
  type ModelsResponse,
} from '../api'
import {
  Alert,
  Badge,
  Button,
  Checkbox,
  EmptyState,
  ProgressBar,
  Spinner,
  type BadgeTone,
} from './ui'

type Props = {
  /** Идентификаторы моделей, нужных выбранному режиму (предвыбор). */
  requiredIds?: string[]
  /** Вызывается, когда состав загруженных моделей изменился. */
  onChanged?: () => void
}

const KIND_LABELS: Record<string, string> = {
  'whisper-cpp': 'Распознавание (whisper.cpp)',
  llm: 'LLM-постобработка',
  pyannote: 'Диаризация',
  gigaam: 'Распознавание (GigaAM, onnx-asr)',
  sherpa: 'Диаризация (эмбеддинги, sherpa-onnx)',
}

function statusBadge(model: ModelInfo): { label: string; tone: BadgeTone } {
  const download = model.download.status
  if (download === 'downloading') return { label: 'Скачивание…', tone: 'info' }
  if (download === 'error') return { label: 'Ошибка', tone: 'danger' }
  if (download === 'cancelled') return { label: 'Отменено', tone: 'warn' }
  if (model.status.present) return { label: 'Загружена', tone: 'success' }
  if (model.status.partial) return { label: 'Частично', tone: 'warn' }
  return { label: 'Не загружена', tone: 'neutral' }
}

function ModelsPanel({ requiredIds = [], onChanged }: Props) {
  const [data, setData] = useState<ModelsResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const initialized = useRef(false)

  // Колбэк держим в ref: его идентичность не должна пересоздавать EventSource.
  const onChangedRef = useRef(onChanged)
  useEffect(() => {
    onChangedRef.current = onChanged
  }, [onChanged])

  // Номер последнего обработанного SSE-события: сервер может повторно отдать
  // историю при переподключении, а мы её второй раз не применяем.
  const lastSeenSeq = useRef(-1)

  const refresh = useCallback(async () => {
    try {
      const next = await api<ModelsResponse>('/api/models')
      setData(next)
      setError(null)
    } catch (cause) {
      setError(errorMessage(cause))
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  useEffect(() => {
    if (initialized.current || !data) return
    const wanted = new Set(requiredIds)
    setSelected(
      new Set(
        data.models
          .filter((model) => wanted.has(model.id) && !model.status.present)
          .map((model) => model.id),
      ),
    )
    initialized.current = true
  }, [data, requiredIds])

  // Соединение создаётся один раз за время жизни панели. Зависимость только от
  // стабильного `refresh`; сам `onChanged` берётся из ref. Иначе инлайн-стрелка
  // из родителя пересоздавала бы EventSource на каждый рендер (петля #65).
  useEffect(() => {
    const source = new EventSource('/api/models/events')
    source.onmessage = (message) => {
      const event = JSON.parse(message.data) as ModelEvent
      const seq = event.seq
      if (typeof seq === 'number') {
        if (seq <= lastSeenSeq.current) return
        lastSeenSeq.current = seq
      }
      setData((current) => {
        if (!current) return current
        return {
          ...current,
          models: current.models.map((model) =>
            model.id === event.id
              ? {
                  ...model,
                  download: {
                    ...model.download,
                    status: event.status,
                    fraction: event.fraction,
                    bytes_done: event.bytes_done,
                    total: event.total,
                    message: event.message,
                    error: event.status === 'error' ? event.message : null,
                  },
                  status:
                    event.status === 'done'
                      ? {
                          ...model.status,
                          present: true,
                          size: event.bytes_done,
                          missing_files: [],
                          partial: false,
                        }
                      : model.status,
                }
              : model,
          ),
        }
      })
      if (event.status === 'done' || event.status === 'error' || event.status === 'cancelled') {
        void refresh()
        onChangedRef.current?.()
      }
    }
    source.onerror = () => {
      // Соединение переподключится само; ошибку не показываем.
    }
    return () => source.close()
  }, [refresh])

  const toggle = (id: string) => {
    setSelected((current) => {
      const next = new Set(current)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const download = async (id: string) => {
    setBusy(id)
    setError(null)
    try {
      await api(`/api/models/${id}/download`, { method: 'POST' })
      await refresh()
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setBusy(null)
    }
  }

  const downloadSelected = async () => {
    const ids =
      data?.models
        .filter((model) => selected.has(model.id) && !model.status.present)
        .map((model) => model.id) ?? []
    if (ids.length === 0) return
    setBusy('__selected__')
    setError(null)
    try {
      for (const id of ids) {
        try {
          await api(`/api/models/${id}/download`, { method: 'POST' })
        } catch (cause) {
          setError(errorMessage(cause))
        }
      }
      await refresh()
    } finally {
      setBusy(null)
    }
  }

  const cancel = async (id: string) => {
    setBusy(id)
    try {
      await api(`/api/models/${id}/cancel`, { method: 'POST' })
      await refresh()
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setBusy(null)
    }
  }

  const remove = async (model: ModelInfo) => {
    if (!window.confirm(`Удалить локальные файлы модели «${model.title}»?`)) return
    setBusy(model.id)
    try {
      await api(`/api/models/${model.id}`, { method: 'DELETE' })
      await refresh()
      onChanged?.()
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setBusy(null)
    }
  }

  const selectedModels =
    data?.models.filter((model) => selected.has(model.id) && !model.status.present) ?? []
  const selectedSize = selectedModels.reduce((total, model) => total + model.approx_size, 0)
  const diskFree = data?.disk.free ?? 0

  return (
    <div className="space-y-3">
      {error && (
        <Alert tone="danger" live>
          {error}
        </Alert>
      )}

      <div className="flex flex-wrap items-center gap-3 text-xs text-muted">
        <Button
          variant="primary"
          size="sm"
          icon={<Download aria-hidden className="h-4 w-4" />}
          loading={busy === '__selected__'}
          disabled={busy !== null || selectedModels.length === 0}
          onClick={() => void downloadSelected()}
        >
          Скачать выбранные ({selectedModels.length})
        </Button>
        <span>
          Объём выбранных: <strong className="text-text">{formatBytes(selectedSize)}</strong>
        </span>
        {diskFree > 0 && (
          <span className="inline-flex items-center gap-1">
            <HardDrive aria-hidden className="h-3.5 w-3.5" />
            Свободно на диске: <strong className="text-text">{formatBytes(diskFree)}</strong>
          </span>
        )}
      </div>
      <p className="text-xs text-warn">
        Внимание: модели большие (несколько ГБ). Скачивание идёт с Hugging Face; нужен запас
        места на диске.
      </p>

      {!data ? (
        <div className="flex items-center justify-center py-6">
          <Spinner size={20} label="Загрузка списка моделей" />
        </div>
      ) : data.models.length === 0 ? (
        <EmptyState title="Модели не найдены" description="Список моделей пуст" />
      ) : (
        <ul className="divide-y divide-border">
          {data.models.map((model) => {
            const badge = statusBadge(model)
            const downloading = model.download.status === 'downloading'
            const fraction = model.download.fraction ?? 0
            const isRequired = requiredIds.includes(model.id)
            return (
              <li key={model.id} className="py-3">
                <div className="flex items-start gap-3">
                  <Checkbox
                    className="mt-1"
                    checked={selected.has(model.id)}
                    onChange={() => toggle(model.id)}
                    aria-label={`Выбрать ${model.title}`}
                  />
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <p className="text-sm font-medium text-text">{model.title}</p>
                      {isRequired && <Badge tone="neutral">нужна</Badge>}
                      <Badge tone={badge.tone}>{badge.label}</Badge>
                    </div>
                    <p className="text-xs text-muted">
                      {KIND_LABELS[model.kind] ?? model.kind} · ~{formatBytes(model.approx_size)} ·{' '}
                      <a
                        href={`https://huggingface.co/${model.repo}`}
                        target="_blank"
                        rel="noreferrer noopener"
                        className="text-primary underline hover:opacity-80"
                      >
                        {model.repo}
                      </a>
                    </p>
                    {model.note && <p className="text-xs text-muted">{model.note}</p>}
                    {downloading && (
                      <div className="mt-1">
                        <ProgressBar
                          size="sm"
                          value={fraction * 100}
                          label={`Скачивание: ${model.title}`}
                        />
                        <p className="mt-0.5 text-[11px] tabular-nums text-muted">
                          {formatBytes(model.download.bytes_done)} / {formatBytes(model.download.total)}
                        </p>
                      </div>
                    )}
                    {model.download.error && (
                      <p className="mt-0.5 text-xs text-danger">{model.download.error}</p>
                    )}
                  </div>
                  <div className="flex shrink-0 flex-col items-end gap-1">
                    {downloading ? (
                      <Button
                        variant="secondary"
                        size="sm"
                        disabled={busy !== null}
                        onClick={() => void cancel(model.id)}
                      >
                        Отменить
                      </Button>
                    ) : (
                      <Button
                        variant="secondary"
                        size="sm"
                        disabled={busy !== null}
                        onClick={() => void download(model.id)}
                      >
                        {model.status.partial ? 'Докачать' : 'Скачать'}
                      </Button>
                    )}
                    {model.status.present && !downloading && (
                      <Button
                        variant="ghost"
                        size="sm"
                        disabled={busy !== null}
                        onClick={() => void remove(model)}
                      >
                        Удалить
                      </Button>
                    )}
                  </div>
                </div>
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}

export default ModelsPanel
