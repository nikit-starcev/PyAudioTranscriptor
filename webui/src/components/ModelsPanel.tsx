import { useCallback, useEffect, useRef, useState } from 'react'

import {
  api,
  errorMessage,
  formatBytes,
  type ModelEvent,
  type ModelInfo,
  type ModelsResponse,
} from '../api'

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
}

function statusBadge(model: ModelInfo) {
  const download = model.download.status
  if (download === 'downloading') {
    return { label: 'Скачивание…', className: 'bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300' }
  }
  if (download === 'error') {
    return { label: 'Ошибка', className: 'bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300' }
  }
  if (download === 'cancelled') {
    return { label: 'Отменено', className: 'bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300' }
  }
  if (model.status.present) {
    return { label: 'Загружена', className: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300' }
  }
  if (model.status.partial) {
    return { label: 'Частично', className: 'bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300' }
  }
  return { label: 'Не загружена', className: 'bg-slate-100 text-slate-500 dark:bg-slate-800 dark:text-slate-400' }
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
      new Set(data.models.filter((model) => wanted.has(model.id) && !model.status.present).map((model) => model.id)),
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
                      ? { ...model.status, present: true, size: event.bytes_done, missing_files: [], partial: false }
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
    const ids = data?.models.filter((model) => selected.has(model.id) && !model.status.present).map((model) => model.id) ?? []
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

  const selectedModels = data?.models.filter((model) => selected.has(model.id) && !model.status.present) ?? []
  const selectedSize = selectedModels.reduce((total, model) => total + model.approx_size, 0)
  const diskFree = data?.disk.free ?? 0

  return (
    <div className="space-y-3">
      {error && (
        <p className="rounded-md bg-red-50 px-3 py-1.5 text-xs text-red-700 dark:bg-red-950/50 dark:text-red-300">
          {error}
        </p>
      )}

      <div className="flex flex-wrap items-center gap-3 text-xs text-slate-500 dark:text-slate-400">
        <button
          type="button"
          onClick={() => void downloadSelected()}
          disabled={busy !== null || selectedModels.length === 0}
          className="rounded-md bg-slate-800 px-3 py-1.5 text-sm text-white hover:bg-slate-700 disabled:opacity-40 dark:bg-slate-200 dark:text-slate-900 dark:hover:bg-white"
        >
          Скачать выбранные ({selectedModels.length})
        </button>
        <span>
          Объём выбранных: <strong>{formatBytes(selectedSize)}</strong>
        </span>
        {diskFree > 0 && (
          <span>
            Свободно на диске: <strong>{formatBytes(diskFree)}</strong>
          </span>
        )}
      </div>
      <p className="text-xs text-amber-700 dark:text-amber-400">
        Внимание: модели большие (несколько ГБ). Скачивание идёт с Hugging Face; нужен запас места на диске.
      </p>

      {!data ? (
        <p className="py-4 text-center text-sm text-slate-400 dark:text-slate-500">Загрузка списка…</p>
      ) : (
        <ul className="divide-y divide-slate-100 dark:divide-slate-800">
          {data.models.map((model) => {
            const badge = statusBadge(model)
            const downloading = model.download.status === 'downloading'
            const fraction = model.download.fraction ?? 0
            const isRequired = requiredIds.includes(model.id)
            return (
              <li key={model.id} className="py-3">
                <div className="flex items-start gap-3">
                  <input
                    type="checkbox"
                    className="mt-1"
                    checked={selected.has(model.id)}
                    onChange={() => toggle(model.id)}
                    aria-label={`Выбрать ${model.title}`}
                  />
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <p className="text-sm font-medium">{model.title}</p>
                      {isRequired && (
                        <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[10px] uppercase text-slate-500 dark:bg-slate-800 dark:text-slate-400">
                          нужна
                        </span>
                      )}
                      <span className={`rounded-full px-2 py-0.5 text-xs ${badge.className}`}>{badge.label}</span>
                    </div>
                    <p className="text-xs text-slate-400 dark:text-slate-500">
                      {KIND_LABELS[model.kind] ?? model.kind} · ~{formatBytes(model.approx_size)} ·{' '}
                      <a
                        href={`https://huggingface.co/${model.repo}`}
                        target="_blank"
                        rel="noreferrer noopener"
                        className="text-blue-600 underline hover:text-blue-500 dark:text-blue-400 dark:hover:text-blue-300"
                      >
                        {model.repo}
                      </a>
                    </p>
                    {model.note && (
                      <p className="text-xs text-slate-500 dark:text-slate-400">{model.note}</p>
                    )}
                    {downloading && (
                      <div className="mt-1">
                        <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
                          <div
                            className="h-full rounded-full bg-blue-500 transition-all"
                            style={{ width: `${Math.round(fraction * 100)}%` }}
                          />
                        </div>
                        <p className="mt-0.5 text-[11px] text-slate-500 dark:text-slate-400">
                          {formatBytes(model.download.bytes_done)} / {formatBytes(model.download.total)}
                        </p>
                      </div>
                    )}
                    {model.download.error && (
                      <p className="mt-0.5 text-xs text-red-600 dark:text-red-400">{model.download.error}</p>
                    )}
                  </div>
                  <div className="flex shrink-0 flex-col items-end gap-1">
                    {downloading ? (
                      <button
                        type="button"
                        onClick={() => void cancel(model.id)}
                        disabled={busy !== null}
                        className="rounded-md border border-slate-300 px-2 py-1 text-xs hover:bg-slate-100 disabled:opacity-40 dark:border-slate-700 dark:hover:bg-slate-800"
                      >
                        Отменить
                      </button>
                    ) : (
                      <button
                        type="button"
                        onClick={() => void download(model.id)}
                        disabled={busy !== null}
                        className="rounded-md border border-slate-300 px-2 py-1 text-xs hover:bg-slate-100 disabled:opacity-40 dark:border-slate-700 dark:hover:bg-slate-800"
                      >
                        {model.status.partial ? 'Докачать' : 'Скачать'}
                      </button>
                    )}
                    {model.status.present && !downloading && (
                      <button
                        type="button"
                        onClick={() => void remove(model)}
                        disabled={busy !== null}
                        className="rounded-md border border-slate-300 px-2 py-1 text-xs text-slate-500 hover:bg-red-50 hover:text-red-600 disabled:opacity-40 dark:border-slate-700 dark:text-slate-400 dark:hover:bg-red-950/50 dark:hover:text-red-300"
                      >
                        Удалить
                      </button>
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
