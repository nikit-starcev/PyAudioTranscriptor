import { useCallback, useEffect, useState } from 'react'

import {
  api,
  errorMessage,
  formatDuration,
  formatSize,
  type VoiceGroup,
  type VoiceInfo,
} from '../api'

type Props = {
  open: boolean
  onClose: () => void
}

function sampleAudioUrl(filename: string): string {
  return `/api/voices/samples/${encodeURIComponent(filename)}/audio`
}

function VoicesModal({ open, onClose }: Props) {
  const [groups, setGroups] = useState<VoiceGroup[]>([])
  const [loading, setLoading] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [status, setStatus] = useState<string | null>(null)
  const [warning, setWarning] = useState<string | null>(null)
  const [newName, setNewName] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [edit, setEdit] = useState<{ filename: string; name: string; value: string } | null>(null)
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({})

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      setGroups(await api<VoiceGroup[]>('/api/voices'))
      setError(null)
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (open) void refresh()
  }, [open, refresh])

  useEffect(() => {
    if (!open) return
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])

  if (!open) return null

  const upload = async () => {
    const name = newName.trim()
    if (!name) {
      setError('Введите имя образца')
      return
    }
    if (!file) {
      setError('Выберите аудиофайл')
      return
    }
    setBusy(true)
    setError(null)
    setWarning(null)
    try {
      const body = new FormData()
      body.append('file', file)
      body.append('name', name)
      const created = await api<VoiceInfo>('/api/voices', { method: 'POST', body })
      const warnings = created.quality?.warnings ?? []
      if (warnings.length > 0) {
        const speech = created.quality?.speech_seconds
        const speechNote = typeof speech === 'number' ? ` Речь: ${speech.toFixed(1)} с.` : ''
        setWarning(`Загружено: ${created.filename}.${speechNote} ${warnings.join('; ')}.`)
        setStatus(null)
      } else {
        setStatus(`Загружено: ${created.filename}`)
      }
      setNewName('')
      setFile(null)
      await refresh()
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setBusy(false)
    }
  }

  const rename = async (state: { filename: string; name: string; value: string }) => {
    const next = state.value.trim()
    if (!next || next === state.name) {
      setEdit(null)
      return
    }
    setBusy(true)
    setError(null)
    try {
      const response = await fetch(sampleAudioUrl(state.filename))
      if (!response.ok) throw new Error(`Не удалось прочитать образец (${response.status})`)
      const blob = await response.blob()
      const body = new FormData()
      body.append('file', new File([blob], `${next}.wav`, { type: 'audio/wav' }))
      body.append('name', next)
      await api<VoiceInfo>('/api/voices', { method: 'POST', body })
      await api<{ deleted: string }>(
        `/api/voices/samples/${encodeURIComponent(state.filename)}`,
        { method: 'DELETE' },
      )
      setStatus(`${state.filename} → ${next}.wav`)
      setEdit(null)
      await refresh()
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setBusy(false)
    }
  }

  const removeSample = async (filename: string) => {
    if (!window.confirm(`Удалить образец «${filename}» из библиотеки?`)) return
    setBusy(true)
    setError(null)
    try {
      await api<{ deleted: string }>(`/api/voices/samples/${encodeURIComponent(filename)}`, {
        method: 'DELETE',
      })
      setStatus(`Удалено: ${filename}`)
      await refresh()
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setBusy(false)
    }
  }

  const removePerson = async (name: string, count: number) => {
    if (!window.confirm(`Удалить все образцы «${name}» (${count} шт.) из библиотеки?`)) return
    setBusy(true)
    setError(null)
    try {
      await api<{ deleted: string; count: number }>(
        `/api/voices/people/${encodeURIComponent(name)}`,
        { method: 'DELETE' },
      )
      setStatus(`Удалены все образцы: ${name}`)
      await refresh()
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-slate-900/40 p-4 sm:items-center dark:bg-black/60"
      onClick={onClose}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Библиотека голосов"
        className="flex max-h-[85vh] w-full max-w-2xl flex-col overflow-hidden rounded-lg bg-white shadow-xl dark:bg-slate-900"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3 dark:border-slate-800">
          <h2 className="font-medium">Библиотека голосов</h2>
          <button
            type="button"
            onClick={onClose}
            className="rounded-md border border-slate-300 px-2 py-1 text-xs hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800"
          >
            Закрыть
          </button>
        </div>

        <div className="space-y-3 overflow-y-auto px-4 py-3">
          {error && (
            <p className="rounded-md bg-red-50 px-3 py-1.5 text-xs text-red-700 dark:bg-red-950/50 dark:text-red-300">
              {error}
            </p>
          )}
          {status && (
            <p
              role="status"
              className="rounded-md bg-slate-50 px-3 py-1.5 text-xs text-slate-600 dark:bg-slate-800 dark:text-slate-300"
            >
              {status}
            </p>
          )}
          {warning && (
            <p
              role="alert"
              className="rounded-md bg-amber-50 px-3 py-1.5 text-xs text-amber-800 dark:bg-amber-950/50 dark:text-amber-300"
            >
              ⚠ {warning}
            </p>
          )}

          <div className="rounded-md border border-slate-200 p-3 dark:border-slate-800">
            <p className="mb-2 text-xs font-medium text-slate-500 dark:text-slate-400">
              Добавить образец
            </p>
            <div className="flex flex-wrap items-center gap-2">
              <input
                value={newName}
                onChange={(event) => setNewName(event.target.value)}
                placeholder="Имя участника"
                className="w-48 rounded-md border border-slate-300 px-2 py-1 text-sm focus:border-blue-400 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
              />
              <input
                type="file"
                accept="audio/*,.wav"
                onChange={(event) => setFile(event.target.files?.[0] ?? null)}
                className="text-xs"
              />
              <button
                type="button"
                onClick={() => void upload()}
                disabled={busy}
                className="rounded-md bg-slate-800 px-3 py-1.5 text-xs text-white hover:bg-slate-700 disabled:opacity-40 dark:bg-slate-200 dark:text-slate-900 dark:hover:bg-white"
              >
                Загрузить
              </button>
            </div>
            <p className="mt-1 text-xs text-slate-400 dark:text-slate-500">
              WAV сохраняется как есть, другие форматы конвертируются в 16 кГц моно. Повторная
              загрузка к тому же имени добавит образец «(2)», «(3)» и т.д.
            </p>
          </div>

          {loading ? (
            <p className="py-6 text-center text-sm text-slate-400 dark:text-slate-500">
              Загрузка…
            </p>
          ) : groups.length === 0 ? (
            <p className="py-6 text-center text-sm text-slate-400 dark:text-slate-500">
              Библиотека пуста
            </p>
          ) : (
            <ul className="space-y-2">
              {groups.map((group) => {
                const isCollapsed = collapsed[group.name] ?? false
                return (
                  <li
                    key={group.name}
                    className="rounded-md border border-slate-200 dark:border-slate-800"
                  >
                    <div className="flex items-center gap-2 px-3 py-2">
                      <button
                        type="button"
                        onClick={() =>
                          setCollapsed((prev) => ({ ...prev, [group.name]: !isCollapsed }))
                        }
                        className="w-6 shrink-0 text-xs text-slate-400 hover:text-slate-700 dark:text-slate-500 dark:hover:text-slate-200"
                        title={isCollapsed ? 'Развернуть' : 'Свернуть'}
                      >
                        {isCollapsed ? '▸' : '▾'}
                      </button>
                      <span className="min-w-0 truncate text-sm font-medium" title={group.name}>
                        {group.name}
                      </span>
                      <span className="shrink-0 text-xs text-slate-400 dark:text-slate-500">
                        {group.count} обр.
                      </span>
                      <button
                        type="button"
                        onClick={() => void removePerson(group.name, group.count)}
                        disabled={busy}
                        className="ml-auto shrink-0 rounded-md border border-slate-300 px-2 py-1 text-xs text-slate-500 hover:bg-red-50 hover:text-red-600 disabled:opacity-40 dark:border-slate-700 dark:text-slate-400 dark:hover:bg-red-950/50 dark:hover:text-red-300"
                      >
                        Удалить все
                      </button>
                    </div>

                    {!isCollapsed && (
                      <ul className="divide-y divide-slate-100 border-t border-slate-100 dark:divide-slate-800 dark:border-slate-800">
                        {group.samples.map((sample) => (
                          <li key={sample.filename} className="flex flex-col gap-2 px-3 py-2">
                            <div className="flex flex-wrap items-center gap-2">
                              <div className="min-w-0 flex-1">
                                {edit?.filename === sample.filename ? (
                                  <div className="flex flex-wrap items-center gap-1">
                                    <input
                                      autoFocus
                                      value={edit.value}
                                      onChange={(event) =>
                                        setEdit({ ...edit, value: event.target.value })
                                      }
                                      onKeyDown={(event) => {
                                        if (event.key === 'Enter') void rename(edit)
                                        if (event.key === 'Escape') setEdit(null)
                                      }}
                                      className="w-40 min-w-0 max-w-full rounded-md border border-slate-300 px-2 py-1 text-sm focus:border-blue-400 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
                                    />
                                    <button
                                      type="button"
                                      onClick={() => void rename(edit)}
                                      className="rounded-md bg-slate-800 px-2 py-1 text-xs text-white hover:bg-slate-700 dark:bg-slate-200 dark:text-slate-900 dark:hover:bg-white"
                                    >
                                      ОК
                                    </button>
                                    <button
                                      type="button"
                                      onClick={() => setEdit(null)}
                                      className="rounded-md border border-slate-300 px-2 py-1 text-xs hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800"
                                    >
                                      Отмена
                                    </button>
                                  </div>
                                ) : (
                                  <p className="truncate text-xs" title={sample.filename}>
                                    {sample.filename}
                                  </p>
                                )}
                              </div>
                              <span
                                className="tabular-nums text-xs text-slate-400 dark:text-slate-500"
                                title="Длительность"
                              >
                                {formatDuration(sample.duration)}
                              </span>
                              <span
                                className="tabular-nums text-xs text-slate-400 dark:text-slate-500"
                                title="Размер файла"
                              >
                                {formatSize(sample.size)}
                              </span>
                              <div className="flex items-center gap-1">
                                <button
                                  type="button"
                                  onClick={() =>
                                    setEdit({
                                      filename: sample.filename,
                                      name: sample.name,
                                      value: sample.name,
                                    })
                                  }
                                  className="rounded-md border border-slate-300 px-2 py-1 text-xs hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800"
                                >
                                  Переименовать
                                </button>
                                <button
                                  type="button"
                                  onClick={() => void removeSample(sample.filename)}
                                  disabled={busy}
                                  className="rounded-md border border-slate-300 px-2 py-1 text-xs text-slate-500 hover:bg-red-50 hover:text-red-600 disabled:opacity-40 dark:border-slate-700 dark:text-slate-400 dark:hover:bg-red-950/50 dark:hover:text-red-300"
                                >
                                  Удалить
                                </button>
                              </div>
                            </div>
                            <audio
                              controls
                              preload="none"
                              className="h-8 w-full sm:max-w-md"
                              src={sampleAudioUrl(sample.filename)}
                            />
                          </li>
                        ))}
                      </ul>
                    )}
                  </li>
                )
              })}
            </ul>
          )}
        </div>
      </div>
    </div>
  )
}

export default VoicesModal
