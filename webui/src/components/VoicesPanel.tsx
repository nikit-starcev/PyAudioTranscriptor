import { useCallback, useEffect, useState } from 'react'
import { ChevronDown, ChevronRight } from 'lucide-react'

import {
  api,
  errorMessage,
  formatBytes,
  formatDuration,
  type VoiceGroup,
  type VoiceInfo,
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
  IconButton,
  Input,
  Spinner,
} from './ui'

function sampleAudioUrl(filename: string): string {
  return `/api/voices/samples/${encodeURIComponent(filename)}/audio`
}

function VoicesPanel() {
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
    void refresh()
  }, [refresh])

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
    <div className="space-y-5">
      {error && (
        <Alert tone="danger" live onDismiss={() => setError(null)}>
          {error}
        </Alert>
      )}
      {status && (
        <Alert tone="info" live onDismiss={() => setStatus(null)}>
          {status}
        </Alert>
      )}
      {warning && (
        <Alert tone="warn" live onDismiss={() => setWarning(null)}>
          {warning}
        </Alert>
      )}

      <Card>
        <CardHeader title="Добавить образец" description="WAV-клип или аудио другого формата" />
        <CardContent className="space-y-3">
          <div className="flex flex-wrap items-end gap-3">
            <Field label="Имя участника" htmlFor="voice-new-name">
              <Input
                id="voice-new-name"
                value={newName}
                onChange={(event) => setNewName(event.target.value)}
                placeholder="Имя участника"
                className="w-48"
              />
            </Field>
            <Field label="Файл" htmlFor="voice-file">
              <input
                id="voice-file"
                type="file"
                accept="audio/*,.wav"
                onChange={(event) => setFile(event.target.files?.[0] ?? null)}
                className="block w-full text-xs text-muted file:mr-3 file:rounded-md file:border-0 file:bg-surface-2 file:px-3 file:py-1.5 file:text-xs file:font-medium file:text-text focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"
              />
            </Field>
            <Button variant="primary" loading={busy} onClick={() => void upload()}>
              Загрузить
            </Button>
          </div>
          <p className="text-xs text-muted">
            WAV сохраняется как есть, другие форматы конвертируются в 16 кГц моно. Повторная
            загрузка к тому же имени добавит образец «(2)», «(3)» и т.д.
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader
          title="Образцы"
          description={loading ? 'Загрузка…' : `Групп: ${groups.length}`}
        />
        <CardContent>
          {loading ? (
            <div className="flex items-center justify-center py-6">
              <Spinner size={20} label="Загрузка библиотеки голосов" />
            </div>
          ) : groups.length === 0 ? (
            <EmptyState
              title="Библиотека пуста"
              description="Добавьте образцы голоса участников"
            />
          ) : (
            <ul className="space-y-2">
              {groups.map((group) => {
                const isCollapsed = collapsed[group.name] ?? false
                return (
                  <li key={group.name} className="rounded-md border border-border">
                    <div className="flex items-center gap-2 px-3 py-2">
                      <IconButton
                        aria-label={isCollapsed ? `Развернуть ${group.name}` : `Свернуть ${group.name}`}
                        size="sm"
                        onClick={() =>
                          setCollapsed((prev) => ({ ...prev, [group.name]: !isCollapsed }))
                        }
                      >
                        {isCollapsed ? (
                          <ChevronRight aria-hidden className="h-4 w-4" />
                        ) : (
                          <ChevronDown aria-hidden className="h-4 w-4" />
                        )}
                      </IconButton>
                      <span className="min-w-0 truncate text-sm font-medium text-text" title={group.name}>
                        {group.name}
                      </span>
                      <Badge tone="neutral">{group.count} обр.</Badge>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="ml-auto"
                        disabled={busy}
                        onClick={() => void removePerson(group.name, group.count)}
                      >
                        Удалить все
                      </Button>
                    </div>

                    {!isCollapsed && (
                      <ul className="divide-y divide-border border-t border-border">
                        {group.samples.map((sample) => (
                          <li key={sample.filename} className="flex flex-col gap-2 px-3 py-2">
                            <div className="flex flex-wrap items-center gap-2">
                              <div className="min-w-0 flex-1">
                                {edit?.filename === sample.filename ? (
                                  <div className="flex flex-wrap items-center gap-1">
                                    <Input
                                      autoFocus
                                      aria-label="Новое имя образца"
                                      value={edit.value}
                                      onChange={(event) =>
                                        setEdit({ ...edit, value: event.target.value })
                                      }
                                      onKeyDown={(event) => {
                                        if (event.key === 'Enter') void rename(edit)
                                        if (event.key === 'Escape') setEdit(null)
                                      }}
                                      className="w-40"
                                    />
                                    <Button
                                      variant="primary"
                                      size="sm"
                                      onClick={() => void rename(edit)}
                                    >
                                      ОК
                                    </Button>
                                    <Button variant="ghost" size="sm" onClick={() => setEdit(null)}>
                                      Отмена
                                    </Button>
                                  </div>
                                ) : (
                                  <p className="truncate text-xs" title={sample.filename}>
                                    {sample.filename}
                                  </p>
                                )}
                              </div>
                              <span className="tabular-nums text-xs text-muted" title="Длительность">
                                {formatDuration(sample.duration)}
                              </span>
                              <span className="tabular-nums text-xs text-muted" title="Размер файла">
                                {formatBytes(sample.size)}
                              </span>
                              <div className="flex items-center gap-1">
                                <Button
                                  variant="secondary"
                                  size="sm"
                                  onClick={() =>
                                    setEdit({
                                      filename: sample.filename,
                                      name: sample.name,
                                      value: sample.name,
                                    })
                                  }
                                >
                                  Переименовать
                                </Button>
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  disabled={busy}
                                  onClick={() => void removeSample(sample.filename)}
                                >
                                  Удалить
                                </Button>
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
        </CardContent>
      </Card>
    </div>
  )
}

export default VoicesPanel
