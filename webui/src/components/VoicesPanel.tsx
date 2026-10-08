import { useCallback, useEffect, useState } from 'react'
import { Check, ChevronDown, ChevronRight, GitMerge, Pencil, Search, Trash2 } from 'lucide-react'

import {
  api,
  errorMessage,
  formatBytes,
  formatDuration,
  type VoiceDedupReport,
  type VoiceDuplicateGroup,
  type VoiceDuplicateKind,
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
  FileInput,
  IconButton,
  Input,
  Spinner,
  Tooltip,
} from './ui'

function sampleAudioUrl(filename: string): string {
  return `/api/voices/samples/${encodeURIComponent(filename)}/audio`
}

const DUP_KIND_LABEL: Record<VoiceDuplicateKind, string> = {
  exact: 'Точные',
  embedding: 'По голосу',
  audio: 'Похожие',
}

const DUP_KIND_TONE: Record<VoiceDuplicateKind, 'danger' | 'info' | 'warn'> = {
  exact: 'danger',
  embedding: 'info',
  audio: 'warn',
}

function dedupKey(group: VoiceDuplicateGroup): string {
  return group.members
    .map((member) => member.filename)
    .sort()
    .join('|')
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
  const [dedup, setDedup] = useState<VoiceDedupReport | null>(null)
  const [dedupBusy, setDedupBusy] = useState(false)
  const [dedupKeep, setDedupKeep] = useState<Record<string, string>>({})

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
      const similar = created.similar ?? []
      const notes: string[] = []
      if (similar.length > 0) {
        const names = similar
          .map((item) => `«${item.name}» (~${Math.round(item.score * 100)}%)`)
          .join(', ')
        notes.push(`похожий образец уже есть: ${names}`)
      }
      if (warnings.length > 0) {
        const speech = created.quality?.speech_seconds
        const speechNote = typeof speech === 'number' ? `речь ${speech.toFixed(1)} с` : ''
        notes.push([speechNote, warnings.join('; ')].filter(Boolean).join('; '))
      }
      if (notes.length > 0) {
        setWarning(`Загружено: ${created.filename}. ${notes.join('. ')}.`)
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

  const runDedup = useCallback(async () => {
    setDedupBusy(true)
    setError(null)
    try {
      const report = await api<VoiceDedupReport>('/api/voices/dedup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ embeddings: true }),
      })
      setDedup(report)
      setDedupKeep({})
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setDedupBusy(false)
    }
  }, [])

  const resolveGroup = async (group: VoiceDuplicateGroup, keep: string) => {
    const victims = group.members
      .map((member) => member.filename)
      .filter((filename) => filename !== keep)
    if (victims.length === 0) return
    if (!window.confirm(`Удалить дубликатов: ${victims.length}, оставив «${keep}»?`)) return
    setBusy(true)
    setError(null)
    try {
      for (const filename of victims) {
        await api<{ deleted: string }>(`/api/voices/samples/${encodeURIComponent(filename)}`, {
          method: 'DELETE',
        })
      }
      setStatus(`Удалено дубликатов: ${victims.length}`)
      await refresh()
      await runDedup()
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setBusy(false)
    }
  }

  const mergeGroup = async (group: VoiceDuplicateGroup, keep: string) => {
    const keeper = group.members.find((member) => member.filename === keep) ?? group.members[0]
    const target = keeper.name
    const sources = group.names.filter((name) => name !== target)
    if (sources.length === 0) return
    if (!window.confirm(`Перенести образцы «${sources.join(', ')}» в «${target}»?`)) return
    setBusy(true)
    setError(null)
    try {
      for (const source of sources) {
        await api<{ count: number }>('/api/voices/dedup/merge', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ source, target }),
        })
      }
      setStatus(`Объединено под именем «${target}»`)
      await refresh()
      await runDedup()
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
              <FileInput
                id="voice-file"
                accept="audio/*,.wav"
                value={file}
                onChange={setFile}
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
          title="Дубликаты"
          description={
            dedup
              ? `Проверено образцов: ${dedup.scanned}`
              : 'Точные и похожие записи: проверьте перед удалением'
          }
          actions={
            <Button
              variant="secondary"
              loading={dedupBusy}
              icon={<Search aria-hidden className="h-4 w-4" />}
              onClick={() => void runDedup()}
            >
              Найти дубликаты
            </Button>
          }
        />
        <CardContent className="space-y-3">
          {dedup?.error && (
            <Alert tone="warn" live onDismiss={() => setDedup(null)}>
              {dedup.error}
            </Alert>
          )}
          {dedup && !dedup.error && dedup.groups.length === 0 && (
            <p className="text-sm text-muted">Дубликаты не найдены.</p>
          )}
          {dedup &&
            dedup.groups.map((group) => {
              const key = dedupKey(group)
              const keep = dedupKeep[key] ?? group.keep
              return (
                <div key={key} className="rounded-md border border-border p-3">
                  <div className="mb-2 flex flex-wrap items-center gap-2">
                    <Badge tone={DUP_KIND_TONE[group.kind]}>{DUP_KIND_LABEL[group.kind]}</Badge>
                    <span className="text-xs text-muted">
                      совпадение {Math.round(group.score * 100)}%
                    </span>
                    <span className="min-w-0 truncate text-sm text-text">
                      {group.names.join(' / ')}
                    </span>
                    <span className="text-xs text-muted">· {group.members.length} обр.</span>
                  </div>
                  <ul className="divide-y divide-border">
                    {group.members.map((member) => {
                      const isKeep = member.filename === keep
                      return (
                        <li
                          key={member.filename}
                          className="flex flex-wrap items-center gap-2 py-2"
                        >
                          <button
                            type="button"
                            aria-label={`Оставить ${member.filename}`}
                            aria-pressed={isKeep}
                            title="Оставить этот образец"
                            className={`inline-flex h-6 w-6 items-center justify-center rounded-full border transition-colors ${
                              isKeep
                                ? 'border-primary bg-primary text-primary-fg'
                                : 'border-border-strong text-transparent hover:border-primary'
                            }`}
                            onClick={() => setDedupKeep((prev) => ({ ...prev, [key]: member.filename }))}
                          >
                            <Check aria-hidden className="h-3.5 w-3.5" />
                          </button>
                          <div className="min-w-0 flex-1">
                            <p className="truncate text-xs text-text" title={member.filename}>
                              {member.name} — {member.filename}
                            </p>
                          </div>
                          <span className="tabular-nums text-xs text-muted" title="Длительность">
                            {formatDuration(member.duration)}
                          </span>
                          <span className="tabular-nums text-xs text-muted" title="Размер файла">
                            {formatBytes(member.size)}
                          </span>
                        </li>
                      )
                    })}
                  </ul>
                  <div className="mt-2 flex flex-wrap gap-2">
                    <Button
                      variant="danger"
                      size="sm"
                      icon={<Trash2 aria-hidden className="h-3.5 w-3.5" />}
                      disabled={busy || dedupBusy}
                      onClick={() => void resolveGroup(group, keep)}
                    >
                      Удалить дубликаты
                    </Button>
                    {group.names.length > 1 && (
                      <Button
                        variant="secondary"
                        size="sm"
                        icon={<GitMerge aria-hidden className="h-3.5 w-3.5" />}
                        disabled={busy || dedupBusy}
                        onClick={() => void mergeGroup(group, keep)}
                      >
                        Объединить имена
                      </Button>
                    )}
                  </div>
                </div>
              )
            })}
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
                        variant="danger"
                        size="sm"
                        className="ml-auto"
                        icon={<Trash2 aria-hidden className="h-3.5 w-3.5" />}
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
                                  icon={<Pencil aria-hidden className="h-3.5 w-3.5" />}
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
                                <Tooltip label="Удалить образец">
                                  <IconButton
                                    aria-label={`Удалить образец ${sample.filename}`}
                                    size="sm"
                                    className="text-danger"
                                    disabled={busy}
                                    onClick={() => void removeSample(sample.filename)}
                                  >
                                    <Trash2 aria-hidden className="h-4 w-4" />
                                  </IconButton>
                                </Tooltip>
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
