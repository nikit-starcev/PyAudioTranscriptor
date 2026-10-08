import { useMemo, useState } from 'react'
import { ArrowRight } from 'lucide-react'

import {
  collectSpeakers,
  describeApply,
  errorMessage,
  formatDuration,
  type ApplyNamesResponse,
  type LibraryWindow,
  type ReassignRequest,
  type ReassignResponse,
  type SampleMeta,
  type TranscriptResult,
  type VoiceInfo,
} from '../api'
import { Alert, Badge, Button, Card, CardHeader, IconButton, Input, Select, Spinner } from './ui'
import SpeakerVariants from './SpeakerVariants'

type Props = {
  jobId: string
  result: TranscriptResult
  sampleMeta: Record<string, SampleMeta>
  onRename: (speakerId: string, name: string) => Promise<void>
  onMerge: (source: string, target: string) => Promise<void>
  onToLibrary: (speakerId: string, name: string, window?: LibraryWindow) => Promise<VoiceInfo>
  onReassign: (speakerId: string, body: ReassignRequest) => Promise<ReassignResponse>
  onUndo: () => Promise<void>
  undoAvailable: boolean
  onApplyNames: () => Promise<ApplyNamesResponse>
  onOpenVoices: () => void
}

type EditState = { sid: string; kind: 'rename' | 'library'; value: string }

function SpeakersPanel({
  jobId,
  result,
  sampleMeta,
  onRename,
  onMerge,
  onToLibrary,
  onReassign,
  onUndo,
  undoAvailable,
  onApplyNames,
  onOpenVoices,
}: Props) {
  const [busy, setBusy] = useState(false)
  const [applyingNames, setApplyingNames] = useState(false)
  const [status, setStatus] = useState<{ kind: 'info' | 'error'; text: string } | null>(null)
  const [edit, setEdit] = useState<EditState | null>(null)
  const [mergeTarget, setMergeTarget] = useState<Record<string, string>>({})

  //: Все говорящие результата (включая упомянутых только как участники
  //: наложения) в устойчивом порядке SPEAKER_00, SPEAKER_01, …
  const speakers = useMemo(() => collectSpeakers(result), [result])

  //: Реплики, где говорящий основной, и отдельно — где он участник наложения.
  //: Раньше считались только основные: говорящий, встречающийся лишь в
  //: наложении, показывался как «0 реплик» и выглядел отсутствующим.
  const { counts, extraCounts } = useMemo(() => {
    const primary: Record<string, number> = {}
    const extra: Record<string, number> = {}
    for (const entry of result.entries) {
      if (entry.speaker_id) primary[entry.speaker_id] = (primary[entry.speaker_id] ?? 0) + 1
      for (const id of entry.extra_speaker_ids) {
        extra[id] = (extra[id] ?? 0) + 1
      }
    }
    return { counts: primary, extraCounts: extra }
  }, [result.entries])

  const run = async (task: () => Promise<void>) => {
    setBusy(true)
    setStatus(null)
    try {
      await task()
    } catch (cause) {
      setStatus({ kind: 'error', text: errorMessage(cause) })
    } finally {
      setBusy(false)
    }
  }

  const submitEdit = (state: EditState) => {
    const name = state.value.trim()
    if (!name) {
      setStatus({ kind: 'error', text: 'Введите имя' })
      return
    }
    void run(async () => {
      if (state.kind === 'rename') {
        await onRename(state.sid, name)
        setStatus({ kind: 'info', text: `${state.sid} → ${name}` })
      } else {
        const voice = await onToLibrary(state.sid, name)
        const warnings = voice.quality?.warnings ?? []
        const suffix = warnings.length > 0 ? ` — ⚠ ${warnings.join('; ')}` : ''
        setStatus({
          kind: 'info',
          text: `Сохранено в библиотеку: ${voice.name}.wav${suffix}`,
        })
      }
      setEdit(null)
    })
  }

  const applyNames = () => {
    const total = speakers.length
    void run(async () => {
      setApplyingNames(true)
      try {
        const response = await onApplyNames()
        setStatus({
          kind: response.error ? 'error' : 'info',
          text: describeApply(response, total),
        })
      } finally {
        setApplyingNames(false)
      }
    })
  }

  const merge = (speakerId: string) => {
    const target = mergeTarget[speakerId]
    if (!target) return
    void run(async () => {
      await onMerge(speakerId, target)
      setStatus({ kind: 'info', text: `${speakerId} объединён в ${target}` })
      setMergeTarget((prev) => ({ ...prev, [speakerId]: '' }))
    })
  }

  const undo = () => {
    void run(async () => {
      await onUndo()
      setStatus({ kind: 'info', text: 'Последний перенос окна отменён' })
    })
  }

  return (
    <Card>
      <CardHeader
        title="Говорящие"
        description={`${speakers.length} шт.`}
        actions={
          <>
            {undoAvailable && (
              <Button
                variant="secondary"
                size="sm"
                disabled={busy}
                title="Отменить последний перенос окна (#40/#41)"
                onClick={undo}
              >
                Отменить перенос
              </Button>
            )}
            <Button
              variant="primary"
              size="sm"
              loading={applyingNames}
              disabled={busy || speakers.length === 0}
              onClick={applyNames}
            >
              Применить имена
            </Button>
            <Button variant="secondary" size="sm" onClick={onOpenVoices}>
              Библиотека голосов
            </Button>
          </>
        }
      />
      <div className="space-y-2 p-4">
        {status && (
          <Alert
            tone={status.kind === 'error' ? 'danger' : 'info'}
            live
            onDismiss={() => setStatus(null)}
          >
            {status.text}
          </Alert>
        )}

        {busy && !applyingNames && (
          <p className="flex items-center gap-2 text-xs text-muted">
            <Spinner size={14} /> Обработка…
          </p>
        )}

        <ul className="space-y-2">
          {speakers.map((speaker) => {
            const editing = edit?.sid === speaker.id ? edit : null
            const duration = sampleMeta[speaker.id]?.duration
            const others = speakers.filter((item) => item.id !== speaker.id)
            return (
              <li
                key={speaker.id}
                className="flex flex-col gap-2 rounded-md border border-border bg-surface-2/40 px-3 py-2 sm:grid sm:grid-cols-[minmax(0,1fr)_auto] sm:items-center sm:gap-x-3 sm:gap-y-2"
              >
                <div className="flex min-w-0 items-center gap-2 sm:col-start-1 sm:row-start-1">
                  <span className="shrink-0" title={speaker.id}>
                    <Badge tone="neutral" className="font-mono">
                      {speaker.id}
                    </Badge>
                  </span>

                  {editing?.kind === 'rename' ? (
                    <div className="flex flex-wrap items-center gap-1">
                      <Input
                        autoFocus
                        aria-label="Новое имя говорящего"
                        value={editing.value}
                        onChange={(event) => setEdit({ ...editing, value: event.target.value })}
                        onKeyDown={(event) => {
                          if (event.key === 'Enter') submitEdit(editing)
                          if (event.key === 'Escape') setEdit(null)
                        }}
                        className="w-48"
                      />
                      <Button variant="primary" size="sm" onClick={() => submitEdit(editing)}>
                        ОК
                      </Button>
                      <Button variant="ghost" size="sm" onClick={() => setEdit(null)}>
                        Отмена
                      </Button>
                    </div>
                  ) : (
                    <>
                      <span
                        className="min-w-0 break-words text-sm font-medium leading-snug text-text"
                        title={speaker.display_name}
                      >
                        {speaker.display_name}
                      </span>
                      <IconButton
                        aria-label={`Переименовать говорящего ${speaker.display_name}`}
                        size="sm"
                        onClick={() =>
                          setEdit({ sid: speaker.id, kind: 'rename', value: speaker.display_name })
                        }
                      >
                        <span aria-hidden className="text-sm">
                          ✎
                        </span>
                      </IconButton>
                    </>
                  )}
                </div>

                {/* Служебная строка: реплики · образец · длительность — фиксированные слоты,
                    чтобы значения не «прыгали» при разной длине имени. */}
                <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted sm:col-span-2 sm:row-start-2">
                  <span className="tabular-nums" title="Реплики, где говорящий основной">
                    {counts[speaker.id] ?? 0} реплик
                  </span>
                  {(extraCounts[speaker.id] ?? 0) > 0 && (
                    <span className="tabular-nums" title="Реплики, где говорящий — участник наложения">
                      +{extraCounts[speaker.id]} в наложении
                    </span>
                  )}
                  <span>Образец {speaker.has_sample ? '✓' : '—'}</span>
                  {speaker.has_sample && (
                    <span className="tabular-nums" title="Длительность образца">
                      {formatDuration(duration ?? null)}
                    </span>
                  )}
                </div>

                <div className="flex flex-wrap items-center gap-2 sm:col-start-2 sm:row-start-1 sm:flex-nowrap sm:justify-self-end">
                  {editing?.kind === 'library' ? (
                    <div className="flex flex-wrap items-center gap-1">
                      <Input
                        autoFocus
                        aria-label="Имя образца"
                        value={editing.value}
                        onChange={(event) => setEdit({ ...editing, value: event.target.value })}
                        onKeyDown={(event) => {
                          if (event.key === 'Enter') submitEdit(editing)
                          if (event.key === 'Escape') setEdit(null)
                        }}
                        className="w-40"
                      />
                      <Button variant="primary" size="sm" onClick={() => submitEdit(editing)}>
                        Сохранить
                      </Button>
                      <Button variant="ghost" size="sm" onClick={() => setEdit(null)}>
                        Отмена
                      </Button>
                    </div>
                  ) : (
                    <Button
                      variant="secondary"
                      size="sm"
                      disabled={!speaker.has_sample || busy}
                      title={speaker.has_sample ? undefined : 'Нет образца голоса'}
                      onClick={() =>
                        setEdit({ sid: speaker.id, kind: 'library', value: speaker.display_name })
                      }
                    >
                      В библиотеку
                    </Button>
                  )}

                  {others.length > 0 && (
                    <div className="flex items-center gap-1">
                      <Select
                        aria-label={`Объединить говорящего ${speaker.display_name}`}
                        value={mergeTarget[speaker.id] ?? ''}
                        onChange={(event) =>
                          setMergeTarget((prev) => ({ ...prev, [speaker.id]: event.target.value }))
                        }
                        className="max-w-48"
                      >
                        <option value="">Объединить в…</option>
                        {others.map((item) => (
                          <option key={item.id} value={item.id}>
                            {item.display_name} ({item.id})
                          </option>
                        ))}
                      </Select>
                      <Button
                        variant="secondary"
                        size="sm"
                        disabled={!mergeTarget[speaker.id] || busy}
                        aria-label="Объединить"
                        icon={<ArrowRight aria-hidden className="h-4 w-4" />}
                        onClick={() => merge(speaker.id)}
                      >
                        Объединить
                      </Button>
                    </div>
                  )}
                </div>

                {speaker.has_sample && (
                  <audio
                    controls
                    preload="none"
                    className="h-8 w-full sm:col-span-2 sm:row-start-3 sm:max-w-md"
                    src={`/api/jobs/${jobId}/samples/${encodeURIComponent(speaker.id)}`}
                  />
                )}

                <SpeakerVariants
                  jobId={jobId}
                  speakerId={speaker.id}
                  speakerName={speaker.display_name}
                  speakers={speakers}
                  onToLibrary={onToLibrary}
                  onReassign={onReassign}
                  onRename={onRename}
                />
              </li>
            )
          })}
        </ul>
      </div>
    </Card>
  )
}

export default SpeakersPanel
