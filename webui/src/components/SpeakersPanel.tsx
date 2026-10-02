import { useMemo, useState } from 'react'

import {
  collectSpeakers,
  describeApply,
  errorMessage,
  formatDuration,
  type ApplyNamesResponse,
  type LibraryWindow,
  type SampleMeta,
  type TranscriptResult,
  type VoiceInfo,
} from '../api'
import SpeakerVariants from './SpeakerVariants'

type Props = {
  jobId: string
  result: TranscriptResult
  sampleMeta: Record<string, SampleMeta>
  onRename: (speakerId: string, name: string) => Promise<void>
  onMerge: (source: string, target: string) => Promise<void>
  onToLibrary: (speakerId: string, name: string, window?: LibraryWindow) => Promise<VoiceInfo>
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
  onApplyNames,
  onOpenVoices,
}: Props) {
  const [busy, setBusy] = useState(false)
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
      const response = await onApplyNames()
      setStatus({
        kind: response.error ? 'error' : 'info',
        text: describeApply(response, total),
      })
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

  return (
    <div className="rounded-md border border-slate-200 p-3 dark:border-slate-800">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <h3 className="font-medium">Говорящие</h3>
        <span className="text-xs text-slate-400 dark:text-slate-500">
          {speakers.length} шт.
        </span>
        <div className="ml-auto flex flex-wrap gap-2">
          <button
            type="button"
            onClick={applyNames}
            disabled={busy || speakers.length === 0}
            className="rounded-md bg-slate-800 px-3 py-1.5 text-xs text-white hover:bg-slate-700 disabled:opacity-40 dark:bg-slate-200 dark:text-slate-900 dark:hover:bg-white"
          >
            Применить имена
          </button>
          <button
            type="button"
            onClick={onOpenVoices}
            className="rounded-md border border-slate-300 px-3 py-1.5 text-xs hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800"
          >
            Библиотека голосов
          </button>
        </div>
      </div>

      {status && (
        <p
          role="status"
          className={`mb-3 rounded-md px-3 py-1.5 text-xs ${
            status.kind === 'error'
              ? 'bg-red-50 text-red-700 dark:bg-red-950/50 dark:text-red-300'
              : 'bg-slate-50 text-slate-600 dark:bg-slate-800 dark:text-slate-300'
          }`}
        >
          {status.text}
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
              className="flex flex-col gap-2 rounded-md border border-slate-100 bg-slate-50/60 px-3 py-2 dark:border-slate-800 dark:bg-slate-800/40 sm:grid sm:grid-cols-[minmax(0,1fr)_auto] sm:items-center sm:gap-x-3 sm:gap-y-2"
            >
              <div className="flex min-w-0 items-center gap-2 sm:col-start-1 sm:row-start-1">
                <span
                  className="shrink-0 rounded bg-slate-200 px-1.5 py-0.5 font-mono text-[10px] text-slate-500 dark:bg-slate-700 dark:text-slate-300"
                  title={speaker.id}
                >
                  {speaker.id}
                </span>

                {editing?.kind === 'rename' ? (
                  <div className="flex flex-wrap items-center gap-1">
                    <input
                      autoFocus
                      value={editing.value}
                      onChange={(event) => setEdit({ ...editing, value: event.target.value })}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter') submitEdit(editing)
                        if (event.key === 'Escape') setEdit(null)
                      }}
                      className="w-48 min-w-0 max-w-full rounded-md border border-slate-300 px-2 py-1 text-sm focus:border-blue-400 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
                    />
                    <button
                      type="button"
                      onClick={() => submitEdit(editing)}
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
                  <>
                    <span
                      className="min-w-0 break-words text-sm font-medium leading-snug"
                      title={speaker.display_name}
                    >
                      {speaker.display_name}
                    </span>
                    <button
                      type="button"
                      title="Переименовать"
                      onClick={() =>
                        setEdit({ sid: speaker.id, kind: 'rename', value: speaker.display_name })
                      }
                      className="shrink-0 text-xs text-slate-400 hover:text-slate-700 dark:text-slate-500 dark:hover:text-slate-200"
                    >
                      ✎
                    </button>
                  </>
                )}
              </div>

              {/* Служебная строка: реплики · образец · длительность — фиксированные слоты,
                  чтобы значения не «прыгали» при разной длине имени. */}
              <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-400 dark:text-slate-500 sm:col-span-2 sm:row-start-2">
                <span className="tabular-nums" title="Реплики, где говорящий основной">
                  {counts[speaker.id] ?? 0} реплик
                </span>
                {(extraCounts[speaker.id] ?? 0) > 0 && (
                  <span
                    className="tabular-nums"
                    title="Реплики, где говорящий — участник наложения"
                  >
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

              <div className="flex flex-wrap items-center gap-2 sm:col-start-2 sm:row-start-1 sm:justify-self-end sm:flex-nowrap">
                {editing?.kind === 'library' ? (
                  <div className="flex flex-wrap items-center gap-1">
                    <input
                      autoFocus
                      value={editing.value}
                      onChange={(event) => setEdit({ ...editing, value: event.target.value })}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter') submitEdit(editing)
                        if (event.key === 'Escape') setEdit(null)
                      }}
                      className="w-40 min-w-0 max-w-full rounded-md border border-slate-300 px-2 py-1 text-xs focus:border-blue-400 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
                    />
                    <button
                      type="button"
                      onClick={() => submitEdit(editing)}
                      className="rounded-md bg-slate-800 px-2 py-1 text-xs text-white hover:bg-slate-700 dark:bg-slate-200 dark:text-slate-900 dark:hover:bg-white"
                    >
                      Сохранить
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
                  <button
                    type="button"
                    disabled={!speaker.has_sample || busy}
                    title={speaker.has_sample ? undefined : 'Нет образца голоса'}
                    onClick={() =>
                      setEdit({
                        sid: speaker.id,
                        kind: 'library',
                        value: speaker.display_name,
                      })
                    }
                    className="rounded-md border border-slate-300 px-2 py-1 text-xs hover:bg-slate-100 disabled:opacity-40 dark:border-slate-700 dark:hover:bg-slate-800"
                  >
                    В библиотеку
                  </button>
                )}

                {others.length > 0 && (
                  <div className="flex items-center gap-1">
                    <select
                      value={mergeTarget[speaker.id] ?? ''}
                      onChange={(event) =>
                        setMergeTarget((prev) => ({ ...prev, [speaker.id]: event.target.value }))
                      }
                      className="max-w-[12rem] rounded-md border border-slate-300 px-2 py-1 text-xs dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
                    >
                      <option value="">Объединить в…</option>
                      {others.map((item) => (
                        <option key={item.id} value={item.id}>
                          {item.display_name} ({item.id})
                        </option>
                      ))}
                    </select>
                    <button
                      type="button"
                      disabled={!mergeTarget[speaker.id] || busy}
                      onClick={() => merge(speaker.id)}
                      className="rounded-md border border-slate-300 px-2 py-1 text-xs hover:bg-slate-100 disabled:opacity-40 dark:border-slate-700 dark:hover:bg-slate-800"
                    >
                      →
                    </button>
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
                onToLibrary={onToLibrary}
              />
            </li>
          )
        })}
      </ul>
    </div>
  )
}

export default SpeakersPanel
