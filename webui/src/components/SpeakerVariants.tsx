import { useCallback, useEffect, useRef, useState } from 'react'

import {
  api,
  errorMessage,
  formatTime,
  type LibraryWindow,
  type ReassignRequest,
  type ReassignResponse,
  type SampleVariant,
  type SpeakerChange,
  type SpeakerInfo,
  type SpeakerVariants as SpeakerVariantsResponse,
  type VoiceInfo,
} from '../api'

type Props = {
  jobId: string
  speakerId: string
  speakerName: string
  speakers: SpeakerInfo[]
  onToLibrary: (speakerId: string, name: string, window?: LibraryWindow) => Promise<VoiceInfo>
  onReassign: (speakerId: string, body: ReassignRequest) => Promise<ReassignResponse>
  onRename: (speakerId: string, name: string) => Promise<void>
}

type ActionKind = 'library' | 'rename' | 'new'
type ActionState = { index: number; kind: ActionKind }

function variantsUrl(jobId: string, speakerId: string): string {
  return `/api/jobs/${jobId}/speakers/${encodeURIComponent(speakerId)}/variants?count=5`
}

/** Человекочитаемый итог переноса окна: созданный говорящий и изменения реплик. */
function describeChanges(
  changes: SpeakerChange[],
  created: ReassignResponse['created_speaker'],
): string {
  const parts: string[] = []
  if (created) parts.push(`создан ${created.display_name} (${created.id})`)
  parts.push(`изменено реплик: ${changes.length}`)
  const details = changes.slice(0, 3).map((change) => {
    if (change.after_speaker_id === change.before_speaker_id) {
      return `[${change.index}] + ${change.after_extra_ids.join(', ') || '—'}`
    }
    const from = change.before_speaker_id ?? '—'
    const to = change.after_speaker_id ?? '—'
    return `[${change.index}] ${from} → ${to}`
  })
  if (details.length > 0) parts.push(details.join('; '))
  if (changes.length > 3) parts.push('…')
  return parts.join(' · ')
}

/**
 * Раскрываемый список вариантов прослушивания говорящего (#25).
 *
 * Варианты — неперекрывающиеся окна исходного аудио. Звук берётся существующим
 * эндпоинтом задачи с поддержкой Range (один `<audio>` на компонент), а ▶
 * выставляет `currentTime = start` и останавливает воспроизведение на `end`.
 *
 * Поверх вариантов — правки говорящих: перенос окна другому говорящему (#40),
 * создание нового и назначение ему окна (#41), именование голоса «по клику»
 * (#52) и сохранение клипа в библиотеку (существующий `to-library`).
 */
function SpeakerVariants({
  jobId,
  speakerId,
  speakerName,
  speakers,
  onToLibrary,
  onReassign,
  onRename,
}: Props) {
  const [open, setOpen] = useState(false)
  const [variants, setVariants] = useState<SampleVariant[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [playing, setPlaying] = useState<number | null>(null)
  const [progress, setProgress] = useState(0)
  const [action, setAction] = useState<ActionState | null>(null)
  const [value, setValue] = useState('')
  const [split, setSplit] = useState(false)
  const [targets, setTargets] = useState<Record<number, string>>({})
  const [saving, setSaving] = useState(false)
  const [status, setStatus] = useState<string | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)

  const others = speakers.filter((speaker) => speaker.id !== speakerId)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await api<SpeakerVariantsResponse>(variantsUrl(jobId, speakerId))
      setVariants(response.variants)
    } catch (cause) {
      setError(errorMessage(cause))
      setVariants([])
    } finally {
      setLoading(false)
    }
  }, [jobId, speakerId])

  useEffect(() => {
    if (open) void load()
  }, [open, load])

  const stop = useCallback(() => {
    const audio = audioRef.current
    if (audio) audio.pause()
    setPlaying(null)
    setProgress(0)
  }, [])

  const play = (index: number) => {
    const audio = audioRef.current
    if (!audio) return
    if (playing === index) {
      stop()
      return
    }
    const variant = variants[index]
    const begin = () => {
      audio.currentTime = variant.start
      setProgress(0)
      setPlaying(index)
      void audio.play().catch((cause: unknown) => {
        setError(errorMessage(cause))
        setPlaying(null)
      })
    }
    if (audio.readyState >= 1) begin()
    else {
      // metadata ещё не загружены — форсируем загрузку и ждём loadedmetadata.
      audio.load()
      audio.addEventListener('loadedmetadata', begin, { once: true })
    }
  }

  const onTimeUpdate = () => {
    const audio = audioRef.current
    if (audio == null || playing == null) return
    const variant = variants[playing]
    if (variant == null) return
    if (audio.currentTime >= variant.end) {
      audio.pause()
      setPlaying(null)
      setProgress(1)
      return
    }
    const span = variant.end - variant.start
    setProgress(span > 0 ? Math.min(1, Math.max(0, (audio.currentTime - variant.start) / span)) : 0)
  }

  const openAction = (index: number, kind: ActionKind) => {
    setError(null)
    setStatus(null)
    setSplit(false)
    setValue(speakerName)
    setAction({ index, kind })
  }

  const runReassign = async (
    index: number,
    body: Omit<ReassignRequest, 'start' | 'end'>,
  ) => {
    const variant = variants[index]
    setSaving(true)
    setError(null)
    setStatus(null)
    try {
      const response = await onReassign(speakerId, {
        ...body,
        start: variant.start,
        end: variant.end,
      })
      setStatus(describeChanges(response.changes, response.created_speaker))
      setAction(null)
      setTargets((prev) => ({ ...prev, [index]: '' }))
      // Окно ушло другому говорящему — варианты этого говорящего изменились.
      await load()
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setSaving(false)
    }
  }

  const submitAction = async (state: ActionState) => {
    const index = state.index
    const name = value.trim()
    if (!name) {
      setError('Введите имя')
      return
    }
    if (state.kind === 'library') {
      const variant = variants[index]
      setSaving(true)
      setError(null)
      try {
        const saved = await onToLibrary(speakerId, name, {
          start: variant.start,
          end: variant.end,
        })
        const warnings = saved.quality?.warnings ?? []
        const suffix = warnings.length > 0 ? ` — ⚠ ${warnings.join('; ')}` : ''
        setStatus(`Сохранено в библиотеку: ${saved.filename}${suffix}`)
        setAction(null)
      } catch (cause) {
        setError(errorMessage(cause))
      } finally {
        setSaving(false)
      }
      return
    }
    if (state.kind === 'rename') {
      setSaving(true)
      setError(null)
      try {
        await onRename(speakerId, name)
        setStatus(`Голос назван: ${name}`)
        setAction(null)
      } catch (cause) {
        setError(errorMessage(cause))
      } finally {
        setSaving(false)
      }
      return
    }
    await runReassign(index, { new_name: name, split })
  }

  const transfer = (index: number) => {
    const target = targets[index]
    if (!target) return
    void runReassign(index, { target_speaker_id: target })
  }

  return (
    <div className="mt-1 sm:col-span-2">
      <button
        type="button"
        onClick={() => setOpen((current) => !current)}
        className="rounded-md border border-slate-300 px-2 py-1 text-xs hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800"
      >
        {open ? '▾' : '▸'} Варианты прослушивания
      </button>

      {open && (
        <div className="mt-2 rounded-md border border-slate-200 bg-white p-2 dark:border-slate-700 dark:bg-slate-900">
          <audio
            ref={audioRef}
            preload="metadata"
            src={`/api/jobs/${jobId}/audio`}
            onTimeUpdate={onTimeUpdate}
            onEnded={stop}
            className="hidden"
          />
          <p className="mb-2 text-xs text-slate-500 dark:text-slate-400">
            Прослушайте клип и назовите голос «по клику» (#52) либо перенесите окно
            другому говорящему (#40/#41).
          </p>
          {status && (
            <p role="status" className="mb-2 text-xs text-slate-600 dark:text-slate-300">
              {status}
            </p>
          )}
          {error && <p className="mb-2 text-xs text-red-600 dark:text-red-300">{error}</p>}
          {loading ? (
            <p className="py-1 text-xs text-slate-400 dark:text-slate-500">Загрузка…</p>
          ) : variants.length === 0 ? (
            <p className="py-1 text-xs text-slate-400 dark:text-slate-500">
              Вариантов не найдено
            </p>
          ) : (
            <ul className="space-y-1">
              {variants.map((variant, index) => {
                const active = action?.index === index ? action : null
                return (
                  <li
                    key={`${variant.start}-${variant.end}`}
                    className="flex flex-col gap-1 rounded border border-slate-100 px-2 py-1 dark:border-slate-800"
                  >
                    <div className="flex flex-wrap items-center gap-2 text-xs">
                      <button
                        type="button"
                        onClick={() => play(index)}
                        className="w-7 shrink-0 rounded-md border border-slate-300 py-0.5 hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800"
                        title={playing === index ? 'Остановить' : 'Прослушать вариант'}
                      >
                        {playing === index ? '⏸' : '▶'}
                      </button>
                      <span className="tabular-nums text-slate-500 dark:text-slate-400">
                        [{formatTime(variant.start)}–{formatTime(variant.end)}]
                      </span>
                      <span className="tabular-nums text-slate-400 dark:text-slate-500">
                        {variant.duration.toFixed(1)} с
                      </span>
                      <span
                        className="tabular-nums text-slate-300 dark:text-slate-600"
                        title="Энергия окна"
                      >
                        {variant.score.toFixed(4)}
                      </span>
                    </div>

                    {active ? (
                      <div className="flex flex-wrap items-center gap-1 text-xs">
                        <input
                          autoFocus
                          value={value}
                          onChange={(event) => setValue(event.target.value)}
                          onKeyDown={(event) => {
                            if (event.key === 'Enter') void submitAction(active)
                            if (event.key === 'Escape') setAction(null)
                          }}
                          placeholder={
                            active.kind === 'rename'
                              ? 'Новое имя говорящего'
                              : active.kind === 'new'
                                ? 'Имя нового говорящего'
                                : 'Имя образца'
                          }
                          className="w-44 rounded-md border border-slate-300 px-2 py-0.5 text-xs dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
                        />
                        {active.kind === 'new' && (
                          <label
                            className="flex items-center gap-1 text-slate-500 dark:text-slate-400"
                            title="Частично перекрывающиеся реплики станут общими с новым говорящим"
                          >
                            <input
                              type="checkbox"
                              checked={split}
                              onChange={(event) => setSplit(event.target.checked)}
                            />
                            разделить
                          </label>
                        )}
                        <button
                          type="button"
                          disabled={saving}
                          onClick={() => void submitAction(active)}
                          className="rounded-md bg-slate-800 px-2 py-0.5 text-white hover:bg-slate-700 disabled:opacity-40 dark:bg-slate-200 dark:text-slate-900"
                        >
                          {active.kind === 'rename'
                            ? 'Назвать'
                            : active.kind === 'new'
                              ? 'Создать'
                              : 'Сохранить'}
                        </button>
                        <button
                          type="button"
                          onClick={() => setAction(null)}
                          className="rounded-md border border-slate-300 px-2 py-0.5 hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800"
                        >
                          Отмена
                        </button>
                      </div>
                    ) : (
                      <div className="flex flex-wrap items-center gap-2 text-xs">
                        {others.length > 0 && (
                          <span className="flex items-center gap-1">
                            <select
                              value={targets[index] ?? ''}
                              onChange={(event) =>
                                setTargets((prev) => ({ ...prev, [index]: event.target.value }))
                              }
                              className="max-w-[11rem] rounded-md border border-slate-300 px-2 py-0.5 text-xs dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
                            >
                              <option value="">Перенести в…</option>
                              {others.map((speaker) => (
                                <option key={speaker.id} value={speaker.id}>
                                  {speaker.display_name} ({speaker.id})
                                </option>
                              ))}
                            </select>
                            <button
                              type="button"
                              disabled={!targets[index] || saving}
                              onClick={() => transfer(index)}
                              className="rounded-md border border-slate-300 px-2 py-0.5 hover:bg-slate-100 disabled:opacity-40 dark:border-slate-700 dark:hover:bg-slate-800"
                            >
                              Перенести
                            </button>
                          </span>
                        )}
                        <button
                          type="button"
                          onClick={() => openAction(index, 'new')}
                          className="rounded-md border border-slate-300 px-2 py-0.5 hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800"
                          title="Создать нового говорящего и назначить ему это окно (#41)"
                        >
                          ＋ Говорящий
                        </button>
                        <button
                          type="button"
                          onClick={() => openAction(index, 'rename')}
                          className="rounded-md border border-slate-300 px-2 py-0.5 hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800"
                          title="Назвать голос по прослушанному клипу (#52)"
                        >
                          Назвать
                        </button>
                        <button
                          type="button"
                          onClick={() => openAction(index, 'library')}
                          className="rounded-md border border-slate-300 px-2 py-0.5 hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800"
                        >
                          В библиотеку
                        </button>
                      </div>
                    )}

                    {playing === index && (
                      <div className="h-1 w-full overflow-hidden rounded bg-slate-100 dark:bg-slate-800">
                        <div
                          className="h-full bg-blue-500 transition-[width] duration-100"
                          style={{ width: `${Math.round(progress * 100)}%` }}
                        />
                      </div>
                    )}
                  </li>
                )
              })}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}

export default SpeakerVariants
