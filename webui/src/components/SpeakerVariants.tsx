import { useCallback, useEffect, useRef, useState } from 'react'
import { ChevronDown, ChevronRight, Pause, Play } from 'lucide-react'

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
import { Alert, Button, Checkbox, IconButton, Input, ProgressBar, Select, Spinner } from './ui'

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
      <Button variant="secondary" size="sm" onClick={() => setOpen((current) => !current)}>
        {open ? (
          <ChevronDown aria-hidden className="h-4 w-4" />
        ) : (
          <ChevronRight aria-hidden className="h-4 w-4" />
        )}
        Варианты прослушивания
      </Button>

      {open && (
        <div className="mt-2 rounded-md border border-border bg-surface p-2">
          <audio
            ref={audioRef}
            preload="metadata"
            src={`/api/jobs/${jobId}/audio`}
            onTimeUpdate={onTimeUpdate}
            onEnded={stop}
            className="hidden"
          />
          <p className="mb-2 text-xs text-muted">
            Прослушайте клип и назовите голос «по клику» (#52) либо перенесите окно другому
            говорящему (#40/#41).
          </p>
          {status && (
            <Alert tone="success" live className="mb-2">
              {status}
            </Alert>
          )}
          {error && (
            <Alert tone="danger" live className="mb-2">
              {error}
            </Alert>
          )}
          {loading ? (
            <div className="flex items-center gap-2 py-1 text-xs text-muted">
              <Spinner size={14} /> Загрузка…
            </div>
          ) : variants.length === 0 ? (
            <p className="py-1 text-xs text-muted">Вариантов не найдено</p>
          ) : (
            <ul className="space-y-1">
              {variants.map((variant, index) => {
                const active = action?.index === index ? action : null
                return (
                  <li
                    key={`${variant.start}-${variant.end}`}
                    className="flex flex-col gap-1 rounded border border-border px-2 py-1"
                  >
                    <div className="flex flex-wrap items-center gap-2 text-xs">
                      <IconButton
                        variant={playing === index ? 'primary' : 'secondary'}
                        size="sm"
                        aria-label={playing === index ? 'Остановить вариант' : 'Прослушать вариант'}
                        title={playing === index ? 'Остановить' : 'Прослушать вариант'}
                        onClick={() => play(index)}
                      >
                        {playing === index ? (
                          <Pause aria-hidden className="h-3.5 w-3.5" />
                        ) : (
                          <Play aria-hidden className="h-3.5 w-3.5" />
                        )}
                      </IconButton>
                      <span className="tabular-nums text-muted">
                        [{formatTime(variant.start)}–{formatTime(variant.end)}]
                      </span>
                      <span className="tabular-nums text-muted">{variant.duration.toFixed(1)} с</span>
                      <span className="tabular-nums text-muted" title="Энергия окна">
                        {variant.score.toFixed(4)}
                      </span>
                    </div>

                    {active ? (
                      <div className="flex flex-wrap items-center gap-1 text-xs">
                        <Input
                          autoFocus
                          aria-label="Имя"
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
                          className="w-44"
                        />
                        {active.kind === 'new' && (
                          <Checkbox
                            label="разделить"
                            checked={split}
                            title="Частично перекрывающиеся реплики станут общими с новым говорящим"
                            onChange={(event) => setSplit(event.target.checked)}
                          />
                        )}
                        <Button
                          variant="primary"
                          size="sm"
                          loading={saving}
                          onClick={() => void submitAction(active)}
                        >
                          {active.kind === 'rename'
                            ? 'Назвать'
                            : active.kind === 'new'
                              ? 'Создать'
                              : 'Сохранить'}
                        </Button>
                        <Button variant="ghost" size="sm" onClick={() => setAction(null)}>
                          Отмена
                        </Button>
                      </div>
                    ) : (
                      <div className="flex flex-wrap items-center gap-2 text-xs">
                        {others.length > 0 && (
                          <span className="flex items-center gap-1">
                            <Select
                              aria-label="Перенести в говорящего"
                              value={targets[index] ?? ''}
                              onChange={(event) =>
                                setTargets((prev) => ({ ...prev, [index]: event.target.value }))
                              }
                              className="max-w-44"
                            >
                              <option value="">Перенести в…</option>
                              {others.map((speaker) => (
                                <option key={speaker.id} value={speaker.id}>
                                  {speaker.display_name} ({speaker.id})
                                </option>
                              ))}
                            </Select>
                            <Button
                              variant="secondary"
                              size="sm"
                              disabled={!targets[index] || saving}
                              onClick={() => transfer(index)}
                            >
                              Перенести
                            </Button>
                          </span>
                        )}
                        <Button
                          variant="secondary"
                          size="sm"
                          title="Создать нового говорящего и назначить ему это окно (#41)"
                          onClick={() => openAction(index, 'new')}
                        >
                          ＋ Говорящий
                        </Button>
                        <Button
                          variant="secondary"
                          size="sm"
                          title="Назвать голос по прослушанному клипу (#52)"
                          onClick={() => openAction(index, 'rename')}
                        >
                          Назвать
                        </Button>
                        <Button
                          variant="secondary"
                          size="sm"
                          onClick={() => openAction(index, 'library')}
                        >
                          В библиотеку
                        </Button>
                      </div>
                    )}

                    {playing === index && (
                      <ProgressBar
                        size="sm"
                        value={progress * 100}
                        label="Прогресс воспроизведения варианта"
                      />
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
