import { useCallback, useEffect, useRef, useState } from 'react'

import {
  api,
  errorMessage,
  formatTime,
  type LibraryWindow,
  type SampleVariant,
  type SpeakerVariants as SpeakerVariantsResponse,
  type VoiceInfo,
} from '../api'

type Props = {
  jobId: string
  speakerId: string
  speakerName: string
  onToLibrary: (speakerId: string, name: string, window?: LibraryWindow) => Promise<VoiceInfo>
}

function variantsUrl(jobId: string, speakerId: string): string {
  return `/api/jobs/${jobId}/speakers/${encodeURIComponent(speakerId)}/variants?count=5`
}

/**
 * Раскрываемый список вариантов прослушивания говорящего (#25).
 *
 * Варианты — неперекрывающиеся окна исходного аудио. Звук берётся существующим
 * эндпоинтом задачи с поддержкой Range (один `<audio>` на компонент), а ▶
 * выставляет `currentTime = start` и останавливает воспроизведение на `end`.
 */
function SpeakerVariants({ jobId, speakerId, speakerName, onToLibrary }: Props) {
  const [open, setOpen] = useState(false)
  const [variants, setVariants] = useState<SampleVariant[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [playing, setPlaying] = useState<number | null>(null)
  const [progress, setProgress] = useState(0)
  const [libraryIndex, setLibraryIndex] = useState<number | null>(null)
  const [libraryName, setLibraryName] = useState(speakerName)
  const [saving, setSaving] = useState(false)
  const [status, setStatus] = useState<string | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)

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
    else audio.addEventListener('loadedmetadata', begin, { once: true })
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

  const submitLibrary = async (index: number) => {
    const name = libraryName.trim() || speakerName
    const variant = variants[index]
    setSaving(true)
    setError(null)
    try {
      const saved = await onToLibrary(speakerId, name, { start: variant.start, end: variant.end })
      setStatus(`Сохранено в библиотеку: ${saved.filename}`)
      setLibraryIndex(null)
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="mt-1 sm:col-span-2">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="rounded-md border border-slate-300 px-2 py-1 text-xs hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800"
      >
        {open ? '▾' : '▸'} Варианты прослушивания
      </button>

      {open && (
        <div className="mt-2 rounded-md border border-slate-200 bg-white p-2 dark:border-slate-700 dark:bg-slate-900">
          <audio
            ref={audioRef}
            preload="none"
            src={`/api/jobs/${jobId}/audio`}
            onTimeUpdate={onTimeUpdate}
            onEnded={stop}
            className="hidden"
          />
          {status && (
            <p role="status" className="mb-2 text-xs text-slate-500 dark:text-slate-400">
              {status}
            </p>
          )}
          {error && (
            <p className="mb-2 text-xs text-red-600 dark:text-red-300">{error}</p>
          )}
          {loading ? (
            <p className="py-1 text-xs text-slate-400 dark:text-slate-500">Загрузка…</p>
          ) : variants.length === 0 ? (
            <p className="py-1 text-xs text-slate-400 dark:text-slate-500">
              Вариантов не найдено
            </p>
          ) : (
            <ul className="space-y-1">
              {variants.map((variant, index) => (
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
                    <span className="tabular-nums text-slate-300 dark:text-slate-600" title="Энергия окна">
                      {variant.score.toFixed(4)}
                    </span>
                    {libraryIndex === index ? (
                      <span className="ml-auto flex items-center gap-1">
                        <input
                          autoFocus
                          value={libraryName}
                          onChange={(event) => setLibraryName(event.target.value)}
                          onKeyDown={(event) => {
                            if (event.key === 'Enter') void submitLibrary(index)
                            if (event.key === 'Escape') setLibraryIndex(null)
                          }}
                          className="w-40 rounded-md border border-slate-300 px-2 py-0.5 text-xs dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
                        />
                        <button
                          type="button"
                          disabled={saving}
                          onClick={() => void submitLibrary(index)}
                          className="rounded-md bg-slate-800 px-2 py-0.5 text-white hover:bg-slate-700 disabled:opacity-40 dark:bg-slate-200 dark:text-slate-900"
                        >
                          Сохранить
                        </button>
                      </span>
                    ) : (
                      <button
                        type="button"
                        onClick={() => {
                          setLibraryName(speakerName)
                          setLibraryIndex(index)
                        }}
                        className="ml-auto rounded-md border border-slate-300 px-2 py-0.5 hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800"
                      >
                        В библиотеку
                      </button>
                    )}
                  </div>
                  {playing === index && (
                    <div className="h-1 w-full overflow-hidden rounded bg-slate-100 dark:bg-slate-800">
                      <div
                        className="h-full bg-blue-500 transition-[width] duration-100"
                        style={{ width: `${Math.round(progress * 100)}%` }}
                      />
                    </div>
                  )}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}

export default SpeakerVariants
