import { useCallback, useEffect, useRef, useState } from 'react'

import { formatTime, speakerName, type Entry, type SpeakerInfo } from '../api'

type Props = {
  jobId: string
  entries: Entry[]
  speakers: SpeakerInfo[]
}

type Fragment = { key: string; start: number; end: number }

type SpeakerPiece = { id: string | null; name: string; extra: boolean }

function entryKey(entry: Entry, index: number): string {
  return `${entry.start}-${entry.end}-${index}`
}

//: Разбивает подпись реплики на основного говорящего и участников наложения.
//: Для старых результатов без поля `extra_speaker_ids` доп. говорящих нет.
function speakerPieces(speakers: SpeakerInfo[], entry: Entry): SpeakerPiece[] {
  const pieces: SpeakerPiece[] = [
    { id: entry.speaker_id, name: speakerName(speakers, entry.speaker_id), extra: false },
  ]
  for (const extraId of entry.extra_speaker_ids ?? []) {
    pieces.push({ id: extraId, name: speakerName(speakers, extraId), extra: true })
  }
  return pieces
}

//: Иконка воспроизведения/паузы (SVG вместо эмодзи — предсказуемый вид и цвет).
function PlayerIcon({ paused }: { paused: boolean }) {
  return (
    <svg viewBox="0 0 16 16" width="10" height="10" fill="currentColor" aria-hidden="true">
      {paused ? (
        <>
          <rect x="4" y="3" width="3" height="10" rx="0.5" />
          <rect x="9" y="3" width="3" height="10" rx="0.5" />
        </>
      ) : (
        <path d="M4.5 2.8v10.4L13 8z" />
      )}
    </svg>
  )
}

//: Минимальная длина фрагмента, чтобы не делить на ноль в прогрессе.
const MIN_FRAGMENT = 0.05
//: Точность остановки: останавливаемся чуть раньше `end`, чтобы не зацепить
//: следующий звук из-за округления `currentTime`.
const STOP_EPSILON = 0.005
//: Как часто обновлять прогресс (мс), чтобы не ререндерить таблицу каждый кадр.
const PAINT_INTERVAL_MS = 100

function TranscriptTable({ jobId, entries, speakers }: Props) {
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const [playingKey, setPlayingKey] = useState<string | null>(null)
  const [position, setPosition] = useState(0)
  const [autoAdvance, setAutoAdvance] = useState(false)

  const fragmentRef = useRef<Fragment | null>(null)
  const rafRef = useRef<number | null>(null)
  const lastPaintRef = useRef(0)
  const playRef = useRef<(fragment: Fragment) => void>(() => {})

  // Актуальные значения для цикла requestAnimationFrame (без пересоздания).
  const entriesRef = useRef(entries)
  const autoAdvanceRef = useRef(autoAdvance)

  useEffect(() => {
    entriesRef.current = entries
  }, [entries])

  useEffect(() => {
    autoAdvanceRef.current = autoAdvance
  }, [autoAdvance])

  const cancelLoop = useCallback(() => {
    if (rafRef.current != null) {
      cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }
  }, [])

  const stop = useCallback(() => {
    cancelLoop()
    fragmentRef.current = null
    audioRef.current?.pause()
    setPlayingKey(null)
    setPosition(0)
  }, [cancelLoop])

  const playFragment = useCallback(
    (fragment: Fragment) => {
      const audio = audioRef.current
      if (!audio) return
      cancelLoop()
      fragmentRef.current = fragment
      setPlayingKey(fragment.key)
      setPosition(0)
      lastPaintRef.current = 0

      const startLoop = () => {
        cancelLoop()
        const tick = (now: number) => {
          const el = audioRef.current
          const current = fragmentRef.current
          if (!el || !current) return
          if (el.currentTime >= current.end - STOP_EPSILON) {
            el.pause()
            fragmentRef.current = null
            setPlayingKey(null)
            setPosition(0)
            cancelLoop()
            if (autoAdvanceRef.current) {
              const list = entriesRef.current
              const index = list.findIndex((item, i) => entryKey(item, i) === current.key)
              const next = index >= 0 ? list[index + 1] : undefined
              if (next) {
                playRef.current({
                  key: entryKey(next, index + 1),
                  start: next.start,
                  end: next.end,
                })
              }
            }
            return
          }
          // Автопауза на конце файла (если end вышел за длительность).
          if (el.ended) {
            stop()
            return
          }
          if (now - lastPaintRef.current >= PAINT_INTERVAL_MS) {
            lastPaintRef.current = now
            setPosition(Math.max(0, el.currentTime - current.start))
          }
          rafRef.current = requestAnimationFrame(tick)
        }
        rafRef.current = requestAnimationFrame(tick)
      }

      const seekAndPlay = () => {
        audio.currentTime = Math.max(0, fragment.start)
        void audio.play().then(startLoop).catch(() => stop())
      }

      if (audio.readyState >= 1 /* HAVE_METADATA */) {
        seekAndPlay()
      } else {
        const onReady = () => {
          audio.removeEventListener('loadedmetadata', onReady)
          seekAndPlay()
        }
        audio.addEventListener('loadedmetadata', onReady)
        audio.load()
      }
    },
    [cancelLoop, stop],
  )
  useEffect(() => {
    playRef.current = playFragment
  }, [playFragment])

  // Останавливаем воспроизведение при смене задачи и размонтировании.
  useEffect(() => stop, [jobId, stop])

  const toggle = (fragment: Fragment) => {
    if (playingKey === fragment.key) stop()
    else playFragment(fragment)
  }

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-3 text-xs text-slate-500 dark:text-slate-400">
        <label className="flex items-center gap-1.5">
          <input
            type="checkbox"
            checked={autoAdvance}
            onChange={(event) => setAutoAdvance(event.target.checked)}
          />
          Автопереход к следующей реплике
        </label>
        {playingKey ? (
          <span className="flex items-center gap-1 font-medium text-blue-600 dark:text-blue-400">
            <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-blue-500" />
            воспроизведение фрагмента
          </span>
        ) : (
          <span>Кнопка воспроизведения проигрывает ровно интервал реплики</span>
        )}
      </div>

      <audio ref={audioRef} preload="metadata" src={`/api/jobs/${jobId}/audio`} className="hidden" />

      <div className="max-h-[28rem] overflow-auto rounded-md border border-slate-200 dark:border-slate-800">
        <table className="w-full border-collapse text-sm">
          <thead className="sticky top-0 z-10 bg-slate-100 text-left text-xs uppercase text-slate-500 dark:bg-slate-800 dark:text-slate-400">
            <tr>
              <th className="w-10 px-2 py-2 font-medium" aria-label="Прослушать" />
              <th className="px-3 py-2 font-medium">Время</th>
              <th className="px-3 py-2 font-medium">Говорящий</th>
              <th className="px-3 py-2 font-medium">Метки</th>
              <th className="px-3 py-2 font-medium">Текст</th>
            </tr>
          </thead>
          <tbody>
            {entries.map((entry, index) => {
              const key = entryKey(entry, index)
              const playing = playingKey === key
              const span = Math.max(MIN_FRAGMENT, entry.end - entry.start)
              const percent = playing
                ? Math.min(100, Math.max(0, (position / span) * 100))
                : 0
              const pieces = speakerPieces(speakers, entry)
              return (
                <tr
                  key={key}
                  className={
                    playing
                      ? 'border-t border-slate-100 bg-blue-50 dark:border-slate-800 dark:bg-blue-950/40'
                      : 'border-t border-slate-100 dark:border-slate-800'
                  }
                >
                  <td className="px-2 py-1.5">
                    <button
                      type="button"
                      onClick={() => toggle({ key, start: entry.start, end: entry.end })}
                      aria-label={playing ? 'Остановить фрагмент' : 'Прослушать фрагмент'}
                      aria-pressed={playing}
                      title={playing ? 'Остановить' : `Прослушать ${formatTime(entry.start)}–${formatTime(entry.end)}`}
                      className={
                        playing
                          ? 'flex h-6 w-6 items-center justify-center rounded-full bg-blue-500 text-xs text-white'
                          : 'flex h-6 w-6 items-center justify-center rounded-full border border-slate-300 text-xs text-slate-600 hover:bg-slate-100 dark:border-slate-600 dark:text-slate-300 dark:hover:bg-slate-800'
                      }
                    >
                      <PlayerIcon paused={playing} />
                    </button>
                  </td>
                  <td className="whitespace-nowrap px-3 py-1.5 font-mono text-xs text-slate-500 dark:text-slate-400">
                    {formatTime(entry.start)}
                  </td>
                  <td className="px-3 py-1.5">
                    <div
                      className="max-w-[9rem] truncate sm:max-w-[16rem]"
                      title={pieces.map((piece) => piece.name).join(' + ')}
                    >
                      {pieces.map((piece, pieceIndex) => (
                        <span key={`${piece.id ?? 'none'}-${pieceIndex}`}>
                          {pieceIndex > 0 && (
                            <span className="mx-1 text-slate-400 dark:text-slate-500">+</span>
                          )}
                          <span
                            className={
                              piece.extra
                                ? 'text-slate-500 dark:text-slate-400'
                                : undefined
                            }
                            title={piece.extra ? 'дополнительный говорящий (наложение)' : undefined}
                          >
                            {piece.name}
                          </span>
                        </span>
                      ))}
                    </div>
                  </td>
                  <td className="whitespace-nowrap px-3 py-1.5 text-base">
                    {entry.low_confidence && <span title="низкая уверенность">⚠</span>}
                    {entry.low_speaker_confidence && (
                      <span title="говорящий под вопросом">?</span>
                    )}
                    {entry.overlap && <span title="наложение речи">⇄</span>}
                  </td>
                  <td className="px-3 py-1.5">
                    <div className="break-words">{entry.text}</div>
                    {playing && (
                      <div className="mt-1 h-1 w-full overflow-hidden rounded-full bg-blue-100 dark:bg-blue-900">
                        <div
                          className="h-full rounded-full bg-blue-500 transition-[width] duration-100 ease-linear"
                          style={{ width: `${percent}%` }}
                        />
                      </div>
                    )}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
        {entries.length === 0 && (
          <p className="py-6 text-center text-sm text-slate-400 dark:text-slate-500">
            Ничего не найдено
          </p>
        )}
      </div>
    </div>
  )
}

export default TranscriptTable
