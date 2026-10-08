import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
  type RefObject,
} from 'react'

export type PlaybackFragment = { key: string; start: number; end: number }

type TimeListener = (time: number) => void

export type PlaybackController = {
  audioRef: RefObject<HTMLAudioElement | null>
  playing: boolean
  /** Воспроизведение хотя бы раз запускалось для текущей задачи. */
  started: boolean
  ready: boolean
  duration: number | null
  /** Ключ реплики, проигрываемой как ограниченный фрагмент (для кнопки ▶). */
  playingKey: string | null
  /** Актуальная позиция (сек); обновляется каждый кадр без React-рендера. */
  timeRef: RefObject<number>
  playRange: (fragment: PlaybackFragment, onEnd?: () => void) => void
  toggleRange: (fragment: PlaybackFragment, onEnd?: () => void) => void
  playFrom: (time: number) => void
  seek: (time: number) => void
  toggle: () => void
  stop: () => void
  subscribe: (listener: TimeListener) => () => void
}

const PlaybackContext = createContext<PlaybackController | null>(null)

const STOP_EPSILON = 0.005

export function PlaybackProvider({
  jobId,
  children,
}: {
  jobId: string
  children: ReactNode
}) {
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const [playing, setPlaying] = useState(false)
  const [started, setStarted] = useState(false)
  const [ready, setReady] = useState(false)
  const [duration, setDuration] = useState<number | null>(null)
  const [playingKey, setPlayingKey] = useState<string | null>(null)

  const timeRef = useRef(0)
  const listenersRef = useRef<Set<TimeListener>>(new Set())
  const rangeEndRef = useRef<number | null>(null)
  const onEndRef = useRef<(() => void) | null>(null)

  const notify = useCallback(() => {
    const time = timeRef.current
    for (const listener of listenersRef.current) listener(time)
  }, [])

  const stop = useCallback(() => {
    rangeEndRef.current = null
    onEndRef.current = null
    const audio = audioRef.current
    if (audio && !audio.paused) audio.pause()
    timeRef.current = audio?.currentTime ?? 0
    setPlayingKey(null)
    setPlaying(false)
    notify()
  }, [notify])

  const finishRange = useCallback(() => {
    rangeEndRef.current = null
    const audio = audioRef.current
    if (audio && !audio.paused) audio.pause()
    timeRef.current = audio?.currentTime ?? 0
    setPlayingKey(null)
    setPlaying(false)
    notify()
    const callback = onEndRef.current
    onEndRef.current = null
    callback?.()
  }, [notify])

  useEffect(() => {
    if (!playing) return
    let raf = 0
    const step = () => {
      const audio = audioRef.current
      if (!audio) return
      timeRef.current = audio.currentTime
      notify()
      const end = rangeEndRef.current
      if (end != null && audio.currentTime >= end - STOP_EPSILON) {
        finishRange()
        return
      }
      if (audio.ended) {
        finishRange()
        return
      }
      raf = requestAnimationFrame(step)
    }
    raf = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf)
  }, [playing, notify, finishRange])

  const beginPlay = useCallback(
    (start: number) => {
      const audio = audioRef.current
      if (!audio) return
      const run = () => {
        audio.currentTime = Math.max(0, start)
        timeRef.current = audio.currentTime
        notify()
        void audio.play().catch(() => stop())
      }
      if (audio.readyState >= 1 /* HAVE_METADATA */) {
        run()
      } else {
        const onReady = () => {
          audio.removeEventListener('loadedmetadata', onReady)
          run()
        }
        audio.addEventListener('loadedmetadata', onReady)
        audio.load()
      }
    },
    [notify, stop],
  )

  const playRange = useCallback(
    (fragment: PlaybackFragment, onEnd?: () => void) => {
      rangeEndRef.current = fragment.end
      onEndRef.current = onEnd ?? null
      setPlayingKey(fragment.key)
      beginPlay(fragment.start)
    },
    [beginPlay],
  )

  const playFrom = useCallback(
    (time: number) => {
      rangeEndRef.current = null
      onEndRef.current = null
      setPlayingKey(null)
      beginPlay(time)
    },
    [beginPlay],
  )

  const seek = useCallback(
    (time: number) => {
      const audio = audioRef.current
      if (!audio) return
      const limit = Number.isFinite(audio.duration) ? audio.duration : time
      audio.currentTime = Math.max(0, Math.min(time, limit))
      timeRef.current = audio.currentTime
      notify()
    },
    [notify],
  )

  const toggle = useCallback(() => {
    const audio = audioRef.current
    if (!audio) return
    if (audio.paused) void audio.play().catch(() => stop())
    else audio.pause()
  }, [stop])

  const toggleRange = useCallback(
    (fragment: PlaybackFragment, onEnd?: () => void) => {
      if (playingKey === fragment.key && playing) stop()
      else playRange(fragment, onEnd)
    },
    [playRange, playing, playingKey, stop],
  )

  const subscribe = useCallback((listener: TimeListener) => {
    listenersRef.current.add(listener)
    return () => {
      listenersRef.current.delete(listener)
    }
  }, [])

  useEffect(() => {
    stop()
    timeRef.current = 0
    setStarted(false)
    setReady(false)
    setDuration(null)
    notify()
  }, [jobId, stop, notify])

  useEffect(() => {
    const audio = audioRef.current
    return () => {
      if (audio) audio.pause()
    }
  }, [])

  const controller = useMemo<PlaybackController>(
    () => ({
      audioRef,
      playing,
      started,
      ready,
      duration,
      playingKey,
      timeRef,
      playRange,
      toggleRange,
      playFrom,
      seek,
      toggle,
      stop,
      subscribe,
    }),
    [
      playing,
      started,
      ready,
      duration,
      playingKey,
      playRange,
      toggleRange,
      playFrom,
      seek,
      toggle,
      stop,
      subscribe,
    ],
  )

  return (
    <PlaybackContext.Provider value={controller}>
      <audio
        ref={audioRef}
        src={`/api/jobs/${jobId}/audio`}
        preload="metadata"
        className="hidden"
        onPlay={() => {
          setPlaying(true)
          setStarted(true)
        }}
        onPause={() => setPlaying(false)}
        onEnded={finishRange}
        onLoadedMetadata={(event) => {
          const audio = event.currentTarget
          setReady(true)
          setDuration(Number.isFinite(audio.duration) ? audio.duration : null)
          timeRef.current = audio.currentTime
          notify()
        }}
        onDurationChange={(event) => {
          const value = event.currentTarget.duration
          setDuration(Number.isFinite(value) ? value : null)
        }}
        onSeeked={(event) => {
          timeRef.current = event.currentTarget.currentTime
          notify()
        }}
        onError={() => {
          setReady(false)
          setPlaying(false)
        }}
      />
      {children}
    </PlaybackContext.Provider>
  )
}

export function usePlayback(): PlaybackController {
  const context = useContext(PlaybackContext)
  if (!context) throw new Error('usePlayback must be used within <PlaybackProvider>')
  return context
}
