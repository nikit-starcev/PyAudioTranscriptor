import { useEffect, useRef } from 'react'
import { Pause, Play } from 'lucide-react'

import { formatClock } from '../api'
import { usePlayback } from '../app/playback'
import { Card, CardContent, IconButton } from './ui'

function JobPlayer() {
  const playback = usePlayback()
  const sliderRef = useRef<HTMLInputElement | null>(null)
  const timeLabelRef = useRef<HTMLSpanElement | null>(null)
  const seek = playback.seek

  useEffect(() => {
    return playback.subscribe((time) => {
      const slider = sliderRef.current
      if (slider && document.activeElement !== slider) {
        slider.value = String(time)
      }
      if (timeLabelRef.current) {
        timeLabelRef.current.textContent = formatClock(time)
      }
    })
  }, [playback])

  const { duration, playing, ready } = playback

  return (
    <Card>
      <CardContent className="flex flex-wrap items-center gap-3 py-3">
        <IconButton
          aria-label={playing ? 'Пауза' : 'Воспроизвести запись'}
          aria-pressed={playing}
          title={playing ? 'Пауза' : 'Воспроизвести'}
          variant={playing ? 'primary' : 'secondary'}
          disabled={!ready && duration == null}
          onClick={() => playback.toggle()}
        >
          {playing ? (
            <Pause aria-hidden className="h-4 w-4" />
          ) : (
            <Play aria-hidden className="h-4 w-4" />
          )}
        </IconButton>

        <span
          ref={timeLabelRef}
          className="w-12 shrink-0 text-xs tabular-nums text-muted"
          aria-hidden
        >
          0:00
        </span>

        <label htmlFor="job-player-seek" className="sr-only">
          Позиция воспроизведения
        </label>
        <input
          id="job-player-seek"
          ref={sliderRef}
          type="range"
          min={0}
          max={duration ?? 0}
          step={0.1}
          defaultValue={0}
          disabled={duration == null}
          onChange={(event) => seek(Number(event.currentTarget.value))}
          className="h-1 min-w-40 flex-1 cursor-pointer accent-primary disabled:cursor-default disabled:opacity-50"
        />

        <span className="w-12 shrink-0 text-right text-xs tabular-nums text-muted">
          {formatClock(duration)}
        </span>
      </CardContent>
    </Card>
  )
}

export default JobPlayer
