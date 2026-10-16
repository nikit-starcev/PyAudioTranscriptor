import { useCallback, useEffect, useState } from 'react'
import { Presentation } from 'lucide-react'

import {
  ApiHttpError,
  errorMessage,
  extractSlides,
  fetchSlides,
  formatTime,
  type Slide,
} from '../../api'
import { usePlayback } from '../../app/playback'
import {
  Alert,
  Badge,
  Button,
  Card,
  CardContent,
  CardHeader,
  EmptyState,
  Spinner,
} from '../../components/ui'

function SlidesTab({ jobId }: { jobId: string }) {
  const playback = usePlayback()
  const [slides, setSlides] = useState<Slide[] | null>(null)
  const [loading, setLoading] = useState(true)
  const [extracting, setExtracting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [videoMissing, setVideoMissing] = useState(false)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    setVideoMissing(false)
    try {
      const data = await fetchSlides(jobId)
      setSlides(data?.slides ?? null)
    } catch (cause) {
      setError(errorMessage(cause))
      setSlides(null)
    } finally {
      setLoading(false)
    }
  }, [jobId])

  useEffect(() => {
    void load()
  }, [load])

  const extract = useCallback(async () => {
    setExtracting(true)
    setError(null)
    setVideoMissing(false)
    try {
      const data = await extractSlides(jobId)
      setSlides(data.slides)
    } catch (cause) {
      if (cause instanceof ApiHttpError && cause.status === 404) {
        setVideoMissing(true)
        setSlides(null)
      } else {
        setError(errorMessage(cause))
      }
    } finally {
      setExtracting(false)
    }
  }, [jobId])

  const jumpTo = useCallback(
    (slide: Slide) => {
      if (slide.end != null && slide.end > slide.start) {
        playback.playRange({
          key: `slide-${slide.index}`,
          start: slide.start,
          end: slide.end,
        })
      } else {
        playback.playFrom(slide.start)
      }
    },
    [playback],
  )

  const extractButton = (
    <Button
      variant="primary"
      size="sm"
      icon={<Presentation aria-hidden className="h-4 w-4" />}
      loading={extracting}
      onClick={() => void extract()}
    >
      {slides && slides.length > 0 ? 'Извлечь заново' : 'Извлечь слайды'}
    </Button>
  )

  let body
  if (loading) {
    body = (
      <div className="flex items-center justify-center py-2">
        <Spinner size={16} label="Чтение слайдов" />
      </div>
    )
  } else if (error) {
    body = (
      <Alert tone="danger" live onDismiss={() => setError(null)}>
        {error}
      </Alert>
    )
  } else if (videoMissing) {
    body = (
      <EmptyState
        title="Видеодорожки нет"
        description="Извлечение слайдов доступно только для задач с видео"
        action={extractButton}
      />
    )
  } else if (!slides) {
    body = (
      <EmptyState
        title="Слайды не извлечены"
        description="Распознайте кадры презентации из видеоряда записи"
        action={extractButton}
      />
    )
  } else if (slides.length === 0) {
    body = (
      <EmptyState
        title="Слайды не найдены"
        description="В видеоряде не обнаружено смены слайдов"
        action={extractButton}
      />
    )
  } else {
    body = (
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3">
        {slides.map((slide) => (
          <Card key={slide.index} className="overflow-hidden">
            <a href={slide.image} target="_blank" rel="noreferrer">
              <img
                src={slide.image}
                alt={`Слайд ${slide.index + 1}`}
                loading="lazy"
                className="h-40 w-full bg-surface-2 object-contain"
              />
            </a>
            <CardContent className="space-y-2">
              <div className="flex items-center justify-between gap-2">
                <Badge tone="neutral">Слайд {slide.index + 1}</Badge>
                <button
                  type="button"
                  title={
                    slide.end != null
                      ? `Перейти к ${formatTime(slide.start)}–${formatTime(slide.end)}`
                      : `Перейти к ${formatTime(slide.start)}`
                  }
                  onClick={() => jumpTo(slide)}
                  className="rounded font-mono text-xs tabular-nums text-primary hover:underline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"
                >
                  {formatTime(slide.start)}
                </button>
              </div>
              <pre className="max-h-40 overflow-auto whitespace-pre-wrap break-words rounded-md bg-surface-2 p-2 font-mono text-xs text-text">
                {slide.text || '—'}
              </pre>
              <p className="text-[11px] text-muted">{slide.engine}</p>
            </CardContent>
          </Card>
        ))}
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader
          title="Слайды"
          description={
            slides && slides.length > 0
              ? `Распознано слайдов: ${slides.length}`
              : 'Текст слайдов презентации, распознанный по видеоряду'
          }
          actions={extractButton}
        />
        <CardContent className="space-y-3">{body}</CardContent>
      </Card>
    </div>
  )
}

export default SlidesTab
