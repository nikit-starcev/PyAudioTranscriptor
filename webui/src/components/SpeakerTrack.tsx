import { useState } from 'react'
import { ChevronDown, ChevronRight, Download, Headphones } from 'lucide-react'

import { speakerTrackUrl } from '../api'
import { Button, Checkbox, Spinner } from './ui'

type Props = {
  jobId: string
  speakerId: string
  speakerName: string
}

const LINK_BUTTON =
  'inline-flex h-8 items-center justify-center gap-2 rounded-md border border-border-strong ' +
  'bg-surface px-3 text-xs font-medium text-text transition-colors hover:bg-surface-2 ' +
  'focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring'

/**
 * Дорожка выбранного говорящего (#108) — «solo»-прослушивание и скачивание.
 *
 * Звук собирается на сервере (`GET .../speakers/{sid}/track`): интервалы речи
 * остальных говорящих заглушаются по таймкодам реплик, поэтому плеер отдаёт
 * запись, где слышно только выбранного участника. Для прослушивания берётся
 * WAV, для скачивания — компактный m4a (AAC).
 */
function SpeakerTrack({ jobId, speakerId, speakerName }: Props) {
  const [open, setOpen] = useState(false)
  const [muteOverlap, setMuteOverlap] = useState(false)
  const [status, setStatus] = useState<'idle' | 'loading' | 'ready' | 'error'>('idle')
  const overlap = muteOverlap ? 'mute' : 'keep'
  const playUrl = speakerTrackUrl(jobId, speakerId, { fmt: 'wav', overlap })
  const downloadUrl = speakerTrackUrl(jobId, speakerId, { fmt: 'm4a', overlap })

  return (
    <div className="mt-1 sm:col-span-2">
      <Button variant="secondary" size="sm" onClick={() => setOpen((current) => !current)}>
        {open ? (
          <ChevronDown aria-hidden className="h-4 w-4" />
        ) : (
          <ChevronRight aria-hidden className="h-4 w-4" />
        )}
        <Headphones aria-hidden className="h-4 w-4" />
        Дорожка говорящего
      </Button>

      {open && (
        <div className="mt-2 space-y-2 rounded-md border border-border bg-surface p-2">
          <p className="text-xs text-muted">
            Слышно только «{speakerName}»: речь остальных говорящих заглушается по таймкодам
            реплик. Первый запуск может занять время — дорожка собирается и кэшируется.
          </p>
          {status === 'loading' && (
            <div className="flex items-center gap-2 text-xs text-muted">
              <Spinner size={14} /> Готовим дорожку…
            </div>
          )}
          {status === 'error' && (
            <p className="text-xs text-danger">Не удалось загрузить дорожку</p>
          )}
          <audio
            controls
            preload="metadata"
            className="h-8 w-full"
            src={playUrl}
            onLoadStart={() => setStatus('loading')}
            onLoadedMetadata={() => setStatus('ready')}
            onCanPlay={() => setStatus('ready')}
            onError={() => setStatus('error')}
          />
          <div className="flex flex-wrap items-center gap-3">
            <Checkbox
              label="Глушить наложения"
              checked={muteOverlap}
              onChange={(event) => setMuteOverlap(event.target.checked)}
              title="Реплики, где говорящий говорит одновременно с другими, тоже заглушаются — остаются только чистые реплики"
            />
            <a className={LINK_BUTTON} href={downloadUrl} download>
              <Download aria-hidden className="h-4 w-4" />
              Скачать m4a
            </a>
          </div>
        </div>
      )}
    </div>
  )
}

export default SpeakerTrack
