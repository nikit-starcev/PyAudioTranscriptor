import { Play, RefreshCw, RotateCcw, Square, Trash2 } from 'lucide-react'

import { formatClock } from '../../api'
import { navigateTo } from '../../app/routes'
import {
  formatSpeakerSetting,
  isLiveJob,
  STATUS_LABELS,
  statusTone,
} from '../../app/jobUtils'
import { useApp } from '../../app/useApp'
import {
  Badge,
  Button,
  Card,
  CardContent,
  CardHeader,
  Checkbox,
  EmptyState,
  IconButton,
  cn,
} from '../../components/ui'

function JobList() {
  const app = useApp()

  return (
    <Card>
      <CardHeader
        title="Задачи"
        description="Все запуски обработки"
        actions={
          <>
            <Checkbox
              label="Показать удалённые"
              checked={app.showDeleted}
              onChange={(event) => app.toggleDeleted(event.target.checked)}
            />
            <IconButton
              aria-label="Обновить список задач"
              size="sm"
              onClick={() => void app.refreshJobs()}
            >
              <RefreshCw aria-hidden className="h-4 w-4" />
            </IconButton>
          </>
        }
      />
      <CardContent>
        {app.jobs.length === 0 ? (
          <EmptyState
            title="Задач пока нет"
            description="Загрузите файл и поставьте его в очередь"
          />
        ) : (
          <ul className="divide-y divide-border">
            {app.jobs.map((job) => {
              const live = isLiveJob(job)
              const canRun =
                job.status === 'queued' ||
                job.status === 'done' ||
                job.status === 'error' ||
                job.status === 'cancelled' ||
                (job.status === 'running' && job.active === false)
              return (
                <li
                  key={job.id}
                  className={cn(
                    'flex flex-wrap items-center gap-x-3 gap-y-2 py-3',
                    job.deleted && 'opacity-70',
                  )}
                >
                  <a
                    href={`#/jobs/${encodeURIComponent(job.id)}`}
                    className="min-w-0 flex-1 rounded-md focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"
                  >
                    <p
                      className={cn('truncate text-sm', job.deleted && 'line-through')}
                      title={job.name}
                    >
                      {job.name}
                    </p>
                    <p className="text-xs tabular-nums text-muted">
                      {job.stage ? `${job.stage} · ` : ''}
                      {job.fraction != null ? `${Math.round(job.fraction * 100)}%` : '—'}
                      {job.duration != null ? ` · запись ${formatClock(job.duration)}` : ''}
                      {' · говорящих: '}
                      {formatSpeakerSetting(job)}
                    </p>
                  </a>

                  {job.deleted ? (
                    <Badge tone="danger">Удалено</Badge>
                  ) : (
                    <Badge tone={statusTone(job.status)}>
                      {STATUS_LABELS[job.status] ?? job.status}
                    </Badge>
                  )}

                  {job.deleted ? (
                    <div className="flex shrink-0 items-center gap-1.5">
                      <Button
                        variant="secondary"
                        size="sm"
                        icon={<RotateCcw aria-hidden className="h-3.5 w-3.5" />}
                        onClick={() => void app.restoreJob(job.id)}
                      >
                        Восстановить
                      </Button>
                      <Button
                        variant="danger"
                        size="sm"
                        onClick={() => void app.purgeJob(job.id, job.name)}
                      >
                        Удалить навсегда
                      </Button>
                    </div>
                  ) : (
                    <div className="flex shrink-0 items-center gap-1.5">
                      {live && (
                        <Button
                          variant="secondary"
                          size="sm"
                          icon={<Square aria-hidden className="h-3.5 w-3.5" />}
                          onClick={() => void app.stopJob(job.id)}
                        >
                          Остановить
                        </Button>
                      )}
                      {canRun && (
                        <Button
                          variant="primary"
                          size="sm"
                          icon={<Play aria-hidden className="h-3.5 w-3.5" />}
                          disabled={app.readinessBlocked}
                          title={app.blockedHint}
                          onClick={() => {
                            void app.runJob(job.id).then((ok) => {
                              if (ok) navigateTo(`/jobs/${encodeURIComponent(job.id)}`)
                            })
                          }}
                        >
                          {job.status === 'cancelled' ? 'Запустить снова' : 'Запустить'}
                        </Button>
                      )}
                      {!live && (
                        <IconButton
                          aria-label={`Удалить задачу ${job.name}`}
                          size="sm"
                          onClick={() => void app.deleteJob(job.id)}
                        >
                          <Trash2 aria-hidden className="h-4 w-4" />
                        </IconButton>
                      )}
                    </div>
                  )}
                </li>
              )
            })}
          </ul>
        )}
      </CardContent>
    </Card>
  )
}

export default JobList
