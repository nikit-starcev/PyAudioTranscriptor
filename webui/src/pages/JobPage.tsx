import { useEffect, useState } from 'react'
import { ArrowLeft, Play, RefreshCw, Square, Trash2 } from 'lucide-react'

import { formatClock } from '../api'
import { formatSpeakerSetting, isLiveJob, STATUS_LABELS, statusTone } from '../app/jobUtils'
import { navigateTo } from '../app/routes'
import { useApp } from '../app/useApp'
import {
  Badge,
  Button,
  Card,
  CardContent,
  IconButton,
  ProgressBar,
  TabPanel,
  Tabs,
} from '../components/ui'
import ChatTab from './job/ChatTab'
import ExportTab from './job/ExportTab'
import SpeakersTab from './job/SpeakersTab'
import StagesTab from './job/StagesTab'
import TranscriptTab from './job/TranscriptTab'

const TABS = [
  { id: 'transcript', label: 'Стенограмма' },
  { id: 'speakers', label: 'Спикеры' },
  { id: 'stages', label: 'Стадии' },
  { id: 'chat', label: 'Чат' },
  { id: 'export', label: 'Экспорт' },
]

function JobPage({ jobId }: { jobId: string }) {
  const app = useApp()
  const { openJob } = app
  const [tab, setTab] = useState('transcript')

  useEffect(() => {
    void openJob(jobId)
  }, [jobId, openJob])

  const job = app.activeJob
  const status = app.progress?.status ?? job?.status ?? 'queued'
  const live = app.progressRunning
  const loaded = job != null || app.progress != null
  const canRun =
    loaded &&
    !isLiveJob({ status, active: app.progress?.active ?? job?.active ?? true })
  const canDelete = !live

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-start gap-3">
        <IconButton aria-label="Назад к списку задач" onClick={() => navigateTo('/jobs')}>
          <ArrowLeft aria-hidden className="h-4 w-4" />
        </IconButton>

        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <h1 className="truncate text-lg font-semibold" title={job?.name}>
              {job?.name ?? 'Задача'}
            </h1>
            <Badge tone={statusTone(status)}>{STATUS_LABELS[status] ?? status}</Badge>
          </div>
          <p className="mt-0.5 text-xs tabular-nums text-muted">
            {job?.fraction != null ? `${Math.round(job.fraction * 100)}%` : '—'}
            {job?.duration != null ? ` · запись ${formatClock(job.duration)}` : ''}
            {job ? ` · говорящих: ${formatSpeakerSetting(job)}` : ''}
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {app.result && (
            <Button
              variant="secondary"
              size="sm"
              icon={<RefreshCw aria-hidden className="h-4 w-4" />}
              loading={app.applyBusy}
              title="Сопоставить говорящих с именами по актуальной библиотеке голосов"
              onClick={() => void app.runApplyNames(jobId)}
            >
              Переопределить говорящих
            </Button>
          )}
          {live && (
            <Button
              variant="secondary"
              size="sm"
              icon={<Square aria-hidden className="h-3.5 w-3.5" />}
              onClick={() => void app.stopJob(jobId)}
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
              onClick={() => void app.runJob(jobId)}
            >
              {status === 'cancelled' ? 'Запустить снова' : 'Запустить'}
            </Button>
          )}
          {canDelete && (
            <IconButton
              aria-label="Удалить задачу"
              onClick={() => {
                void app.deleteJob(jobId).then(() => navigateTo('/jobs'))
              }}
            >
              <Trash2 aria-hidden className="h-4 w-4" />
            </IconButton>
          )}
        </div>
      </div>

      {app.progress && (
        <Card>
          <CardContent className="space-y-2">
            <div className="flex items-center justify-between gap-3">
              <p className="min-w-0 truncate text-sm text-muted">
                {STATUS_LABELS[status] ?? status}
                {app.progress.message ? ` — ${app.progress.message}` : ''}
              </p>
              <span className="shrink-0 text-sm font-medium tabular-nums">
                {Math.round(app.overallPercent)}%
              </span>
            </div>
            <ProgressBar
              value={app.overallPercent}
              label="Прогресс задачи"
              tone={status === 'error' ? 'danger' : status === 'done' ? 'success' : 'primary'}
            />
          </CardContent>
        </Card>
      )}

      <Tabs aria-label="Секции задачи" value={tab} onChange={setTab} items={TABS} />

      <TabPanel value="transcript" active={tab} className="pt-4">
        <TranscriptTab jobId={jobId} />
      </TabPanel>
      <TabPanel value="speakers" active={tab} className="pt-4">
        <SpeakersTab jobId={jobId} />
      </TabPanel>
      <TabPanel value="stages" active={tab} className="pt-4">
        <StagesTab />
      </TabPanel>
      <TabPanel value="chat" active={tab} className="pt-4">
        <ChatTab jobId={jobId} />
      </TabPanel>
      <TabPanel value="export" active={tab} className="pt-4">
        <ExportTab jobId={jobId} />
      </TabPanel>
    </div>
  )
}

export default JobPage
