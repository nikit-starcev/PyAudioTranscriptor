import { useApp } from '../../app/useApp'
import ProgressSummary from '../../components/ProgressSummary'
import StageTimes from '../../components/StageTimes'

function StagesTab() {
  const app = useApp()
  const status = app.progress?.status ?? app.activeJob?.status ?? 'queued'

  return (
    <div className="space-y-4">
      <ProgressSummary
        progress={app.progress}
        running={app.progressRunning}
        asrDevice={app.asrDevice}
      />
      <StageTimes
        times={app.stageTimes}
        plannedStages={app.plannedStages}
        running={app.progressRunning}
        status={status}
        totalStartedAt={app.totalStartedAt}
        finalTotalSeconds={app.finalTotalSeconds}
        currentStage={app.liveStage}
        currentStartedAt={app.liveStageStartedAt}
        failedStage={app.failedStage}
      />
    </div>
  )
}

export default StagesTab
