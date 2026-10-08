import { navigateTo } from '../../app/routes'
import { useApp } from '../../app/useApp'
import SpeakersPanel from '../../components/SpeakersPanel'
import { EmptyState } from '../../components/ui'

function SpeakersTab({ jobId }: { jobId: string }) {
  const app = useApp()
  const result = app.result

  if (!result) {
    return (
      <EmptyState
        title="Говорящих пока нет"
        description="Список появится после завершения обработки"
      />
    )
  }

  return (
    <SpeakersPanel
      jobId={jobId}
      result={result}
      sampleMeta={app.samplesMeta}
      onRename={(speakerId, name) =>
        app.patchSpeakers(jobId, { renames: { [speakerId]: name } })
      }
      onMerge={(source, target) =>
        app.patchSpeakers(jobId, { merges: [{ source, target }] })
      }
      onToLibrary={(speakerId, name, window) =>
        app.saveToLibrary(jobId, speakerId, name, window)
      }
      onReassign={(speakerId, body) => app.reassignSpeaker(jobId, speakerId, body)}
      onUndo={() => app.undoSpeakers(jobId)}
      undoAvailable={app.speakerUndoAvailable}
      onApplyNames={() => app.applyNames(jobId)}
      onOpenVoices={() => navigateTo('/voices')}
    />
  )
}

export default SpeakersTab
