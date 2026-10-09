import { Search } from 'lucide-react'

import { formatDuration, type Entry } from '../../api'
import { useApp } from '../../app/useApp'
import EditorPanel from '../../components/EditorPanel'
import SemanticSuggestionsPanel from '../../components/SemanticSuggestionsPanel'
import TranscriptTable from '../../components/TranscriptTable'
import { EmptyState, Input, cn } from '../../components/ui'
import { TRANSCRIPT_MARKS } from '../../components/transcriptMarks'

function TranscriptTab({ jobId }: { jobId: string }) {
  const app = useApp()
  const result = app.result

  if (!result) {
    return (
      <EmptyState
        title="Стенограммы пока нет"
        description="Она появится после завершения обработки"
      />
    )
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <div className="relative min-w-52 flex-1">
          <Search
            aria-hidden
            className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted"
          />
          <label htmlFor="transcript-search" className="sr-only">
            Поиск по тексту
          </label>
          <Input
            id="transcript-search"
            value={app.query}
            onChange={(event) => app.setQuery(event.target.value)}
            placeholder="Поиск по тексту…"
            className="pl-8"
          />
        </div>
        <span className="text-xs tabular-nums text-muted">
          {result.entries.length} реплик · {result.speakers.length} говорящих ·{' '}
          {formatDuration(result.duration)} · язык {result.language ?? '—'}
        </span>
      </div>

      <div className="flex flex-wrap items-center gap-4 text-xs text-muted">
        {result.marks.map((mark) => {
          const visual = TRANSCRIPT_MARKS[mark.key]
          const Icon = visual?.Icon
          return (
            <span key={mark.key} className="inline-flex items-center gap-1">
              {Icon ? (
                <Icon aria-hidden className={cn('h-4 w-4', visual.tone)} />
              ) : (
                <span aria-hidden className="text-base">
                  {mark.symbol}
                </span>
              )}
              {mark.label}
            </span>
          )
        })}
        <span className="tabular-nums">
          {app.filteredEntries.length} из {result.entries.length}
        </span>
      </div>

      <TranscriptTable
        jobId={jobId}
        entries={app.filteredEntries}
        speakers={result.speakers}
        sourceName={app.activeJob?.name}
        onSaveText={(entry: Entry, text) => {
          const index = result.entries.indexOf(entry)
          if (index < 0) return Promise.resolve()
          return app.patchTranscript(jobId, { edits: [{ index, text }] })
        }}
        onResetText={(entry: Entry) => {
          const index = result.entries.indexOf(entry)
          if (index < 0) return Promise.resolve()
          return app.patchTranscript(jobId, { resets: [index] })
        }}
        onAssignSpeaker={(entries, target) => app.assignSpeaker(jobId, entries, target)}
        onSplitEntry={(entry, boundary, first, second) =>
          app.splitEntry(jobId, entry, boundary, first, second)
        }
        onAddExtraSpeaker={(entries, target, remove) =>
          app.editExtraSpeaker(jobId, entries, target, remove)
        }
        onUndoAssign={() => app.undoSpeakers(jobId)}
        undoAvailable={app.speakerUndoAvailable}
      />

      <EditorPanel jobId={jobId} onResult={app.setResult} />

      <SemanticSuggestionsPanel jobId={jobId} onResult={app.setResult} />
    </div>
  )
}

export default TranscriptTab
