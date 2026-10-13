import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Search } from 'lucide-react'

import { formatDuration, type Entry, type TextSuggestion } from '../../api'
import { useApp } from '../../app/useApp'
import EditorPanel from '../../components/EditorPanel'
import SemanticSuggestionsPanel from '../../components/SemanticSuggestionsPanel'
import TranscriptTable from '../../components/TranscriptTable'
import { EmptyState, Input, cn } from '../../components/ui'
import { TRANSCRIPT_MARKS } from '../../components/transcriptMarks'

function TranscriptTab({ jobId, initialEntry }: { jobId: string; initialEntry?: number }) {
  const app = useApp()
  const result = app.result

  // Правки редактора и семантики (#120) — для подсветки в стенограмме.
  const [editorSuggestions, setEditorSuggestions] = useState<TextSuggestion[]>([])
  const [semanticSuggestions, setSemanticSuggestions] = useState<TextSuggestion[]>([])
  const [editorRejected, setEditorRejected] = useState<string[]>([])
  const [semanticRejected, setSemanticRejected] = useState<string[]>([])
  const [selectedSuggestionId, setSelectedSuggestionId] = useState<string | null>(null)
  const [hoveredSuggestionId, setHoveredSuggestionId] = useState<string | null>(null)
  const [focusTick, setFocusTick] = useState(0)

  // Переход к реплике из архива (#99): отдельный фокус, чтобы не мешать
  // переходу из сводки правок (#120).
  const [deepLinkEntry, setDeepLinkEntry] = useState<Entry | null>(null)
  const [deepLinkTick, setDeepLinkTick] = useState(0)
  const deepLinkApplied = useRef<string | null>(null)

  const suggestions = useMemo(
    () => [...editorSuggestions, ...semanticSuggestions],
    [editorSuggestions, semanticSuggestions],
  )

  const rejectedSuggestionIds = useMemo(
    () => new Set([...editorRejected, ...semanticRejected]),
    [editorRejected, semanticRejected],
  )

  // Правки по репликам: ключ — сам объект реплики (совпадает с filteredEntries).
  const suggestionsByEntry = useMemo(() => {
    const map = new Map<Entry, TextSuggestion[]>()
    if (!result) return map
    for (const item of suggestions) {
      const entry = result.entries[item.index]
      if (!entry) continue
      const list = map.get(entry)
      if (list) list.push(item)
      else map.set(entry, [item])
    }
    return map
  }, [suggestions, result])

  const selectedSuggestion =
    suggestions.find((item) => item.id === selectedSuggestionId) ?? null
  const focusEntry =
    selectedSuggestion && result
      ? result.entries[selectedSuggestion.index] ?? null
      : null

  // Клик по правке в сводке: сброс фильтра (чтобы реплика была видна) и переход.
  const navigateToSuggestion = useCallback(
    (item: TextSuggestion) => {
      setSelectedSuggestionId(item.id)
      app.setQuery('')
      setFocusTick((tick) => tick + 1)
    },
    [app],
  )

  useEffect(() => {
    setSelectedSuggestionId(null)
    setHoveredSuggestionId(null)
  }, [jobId])

  // Переход к конкретной реплике из архива (#99): результат загружается
  // асинхронно, поэтому ждём его появления и наводим фокус один раз.
  useEffect(() => {
    if (initialEntry == null || !result) return
    const key = `${jobId}:${initialEntry}`
    if (deepLinkApplied.current === key) return
    const entry = result.entries[initialEntry]
    if (!entry) return
    deepLinkApplied.current = key
    app.setQuery('')
    setDeepLinkEntry(entry)
    setDeepLinkTick((tick) => tick + 1)
  }, [initialEntry, jobId, result, app])

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
        suggestionsByEntry={suggestionsByEntry}
        accentSuggestionId={hoveredSuggestionId ?? selectedSuggestionId}
        rejectedSuggestionIds={rejectedSuggestionIds}
        focusEntry={focusEntry ?? deepLinkEntry}
        focusTick={focusTick + deepLinkTick}
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

      <EditorPanel
        jobId={jobId}
        onResult={app.setResult}
        onSuggestions={setEditorSuggestions}
        onRejectedChange={setEditorRejected}
        onSelect={navigateToSuggestion}
        onHover={setHoveredSuggestionId}
        activeSuggestionId={selectedSuggestionId}
      />

      <SemanticSuggestionsPanel
        jobId={jobId}
        onResult={app.setResult}
        onSuggestions={setSemanticSuggestions}
        onRejectedChange={setSemanticRejected}
        onSelect={navigateToSuggestion}
        onHover={setHoveredSuggestionId}
        activeSuggestionId={selectedSuggestionId}
      />
    </div>
  )
}

export default TranscriptTab
