import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { MouseEvent as ReactMouseEvent } from 'react'
import { Pause, Pencil, Play, Scissors, X } from 'lucide-react'

import { formatTime, speakerName, type Entry, type SpeakerInfo } from '../api'
import {
  Alert,
  Button,
  Card,
  Checkbox,
  IconButton,
  Input,
  Select,
  Textarea,
  Tooltip,
  cn,
} from './ui'
import { TRANSCRIPT_MARKS } from './transcriptMarks'
import GlossaryQuickModal from './GlossaryQuickModal'

type Props = {
  jobId: string
  entries: Entry[]
  speakers: SpeakerInfo[]
  /** Имя записи/файла активной задачи — «источник» по умолчанию (#31). */
  sourceName?: string
  /** Сохранить ручную правку текста реплики (#26). */
  onSaveText?: (entry: Entry, text: string) => Promise<void>
  /** Сбросить реплику к исходному тексту (#26). */
  onResetText?: (entry: Entry) => Promise<void>
  /** Принудительно назначить говорящего выделенным репликам (#59). */
  onAssignSpeaker?: (entries: Entry[], target: AssignTarget) => Promise<void>
  /** Разрезать реплику по времени на двух говорящих (#78). */
  onSplitEntry?: (
    entry: Entry,
    boundary: number,
    first: AssignTarget,
    second: AssignTarget,
  ) => Promise<void>
  /** Добавить (`remove=false`) или убрать (`remove=true`) второго говорящего (#78). */
  onAddExtraSpeaker?: (
    entries: Entry[],
    target: AssignTarget,
    remove: boolean,
  ) => Promise<void>
  /** Отменить последнее назначение говорящего (#59). */
  onUndoAssign?: () => Promise<void>
  /** Доступна ли одношаговая отмена назначения. */
  undoAvailable?: boolean
}

type AssignTarget = { speakerId?: string; newName?: string }

/** Значение селекта «создать нового говорящего». */
const NEW_SPEAKER = '__new__'

/** Подсказка границы разреза: середина самого большого промежутка между словами. */
function suggestBoundary(entry: Entry): number {
  const words = entry.words ?? []
  const middle = (entry.start + entry.end) / 2
  if (words.length < 2) return middle
  let best = middle
  let bestGap = -1
  for (let index = 1; index < words.length; index += 1) {
    const gap = words[index].start - words[index - 1].end
    if (gap > bestGap) {
      bestGap = gap
      best = (words[index - 1].end + words[index].start) / 2
    }
  }
  if (bestGap < 0) return middle
  return Math.min(Math.max(best, entry.start), entry.end)
}

type Fragment = { key: string; start: number; end: number }

type SpeakerPiece = { id: string | null; name: string; extra: boolean }

type ContextMenu = { x: number; y: number; term: string }

function entryKey(entry: Entry, index: number): string {
  return `${entry.start}-${entry.end}-${index}`
}

//: Разбивает подпись реплики на основного говорящего и участников наложения.
//: Для старых результатов без поля `extra_speaker_ids` доп. говорящих нет.
function speakerPieces(speakers: SpeakerInfo[], entry: Entry): SpeakerPiece[] {
  const pieces: SpeakerPiece[] = [
    { id: entry.speaker_id, name: speakerName(speakers, entry.speaker_id), extra: false },
  ]
  for (const extraId of entry.extra_speaker_ids ?? []) {
    pieces.push({ id: extraId, name: speakerName(speakers, extraId), extra: true })
  }
  return pieces
}

//: Минимальная длина фрагмента, чтобы не делить на ноль в прогрессе.
const MIN_FRAGMENT = 0.05
//: Точность остановки: останавливаемся чуть раньше `end`, чтобы не зацепить
//: следующий звук из-за округления `currentTime`.
const STOP_EPSILON = 0.005
//: Как часто обновлять прогресс (мс), чтобы не ререндерить таблицу каждый кадр.
const PAINT_INTERVAL_MS = 100

type MarkProps = { markKey: string; label: string }

function Mark({ markKey, label }: MarkProps) {
  const visual = TRANSCRIPT_MARKS[markKey]
  if (!visual) return null
  const { Icon, tone } = visual
  return (
    <Tooltip label={label}>
      <span className={cn('inline-flex items-center', tone)}>
        <Icon aria-hidden className="h-4 w-4" />
        <span className="sr-only">{label}</span>
      </span>
    </Tooltip>
  )
}

function TranscriptTable({
  jobId,
  entries,
  speakers,
  sourceName,
  onSaveText,
  onResetText,
  onAssignSpeaker,
  onSplitEntry,
  onAddExtraSpeaker,
  onUndoAssign,
  undoAvailable,
}: Props) {
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const [playingKey, setPlayingKey] = useState<string | null>(null)
  const [position, setPosition] = useState(0)
  const [autoAdvance, setAutoAdvance] = useState(false)

  const fragmentRef = useRef<Fragment | null>(null)
  const rafRef = useRef<number | null>(null)
  const lastPaintRef = useRef(0)
  const playRef = useRef<(fragment: Fragment) => void>(() => {})

  // Ручная правка текста (#26): ключ редактируемой реплики и черновик.
  const [editingKey, setEditingKey] = useState<string | null>(null)
  const [draftText, setDraftText] = useState('')
  const [savingKey, setSavingKey] = useState<string | null>(null)
  const [editError, setEditError] = useState<string | null>(null)

  // Контекстное меню «Добавить в глоссарий» (#17) и модальное окно быстрого
  // добавления термина.
  const [contextMenu, setContextMenu] = useState<ContextMenu | null>(null)
  const [quickTerm, setQuickTerm] = useState<string | null>(null)

  // Принудительное назначение говорящего выделенным репликам (#59).
  const [selectedKeys, setSelectedKeys] = useState<Set<string>>(() => new Set())
  const [assignTargetId, setAssignTargetId] = useState('')
  const [newSpeakerName, setNewSpeakerName] = useState('')
  const [assignBusy, setAssignBusy] = useState(false)

  // Разрезание реплики на двух говорящих (#78): ключ реплики и черновик формы.
  const [splitKey, setSplitKey] = useState<string | null>(null)
  const [splitBoundary, setSplitBoundary] = useState('')
  const [splitFirstId, setSplitFirstId] = useState('')
  const [splitFirstNew, setSplitFirstNew] = useState('')
  const [splitSecondId, setSplitSecondId] = useState('')
  const [splitSecondNew, setSplitSecondNew] = useState('')
  const [splitBusy, setSplitBusy] = useState(false)

  // Актуальные значения для цикла requestAnimationFrame (без пересоздания).
  const entriesRef = useRef(entries)
  const autoAdvanceRef = useRef(autoAdvance)

  useEffect(() => {
    entriesRef.current = entries
  }, [entries])

  useEffect(() => {
    autoAdvanceRef.current = autoAdvance
  }, [autoAdvance])

  // Закрываем контекстное меню при клике/скролле/смене размера/задач.
  useEffect(() => {
    if (!contextMenu) return
    const close = () => setContextMenu(null)
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') close()
    }
    window.addEventListener('click', close)
    window.addEventListener('scroll', close, true)
    window.addEventListener('resize', close)
    window.addEventListener('keydown', onKey)
    return () => {
      window.removeEventListener('click', close)
      window.removeEventListener('scroll', close, true)
      window.removeEventListener('resize', close)
      window.removeEventListener('keydown', onKey)
    }
  }, [contextMenu])

  // При смене задачи выходим из режима правки.
  useEffect(() => {
    setEditingKey(null)
    setEditError(null)
    setContextMenu(null)
    setSplitKey(null)
  }, [jobId])

  // Выделение реплик сбрасываем при смене задачи или списка реплик (после
  // правки/назначения результат приходит заново — выделять больше нечего).
  useEffect(() => {
    setSelectedKeys(new Set())
  }, [jobId, entries])

  const cancelLoop = useCallback(() => {
    if (rafRef.current != null) {
      cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }
  }, [])

  const stop = useCallback(() => {
    cancelLoop()
    fragmentRef.current = null
    audioRef.current?.pause()
    setPlayingKey(null)
    setPosition(0)
  }, [cancelLoop])

  const playFragment = useCallback(
    (fragment: Fragment) => {
      const audio = audioRef.current
      if (!audio) return
      cancelLoop()
      fragmentRef.current = fragment
      setPlayingKey(fragment.key)
      setPosition(0)
      lastPaintRef.current = 0

      const startLoop = () => {
        cancelLoop()
        const tick = (now: number) => {
          const el = audioRef.current
          const current = fragmentRef.current
          if (!el || !current) return
          if (el.currentTime >= current.end - STOP_EPSILON) {
            el.pause()
            fragmentRef.current = null
            setPlayingKey(null)
            setPosition(0)
            cancelLoop()
            if (autoAdvanceRef.current) {
              const list = entriesRef.current
              const index = list.findIndex((item, i) => entryKey(item, i) === current.key)
              const next = index >= 0 ? list[index + 1] : undefined
              if (next) {
                playRef.current({
                  key: entryKey(next, index + 1),
                  start: next.start,
                  end: next.end,
                })
              }
            }
            return
          }
          // Автопауза на конце файла (если end вышел за длительность).
          if (el.ended) {
            stop()
            return
          }
          if (now - lastPaintRef.current >= PAINT_INTERVAL_MS) {
            lastPaintRef.current = now
            setPosition(Math.max(0, el.currentTime - current.start))
          }
          rafRef.current = requestAnimationFrame(tick)
        }
        rafRef.current = requestAnimationFrame(tick)
      }

      const seekAndPlay = () => {
        audio.currentTime = Math.max(0, fragment.start)
        void audio.play().then(startLoop).catch(() => stop())
      }

      if (audio.readyState >= 1 /* HAVE_METADATA */) {
        seekAndPlay()
      } else {
        const onReady = () => {
          audio.removeEventListener('loadedmetadata', onReady)
          seekAndPlay()
        }
        audio.addEventListener('loadedmetadata', onReady)
        audio.load()
      }
    },
    [cancelLoop, stop],
  )
  useEffect(() => {
    playRef.current = playFragment
  }, [playFragment])

  // Останавливаем воспроизведение при смене задачи и размонтировании.
  useEffect(() => stop, [jobId, stop])

  const toggle = (fragment: Fragment) => {
    if (playingKey === fragment.key) stop()
    else playFragment(fragment)
  }

  const beginEdit = useCallback(
    (key: string, text: string) => {
      if (!onSaveText) return
      setEditingKey(key)
      setDraftText(text)
      setEditError(null)
    },
    [onSaveText],
  )

  const saveEdit = useCallback(
    async (entry: Entry, key: string) => {
      if (!onSaveText) return
      setSavingKey(key)
      setEditError(null)
      try {
        await onSaveText(entry, draftText)
        setEditingKey(null)
      } catch (cause) {
        setEditError(cause instanceof Error ? cause.message : String(cause))
      } finally {
        setSavingKey(null)
      }
    },
    [onSaveText, draftText],
  )

  const resetEdit = useCallback(
    async (entry: Entry) => {
      if (!onResetText) return
      setEditError(null)
      try {
        await onResetText(entry)
      } catch (cause) {
        setEditError(cause instanceof Error ? cause.message : String(cause))
      }
    },
    [onResetText],
  )

  const openContextMenu = useCallback((event: ReactMouseEvent) => {
    const selected = window.getSelection()?.toString().trim()
    if (!selected) return
    event.preventDefault()
    setContextMenu({ x: event.clientX, y: event.clientY, term: selected })
  }, [])

  const selectedEntries = onAssignSpeaker
    ? entries.filter((entry, index) => selectedKeys.has(entryKey(entry, index)))
    : []
  const allSelected = entries.length > 0 && selectedEntries.length === entries.length

  const splitTarget = useMemo(() => {
    if (!splitKey) return undefined
    const index = entries.findIndex((entry, i) => entryKey(entry, i) === splitKey)
    return index >= 0 ? entries[index] : undefined
  }, [splitKey, entries])

  const toggleRow = useCallback((key: string) => {
    setSelectedKeys((current) => {
      const next = new Set(current)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }, [])

  const toggleAll = useCallback(() => {
    setSelectedKeys((current) => {
      const keys = entries.map((entry, index) => entryKey(entry, index))
      const everySelected = keys.length > 0 && keys.every((key) => current.has(key))
      return everySelected ? new Set() : new Set(keys)
    })
  }, [entries])

  const undoAssign = useCallback(async () => {
    if (!onUndoAssign) return
    setEditError(null)
    setAssignBusy(true)
    try {
      await onUndoAssign()
    } catch (cause) {
      setEditError(cause instanceof Error ? cause.message : String(cause))
    } finally {
      setAssignBusy(false)
    }
  }, [onUndoAssign])

  const submitAssign = useCallback(async () => {
    if (!onAssignSpeaker || selectedKeys.size === 0) return
    const chosen = entries.filter((entry, index) => selectedKeys.has(entryKey(entry, index)))
    if (chosen.length === 0) return
    const target: AssignTarget =
      assignTargetId === NEW_SPEAKER
        ? { newName: newSpeakerName.trim() }
        : { speakerId: assignTargetId }
    setAssignBusy(true)
    setEditError(null)
    try {
      await onAssignSpeaker(chosen, target)
      setSelectedKeys(new Set())
      setNewSpeakerName('')
      setAssignTargetId('')
    } catch (cause) {
      setEditError(cause instanceof Error ? cause.message : String(cause))
    } finally {
      setAssignBusy(false)
    }
  }, [onAssignSpeaker, entries, selectedKeys, assignTargetId, newSpeakerName])

  const assignReady =
    assignTargetId === NEW_SPEAKER
      ? newSpeakerName.trim().length > 0
      : assignTargetId.length > 0

  const chosenEntries = useCallback(
    () => entries.filter((entry, index) => selectedKeys.has(entryKey(entry, index))),
    [entries, selectedKeys],
  )

  const resolveTarget = useCallback(
    (speakerId: string, newName: string): AssignTarget =>
      speakerId === NEW_SPEAKER ? { newName: newName.trim() } : { speakerId },
    [],
  )

  const submitExtra = useCallback(async () => {
    if (!onAddExtraSpeaker) return
    const chosen = chosenEntries()
    if (chosen.length === 0) return
    const target = resolveTarget(assignTargetId, newSpeakerName)
    setAssignBusy(true)
    setEditError(null)
    try {
      await onAddExtraSpeaker(chosen, target, false)
      setSelectedKeys(new Set())
      setNewSpeakerName('')
      setAssignTargetId('')
    } catch (cause) {
      setEditError(cause instanceof Error ? cause.message : String(cause))
    } finally {
      setAssignBusy(false)
    }
  }, [onAddExtraSpeaker, chosenEntries, resolveTarget, assignTargetId, newSpeakerName])

  const removeExtra = useCallback(
    async (entry: Entry, speakerId: string) => {
      if (!onAddExtraSpeaker) return
      setEditError(null)
      try {
        await onAddExtraSpeaker([entry], { speakerId }, true)
      } catch (cause) {
        setEditError(cause instanceof Error ? cause.message : String(cause))
      }
    },
    [onAddExtraSpeaker],
  )

  const beginSplit = useCallback(
    (entry: Entry, index: number) => {
      if (!onSplitEntry) return
      setSplitKey(entryKey(entry, index))
      setSplitBoundary(suggestBoundary(entry).toFixed(2))
      setSplitFirstId(entry.speaker_id ?? '')
      setSplitFirstNew('')
      setSplitSecondId('')
      setSplitSecondNew('')
      setEditError(null)
    },
    [onSplitEntry],
  )

  const suggestSplitBoundary = useCallback(() => {
    if (!splitKey) return
    const index = entries.findIndex((entry, i) => entryKey(entry, i) === splitKey)
    const entry = index >= 0 ? entries[index] : undefined
    if (entry) setSplitBoundary(suggestBoundary(entry).toFixed(2))
  }, [splitKey, entries])

  const submitSplit = useCallback(async () => {
    if (!onSplitEntry || !splitKey) return
    const index = entries.findIndex((entry, i) => entryKey(entry, i) === splitKey)
    const entry = index >= 0 ? entries[index] : undefined
    if (!entry) return
    const boundary = Number(splitBoundary.replace(',', '.'))
    if (!Number.isFinite(boundary) || boundary <= entry.start || boundary >= entry.end) {
      setEditError('Граница должна быть строго внутри реплики')
      return
    }
    const first = resolveTarget(splitFirstId, splitFirstNew)
    const second = resolveTarget(splitSecondId, splitSecondNew)
    if (!first.speakerId && !first.newName) {
      setEditError('Укажите говорящего первой части')
      return
    }
    if (!second.speakerId && !second.newName) {
      setEditError('Укажите говорящего второй части')
      return
    }
    setSplitBusy(true)
    setEditError(null)
    try {
      await onSplitEntry(entry, boundary, first, second)
      setSplitKey(null)
    } catch (cause) {
      setEditError(cause instanceof Error ? cause.message : String(cause))
    } finally {
      setSplitBusy(false)
    }
  }, [
    onSplitEntry,
    splitKey,
    entries,
    splitBoundary,
    splitFirstId,
    splitFirstNew,
    splitSecondId,
    splitSecondNew,
    resolveTarget,
  ])

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-3 text-xs text-muted">
        <Checkbox
          label="Автопереход к следующей реплике"
          checked={autoAdvance}
          onChange={(event) => setAutoAdvance(event.target.checked)}
        />
        {playingKey ? (
          <span className="flex items-center gap-1 font-medium text-primary">
            <span aria-hidden className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-primary" />
            воспроизведение фрагмента
          </span>
        ) : (
          <span>Двойной клик по тексту — правка; ПКМ по выделению — в глоссарий</span>
        )}
        {onAssignSpeaker && (
          <Checkbox
            label="Выделить все для назначения говорящего"
            checked={allSelected}
            onChange={toggleAll}
            aria-label="Выделить все реплики"
          />
        )}
        {onUndoAssign && undoAvailable && (
          <Button
            variant="secondary"
            size="sm"
            disabled={assignBusy}
            title="Вернуть говорящих к состоянию до последнего назначения"
            onClick={() => void undoAssign()}
          >
            Отменить назначение
          </Button>
        )}
      </div>

      {onAssignSpeaker && selectedEntries.length > 0 && (
        <div className="flex flex-wrap items-center gap-2 rounded-md border border-info/40 bg-info-soft px-3 py-2 text-xs text-info-soft-fg">
          <span className="font-medium">Выбрано реплик: {selectedEntries.length}</span>
          <Select
            aria-label="Говорящий для назначения"
            value={assignTargetId}
            onChange={(event) => setAssignTargetId(event.target.value)}
          >
            <option value="">— говорящий —</option>
            {speakers.map((speaker) => (
              <option key={speaker.id} value={speaker.id}>
                {speaker.display_name}
              </option>
            ))}
            <option value={NEW_SPEAKER}>＋ новый говорящий…</option>
          </Select>
          {assignTargetId === NEW_SPEAKER && (
            <Input
              aria-label="Имя нового говорящего"
              value={newSpeakerName}
              onChange={(event) => setNewSpeakerName(event.target.value)}
              placeholder="Имя нового говорящего"
              className="w-44"
            />
          )}
          <Button
            variant="primary"
            size="sm"
            loading={assignBusy}
            disabled={!assignReady}
            onClick={() => void submitAssign()}
          >
            Назначить
          </Button>
          {onAddExtraSpeaker && (
            <Button
              variant="secondary"
              size="sm"
              disabled={!assignReady || assignBusy}
              title="Добавить выбранного говорящего вторым (наложение), не меняя основного"
              onClick={() => void submitExtra()}
            >
              + второй говорящий
            </Button>
          )}
          <Button variant="ghost" size="sm" onClick={() => setSelectedKeys(new Set())}>
            Снять выделение
          </Button>
        </div>
      )}

      {onSplitEntry && splitKey && splitTarget && (
        <div className="flex flex-wrap items-center gap-2 rounded-md border border-warn/40 bg-warn-soft px-3 py-2 text-xs text-warn-soft-fg">
          <span className="font-medium">
            Разделить реплику {formatTime(splitTarget.start)}–{formatTime(splitTarget.end)}
          </span>
          <label className="flex items-center gap-1">
            Граница, с
            <Input
              type="number"
              min={splitTarget.start}
              max={splitTarget.end}
              step={0.01}
              value={splitBoundary}
              onChange={(event) => setSplitBoundary(event.target.value)}
              className="w-24"
            />
          </label>
          <Button
            variant="secondary"
            size="sm"
            title={
              (splitTarget.words?.length ?? 0) >= 2
                ? 'Подсказать границу по пословным таймкодам'
                : 'Пословных таймкодов нет — середина реплики'
            }
            onClick={suggestSplitBoundary}
          >
            Подсказать
          </Button>
          <Select
            aria-label="Говорящий первой части"
            value={splitFirstId}
            onChange={(event) => setSplitFirstId(event.target.value)}
          >
            <option value="">— 1-я часть —</option>
            {speakers.map((speaker) => (
              <option key={speaker.id} value={speaker.id}>
                {speaker.display_name}
              </option>
            ))}
            <option value={NEW_SPEAKER}>＋ новый говорящий…</option>
          </Select>
          {splitFirstId === NEW_SPEAKER && (
            <Input
              aria-label="Имя говорящего первой части"
              value={splitFirstNew}
              onChange={(event) => setSplitFirstNew(event.target.value)}
              placeholder="Имя"
              className="w-28"
            />
          )}
          <span className="text-muted">+</span>
          <Select
            aria-label="Говорящий второй части"
            value={splitSecondId}
            onChange={(event) => setSplitSecondId(event.target.value)}
          >
            <option value="">— 2-я часть —</option>
            {speakers.map((speaker) => (
              <option key={speaker.id} value={speaker.id}>
                {speaker.display_name}
              </option>
            ))}
            <option value={NEW_SPEAKER}>＋ новый говорящий…</option>
          </Select>
          {splitSecondId === NEW_SPEAKER && (
            <Input
              aria-label="Имя говорящего второй части"
              value={splitSecondNew}
              onChange={(event) => setSplitSecondNew(event.target.value)}
              placeholder="Имя"
              className="w-28"
            />
          )}
          <Button variant="primary" size="sm" loading={splitBusy} onClick={() => void submitSplit()}>
            Разделить
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              setSplitKey(null)
              setEditError(null)
            }}
          >
            Отмена
          </Button>
        </div>
      )}

      {editError && (
        <Alert tone="danger" live onDismiss={() => setEditError(null)}>
          {editError}
        </Alert>
      )}

      <audio ref={audioRef} preload="metadata" src={`/api/jobs/${jobId}/audio`} className="hidden" />

      <Card className="overflow-hidden">
        <div className="max-h-[28rem] overflow-auto">
          <table className="block w-full border-collapse text-sm md:table">
            <thead className="sticky top-0 z-10 hidden bg-surface-3 text-left text-xs uppercase text-muted md:table-header-group">
              <tr>
                {onAssignSpeaker && (
                  <th className="w-8 px-2 py-2 font-medium" aria-label="Выделить реплику" />
                )}
                <th className="w-10 px-2 py-2 font-medium" aria-label="Прослушать" />
                <th className="px-3 py-2 font-medium">Время</th>
                <th className="px-3 py-2 font-medium">Говорящий</th>
                <th className="px-3 py-2 font-medium">Метки</th>
                <th className="px-3 py-2 font-medium">Текст</th>
              </tr>
            </thead>
            <tbody className="block md:table-row-group">
              {entries.map((entry, index) => {
                const key = entryKey(entry, index)
                const playing = playingKey === key
                const editing = editingKey === key && onSaveText != null
                const selected = selectedKeys.has(key)
                const span = Math.max(MIN_FRAGMENT, entry.end - entry.start)
                const percent = playing ? Math.min(100, Math.max(0, (position / span) * 100)) : 0
                const pieces = speakerPieces(speakers, entry)
                return (
                  <tr
                    key={key}
                    className={cn(
                      'group mb-2 flex flex-wrap items-center gap-x-2 gap-y-1 rounded-md border border-border p-3',
                      'md:mb-0 md:table-row md:rounded-none md:border-x-0 md:border-b-0 md:border-t md:p-0',
                      playing && 'bg-info-soft',
                      !playing && selected && 'bg-warn-soft',
                    )}
                  >
                    {onAssignSpeaker && (
                      <td className="px-2 py-1.5">
                        <Checkbox
                          checked={selected}
                          onChange={() => toggleRow(key)}
                          aria-label={`Выделить реплику ${index + 1}`}
                        />
                      </td>
                    )}
                    <td className="px-2 py-1.5">
                      <IconButton
                        aria-label={playing ? 'Остановить фрагмент' : 'Прослушать фрагмент'}
                        aria-pressed={playing}
                        title={
                          playing
                            ? 'Остановить'
                            : `Прослушать ${formatTime(entry.start)}–${formatTime(entry.end)}`
                        }
                        variant={playing ? 'primary' : 'secondary'}
                        size="sm"
                        onClick={() => toggle({ key, start: entry.start, end: entry.end })}
                      >
                        {playing ? (
                          <Pause aria-hidden className="h-3.5 w-3.5" />
                        ) : (
                          <Play aria-hidden className="h-3.5 w-3.5" />
                        )}
                      </IconButton>
                    </td>
                    <td className="whitespace-nowrap px-3 py-1.5 font-mono text-xs tabular-nums text-muted">
                      {formatTime(entry.start)}
                    </td>
                    <td className="order-2 min-w-0 basis-full px-3 py-1.5 md:order-none md:basis-auto">
                      <div className="flex items-start gap-1.5">
                        <div
                          className="min-w-0 whitespace-normal break-words leading-snug sm:max-w-64"
                          title={pieces.map((piece) => piece.name).join(' + ')}
                        >
                          {pieces.map((piece, pieceIndex) => (
                            <span key={`${piece.id ?? 'none'}-${pieceIndex}`}>
                              {pieceIndex > 0 && <span className="mx-1 text-muted">+</span>}
                              <span
                                className={cn(piece.extra && 'text-muted')}
                                title={piece.extra ? 'дополнительный говорящий (наложение)' : undefined}
                              >
                                {piece.name}
                              </span>
                              {piece.extra && piece.id && onAddExtraSpeaker && (
                                <IconButton
                                  aria-label="Убрать второго говорящего"
                                  title="Убрать второго говорящего"
                                  size="sm"
                                  className="ml-1 align-middle"
                                  onClick={() => void removeExtra(entry, piece.id as string)}
                                >
                                  <X aria-hidden className="h-3 w-3" />
                                </IconButton>
                              )}
                            </span>
                          ))}
                        </div>
                        {onSplitEntry && (
                          <IconButton
                            aria-label="Разделить реплику по времени на двух говорящих"
                            title="Разделить реплику"
                            size="sm"
                            className={cn(
                              'mt-0.5 shrink-0 opacity-0 transition-opacity',
                              'focus-visible:opacity-100 group-hover:opacity-100',
                              'group-focus-within:opacity-100 pointer-coarse:opacity-100',
                            )}
                            onClick={() => beginSplit(entry, index)}
                          >
                            <Scissors aria-hidden className="h-3.5 w-3.5" />
                          </IconButton>
                        )}
                      </div>
                    </td>
                    <td className="order-1 whitespace-nowrap px-3 py-1.5 md:order-none">
                      <span className="inline-flex items-center gap-1.5">
                        {entry.low_confidence && (
                          <Mark markKey="low_confidence" label="низкая уверенность" />
                        )}
                        {entry.low_speaker_confidence && (
                          <Mark markKey="speaker_uncertain" label="говорящий под вопросом" />
                        )}
                        {entry.overlap && <Mark markKey="overlap" label="наложение речи" />}
                      </span>
                    </td>
                    <td
                      className="order-3 min-w-0 basis-full px-3 py-1.5 md:order-none md:basis-auto"
                      onContextMenu={(event) => openContextMenu(event)}
                    >
                      {editing ? (
                        <div className="space-y-1">
                          <Textarea
                            aria-label="Текст реплики"
                            value={draftText}
                            onChange={(event) => setDraftText(event.target.value)}
                            rows={2}
                            autoFocus
                          />
                          <div className="flex items-center gap-2">
                            <Button
                              variant="primary"
                              size="sm"
                              loading={savingKey === key}
                              onClick={() => void saveEdit(entry, key)}
                            >
                              Сохранить
                            </Button>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => {
                                setEditingKey(null)
                                setEditError(null)
                              }}
                            >
                              Отмена
                            </Button>
                          </div>
                        </div>
                      ) : (
                        <div className="flex items-start gap-1.5">
                          <div className="break-words" onDoubleClick={() => beginEdit(key, entry.text)}>
                            {entry.text}
                          </div>
                          {onSaveText && (
                            <IconButton
                              aria-label="Править текст реплики"
                              title="Править текст"
                              size="sm"
                              className="mt-0.5 shrink-0"
                              onClick={() => beginEdit(key, entry.text)}
                            >
                              <Pencil aria-hidden className="h-3 w-3" />
                            </IconButton>
                          )}
                          {entry.edited && (
                            <span
                              className="mt-0.5 shrink-0 rounded bg-warn-soft px-1 py-0.5 text-[10px] font-medium text-warn-soft-fg"
                              title={
                                entry.original_text
                                  ? `Исходный текст: ${entry.original_text}`
                                  : 'изменено вручную'
                              }
                            >
                              изменено вручную
                            </span>
                          )}
                          {entry.edited && onResetText && (
                            <Button
                              variant="secondary"
                              size="sm"
                            className="mt-0.5 shrink-0"
                              title="Сбросить к исходному тексту"
                              onClick={() => void resetEdit(entry)}
                            >
                              сбросить
                            </Button>
                          )}
                        </div>
                      )}
                      {playing && (
                        <div className="mt-1 h-1 w-full overflow-hidden rounded-full bg-surface-3">
                          <div
                            className="h-full rounded-full bg-primary transition-[width] duration-100 ease-linear"
                            style={{ width: `${percent}%` }}
                          />
                        </div>
                      )}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
          {entries.length === 0 && (
            <p className="py-6 text-center text-sm text-muted">Ничего не найдено</p>
          )}
        </div>
      </Card>

      {contextMenu && (
        <div
          role="menu"
          className="fixed z-50 min-w-48 rounded-md border border-border bg-surface py-1 text-sm shadow-lg"
          style={{ left: contextMenu.x, top: contextMenu.y }}
          onClick={(event) => event.stopPropagation()}
        >
          <Button
            role="menuitem"
            variant="ghost"
            fullWidth
            onClick={() => {
              setQuickTerm(contextMenu.term)
              setContextMenu(null)
            }}
          >
            Добавить в глоссарий: «{contextMenu.term}»
          </Button>
        </div>
      )}

      <GlossaryQuickModal
        open={quickTerm != null}
        term={quickTerm ?? ''}
        source={sourceName}
        onClose={() => setQuickTerm(null)}
      />
    </div>
  )
}

export default TranscriptTable
