import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { MouseEvent as ReactMouseEvent } from 'react'

import { formatTime, speakerName, type Entry, type SpeakerInfo } from '../api'
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

//: Иконка воспроизведения/паузы (SVG вместо эмодзи — предсказуемый вид и цвет).
function PlayerIcon({ paused }: { paused: boolean }) {
  return (
    <svg viewBox="0 0 16 16" width="10" height="10" fill="currentColor" aria-hidden="true">
      {paused ? (
        <>
          <rect x="4" y="3" width="3" height="10" rx="0.5" />
          <rect x="9" y="3" width="3" height="10" rx="0.5" />
        </>
      ) : (
        <path d="M4.5 2.8v10.4L13 8z" />
      )}
    </svg>
  )
}

//: Карандаш для входа в режим правки текста реплики.
function PencilIcon() {
  return (
    <svg viewBox="0 0 16 16" width="12" height="12" fill="currentColor" aria-hidden="true">
      <path d="M11.5 1.5a1.6 1.6 0 0 1 2.3 0l.7.7a1.6 1.6 0 0 1 0 2.3l-8 8L3 14l1.5-3.5 8-8zM3.9 11.2l-.7 1.6 1.6-.7 7.6-7.6-0.9-.9-7.6 7.6z" />
    </svg>
  )
}

//: Минимальная длина фрагмента, чтобы не делить на ноль в прогрессе.
const MIN_FRAGMENT = 0.05
//: Точность остановки: останавливаемся чуть раньше `end`, чтобы не зацепить
//: следующий звук из-за округления `currentTime`.
const STOP_EPSILON = 0.005
//: Как часто обновлять прогресс (мс), чтобы не ререндерить таблицу каждый кадр.
const PAINT_INTERVAL_MS = 100

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
      <div className="flex flex-wrap items-center gap-3 text-xs text-slate-500 dark:text-slate-400">
        <label className="flex items-center gap-1.5">
          <input
            type="checkbox"
            checked={autoAdvance}
            onChange={(event) => setAutoAdvance(event.target.checked)}
          />
          Автопереход к следующей реплике
        </label>
        {playingKey ? (
          <span className="flex items-center gap-1 font-medium text-blue-600 dark:text-blue-400">
            <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-blue-500" />
            воспроизведение фрагмента
          </span>
        ) : (
          <span>Двойной клик по тексту — правка; ПКМ по выделению — в глоссарий</span>
        )}
        {onAssignSpeaker && (
          <label className="flex items-center gap-1.5">
            <input
              type="checkbox"
              checked={allSelected}
              onChange={toggleAll}
              aria-label="Выделить все реплики"
            />
            Выделить все для назначения говорящего
          </label>
        )}
        {onUndoAssign && undoAvailable && (
          <button
            type="button"
            onClick={() => void undoAssign()}
            disabled={assignBusy}
            title="Вернуть говорящих к состоянию до последнего назначения"
            className="rounded border border-slate-300 px-2 py-0.5 text-xs hover:bg-slate-100 disabled:opacity-40 dark:border-slate-600 dark:text-slate-300 dark:hover:bg-slate-800"
          >
            Отменить назначение
          </button>
        )}
      </div>

      {onAssignSpeaker && selectedEntries.length > 0 && (
        <div className="flex flex-wrap items-center gap-2 rounded-md border border-blue-200 bg-blue-50 px-3 py-2 text-xs text-blue-800 dark:border-blue-900 dark:bg-blue-950/40 dark:text-blue-200">
          <span className="font-medium">Выбрано реплик: {selectedEntries.length}</span>
          <select
            value={assignTargetId}
            onChange={(event) => setAssignTargetId(event.target.value)}
            aria-label="Говорящий для назначения"
            className="rounded-md border border-slate-300 bg-white px-2 py-1 text-xs dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
          >
            <option value="">— говорящий —</option>
            {speakers.map((speaker) => (
              <option key={speaker.id} value={speaker.id}>
                {speaker.display_name}
              </option>
            ))}
            <option value={NEW_SPEAKER}>＋ новый говорящий…</option>
          </select>
          {assignTargetId === NEW_SPEAKER && (
            <input
              value={newSpeakerName}
              onChange={(event) => setNewSpeakerName(event.target.value)}
              placeholder="Имя нового говорящего"
              className="rounded-md border border-slate-300 px-2 py-1 text-xs dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
            />
          )}
          <button
            type="button"
            onClick={() => void submitAssign()}
            disabled={!assignReady || assignBusy}
            className="rounded-md bg-blue-600 px-2.5 py-1 text-xs text-white hover:bg-blue-500 disabled:opacity-40"
          >
            {assignBusy ? 'Применяю…' : 'Назначить'}
          </button>
          {onAddExtraSpeaker && (
            <button
              type="button"
              onClick={() => void submitExtra()}
              disabled={!assignReady || assignBusy}
              title="Добавить выбранного говорящего вторым (наложение), не меняя основного"
              className="rounded-md border border-blue-400 px-2.5 py-1 text-xs text-blue-700 hover:bg-blue-100 disabled:opacity-40 dark:border-blue-700 dark:text-blue-300 dark:hover:bg-blue-950/40"
            >
              + второй говорящий
            </button>
          )}
          <button
            type="button"
            onClick={() => setSelectedKeys(new Set())}
            className="rounded-md border border-slate-300 px-2 py-1 text-xs hover:bg-white/60 dark:border-slate-600 dark:hover:bg-slate-800"
          >
            Снять выделение
          </button>
        </div>
      )}

      {onSplitEntry && splitKey && splitTarget && (
        <div className="flex flex-wrap items-center gap-2 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-200">
          <span className="font-medium">
            Разделить реплику {formatTime(splitTarget.start)}–{formatTime(splitTarget.end)}
          </span>
          <label className="flex items-center gap-1">
            Граница, с
            <input
              type="number"
              min={splitTarget.start}
              max={splitTarget.end}
              step={0.01}
              value={splitBoundary}
              onChange={(event) => setSplitBoundary(event.target.value)}
              className="w-24 rounded-md border border-slate-300 px-2 py-1 text-xs dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
            />
          </label>
          <button
            type="button"
            onClick={suggestSplitBoundary}
            title={
              (splitTarget.words?.length ?? 0) >= 2
                ? 'Подсказать границу по пословным таймкодам'
                : 'Пословных таймкодов нет — середина реплики'
            }
            className="rounded-md border border-slate-300 px-2 py-1 text-xs hover:bg-white/60 dark:border-slate-600 dark:hover:bg-slate-800"
          >
            Подсказать
          </button>
          <select
            value={splitFirstId}
            onChange={(event) => setSplitFirstId(event.target.value)}
            aria-label="Говорящий первой части"
            className="rounded-md border border-slate-300 bg-white px-2 py-1 text-xs dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
          >
            <option value="">— 1-я часть —</option>
            {speakers.map((speaker) => (
              <option key={speaker.id} value={speaker.id}>
                {speaker.display_name}
              </option>
            ))}
            <option value={NEW_SPEAKER}>＋ новый говорящий…</option>
          </select>
          {splitFirstId === NEW_SPEAKER && (
            <input
              value={splitFirstNew}
              onChange={(event) => setSplitFirstNew(event.target.value)}
              placeholder="Имя"
              className="w-28 rounded-md border border-slate-300 px-2 py-1 text-xs dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
            />
          )}
          <span className="text-slate-400">+</span>
          <select
            value={splitSecondId}
            onChange={(event) => setSplitSecondId(event.target.value)}
            aria-label="Говорящий второй части"
            className="rounded-md border border-slate-300 bg-white px-2 py-1 text-xs dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
          >
            <option value="">— 2-я часть —</option>
            {speakers.map((speaker) => (
              <option key={speaker.id} value={speaker.id}>
                {speaker.display_name}
              </option>
            ))}
            <option value={NEW_SPEAKER}>＋ новый говорящий…</option>
          </select>
          {splitSecondId === NEW_SPEAKER && (
            <input
              value={splitSecondNew}
              onChange={(event) => setSplitSecondNew(event.target.value)}
              placeholder="Имя"
              className="w-28 rounded-md border border-slate-300 px-2 py-1 text-xs dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
            />
          )}
          <button
            type="button"
            onClick={() => void submitSplit()}
            disabled={splitBusy}
            className="rounded-md bg-amber-600 px-2.5 py-1 text-xs text-white hover:bg-amber-500 disabled:opacity-40"
          >
            {splitBusy ? 'Разрезаю…' : 'Разделить'}
          </button>
          <button
            type="button"
            onClick={() => {
              setSplitKey(null)
              setEditError(null)
            }}
            className="rounded-md border border-slate-300 px-2 py-1 text-xs hover:bg-white/60 dark:border-slate-600 dark:hover:bg-slate-800"
          >
            Отмена
          </button>
        </div>
      )}

      {editError && (
        <p className="rounded-md bg-red-50 px-3 py-1.5 text-xs text-red-700 dark:bg-red-950/50 dark:text-red-300">
          {editError}
        </p>
      )}

      <audio ref={audioRef} preload="metadata" src={`/api/jobs/${jobId}/audio`} className="hidden" />

      <div className="max-h-[28rem] overflow-auto rounded-md border border-slate-200 dark:border-slate-800">
        <table className="w-full border-collapse text-sm">
          <thead className="sticky top-0 z-10 bg-slate-100 text-left text-xs uppercase text-slate-500 dark:bg-slate-800 dark:text-slate-400">
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
          <tbody>
            {entries.map((entry, index) => {
              const key = entryKey(entry, index)
              const playing = playingKey === key
              const editing = editingKey === key && onSaveText != null
              const selected = selectedKeys.has(key)
              const span = Math.max(MIN_FRAGMENT, entry.end - entry.start)
              const percent = playing
                ? Math.min(100, Math.max(0, (position / span) * 100))
                : 0
              const pieces = speakerPieces(speakers, entry)
              return (
                <tr
                  key={key}
                  className={
                    playing
                      ? 'border-t border-slate-100 bg-blue-50 dark:border-slate-800 dark:bg-blue-950/40'
                      : selected
                        ? 'border-t border-slate-100 bg-amber-50 dark:border-slate-800 dark:bg-amber-950/30'
                        : 'border-t border-slate-100 dark:border-slate-800'
                  }
                >
                  {onAssignSpeaker && (
                    <td className="px-2 py-1.5">
                      <input
                        type="checkbox"
                        checked={selected}
                        onChange={() => toggleRow(key)}
                        aria-label={`Выделить реплику ${index + 1}`}
                      />
                    </td>
                  )}
                  <td className="px-2 py-1.5">
                    <button
                      type="button"
                      onClick={() => toggle({ key, start: entry.start, end: entry.end })}
                      aria-label={playing ? 'Остановить фрагмент' : 'Прослушать фрагмент'}
                      aria-pressed={playing}
                      title={playing ? 'Остановить' : `Прослушать ${formatTime(entry.start)}–${formatTime(entry.end)}`}
                      className={
                        playing
                          ? 'flex h-6 w-6 items-center justify-center rounded-full bg-blue-500 text-xs text-white'
                          : 'flex h-6 w-6 items-center justify-center rounded-full border border-slate-300 text-xs text-slate-600 hover:bg-slate-100 dark:border-slate-600 dark:text-slate-300 dark:hover:bg-slate-800'
                      }
                    >
                      <PlayerIcon paused={playing} />
                    </button>
                  </td>
                  <td className="whitespace-nowrap px-3 py-1.5 font-mono text-xs text-slate-500 dark:text-slate-400">
                    {formatTime(entry.start)}
                  </td>
                  <td className="px-3 py-1.5">
                    <div className="flex items-start gap-1.5">
                      <div
                        className="max-w-[9rem] whitespace-normal break-words leading-snug sm:max-w-[16rem]"
                        title={pieces.map((piece) => piece.name).join(' + ')}
                      >
                        {pieces.map((piece, pieceIndex) => (
                          <span key={`${piece.id ?? 'none'}-${pieceIndex}`}>
                            {pieceIndex > 0 && (
                              <span className="mx-1 text-slate-400 dark:text-slate-500">+</span>
                            )}
                            <span
                              className={
                                piece.extra
                                  ? 'text-slate-500 dark:text-slate-400'
                                  : undefined
                              }
                              title={piece.extra ? 'дополнительный говорящий (наложение)' : undefined}
                            >
                              {piece.name}
                            </span>
                            {piece.extra && piece.id && onAddExtraSpeaker && (
                              <button
                                type="button"
                                onClick={() => void removeExtra(entry, piece.id as string)}
                                title="Убрать второго говорящего"
                                className="ml-1 rounded border border-slate-300 px-1 text-[10px] text-slate-500 hover:bg-slate-100 dark:border-slate-600 dark:text-slate-300 dark:hover:bg-slate-800"
                              >
                                ×
                              </button>
                            )}
                          </span>
                        ))}
                      </div>
                      {onSplitEntry && (
                        <button
                          type="button"
                          onClick={() => beginSplit(entry, index)}
                          title="Разделить реплику по времени на двух говорящих"
                          className="mt-0.5 shrink-0 rounded border border-slate-300 px-1.5 py-0.5 text-[10px] text-slate-500 hover:bg-slate-100 dark:border-slate-600 dark:text-slate-300 dark:hover:bg-slate-800"
                        >
                          разделить
                        </button>
                      )}
                    </div>
                  </td>
                  <td className="whitespace-nowrap px-3 py-1.5 text-base">
                    {entry.low_confidence && <span title="низкая уверенность">⚠</span>}
                    {entry.low_speaker_confidence && (
                      <span title="говорящий под вопросом">?</span>
                    )}
                    {entry.overlap && <span title="наложение речи">⇄</span>}
                  </td>
                  <td
                    className="px-3 py-1.5"
                    onContextMenu={(event) => openContextMenu(event)}
                  >
                    {editing ? (
                      <div className="space-y-1">
                        <textarea
                          value={draftText}
                          onChange={(event) => setDraftText(event.target.value)}
                          rows={2}
                          autoFocus
                          className="w-full rounded-md border border-slate-300 px-2 py-1 text-sm focus:border-blue-400 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
                        />
                        <div className="flex items-center gap-2">
                          <button
                            type="button"
                            onClick={() => void saveEdit(entry, key)}
                            disabled={savingKey === key}
                            className="rounded-md bg-slate-800 px-2.5 py-1 text-xs text-white hover:bg-slate-700 disabled:opacity-40 dark:bg-slate-200 dark:text-slate-900 dark:hover:bg-white"
                          >
                            {savingKey === key ? 'Сохранение…' : 'Сохранить'}
                          </button>
                          <button
                            type="button"
                            onClick={() => {
                              setEditingKey(null)
                              setEditError(null)
                            }}
                            className="rounded-md border border-slate-300 px-2.5 py-1 text-xs hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800"
                          >
                            Отмена
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div className="flex items-start gap-1.5">
                        <div
                          className="break-words"
                          onDoubleClick={() => beginEdit(key, entry.text)}
                        >
                          {entry.text}
                        </div>
                        {onSaveText && (
                          <button
                            type="button"
                            onClick={() => beginEdit(key, entry.text)}
                            aria-label="Править текст реплики"
                            title="Править текст"
                            className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded border border-slate-300 text-slate-500 hover:bg-slate-100 dark:border-slate-600 dark:text-slate-300 dark:hover:bg-slate-800"
                          >
                            <PencilIcon />
                          </button>
                        )}
                        {entry.edited && (
                          <span
                            className="mt-0.5 shrink-0 rounded bg-amber-100 px-1 py-0.5 text-[10px] font-medium text-amber-700 dark:bg-amber-950/60 dark:text-amber-300"
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
                          <button
                            type="button"
                            onClick={() => void resetEdit(entry)}
                            title="Сбросить к исходному тексту"
                            className="mt-0.5 shrink-0 rounded border border-slate-300 px-1.5 py-0.5 text-[10px] text-slate-500 hover:bg-slate-100 dark:border-slate-600 dark:text-slate-300 dark:hover:bg-slate-800"
                          >
                            сбросить
                          </button>
                        )}
                      </div>
                    )}
                    {playing && (
                      <div className="mt-1 h-1 w-full overflow-hidden rounded-full bg-blue-100 dark:bg-blue-900">
                        <div
                          className="h-full rounded-full bg-blue-500 transition-[width] duration-100 ease-linear"
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
          <p className="py-6 text-center text-sm text-slate-400 dark:text-slate-500">
            Ничего не найдено
          </p>
        )}
      </div>

      {contextMenu && (
        <div
          role="menu"
          className="fixed z-50 min-w-[12rem] rounded-md border border-slate-200 bg-white py-1 text-sm shadow-lg dark:border-slate-700 dark:bg-slate-900"
          style={{ left: contextMenu.x, top: contextMenu.y }}
          onClick={(event) => event.stopPropagation()}
        >
          <button
            type="button"
            role="menuitem"
            onClick={() => {
              setQuickTerm(contextMenu.term)
              setContextMenu(null)
            }}
            className="block w-full px-3 py-1.5 text-left hover:bg-slate-100 dark:hover:bg-slate-800"
          >
            Добавить в глоссарий: «{contextMenu.term}»
          </button>
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
