// Прогресс длительных действий веб-интерфейса (#58).
//
// Действия «Применить имена» (enrollment), «Применить глоссарий», «Проверить
// текст» и «Сформировать протокол» выполняются одним POST-запросом, который
// может идти десятки секунд. Чтобы UI не выглядел «зависшим», клиент генерирует
// `action_id`, открывает отдельный SSE-канал `/api/actions/{id}/events` и
// передаёт тот же id заголовком `X-Action-Id`. Сервер публикует этапы, а хук
// `useActionProgress` показывает их и ведёт локальный секундомер.

import { useCallback, useEffect, useRef, useState } from 'react'

import { errorMessage } from './api'

export type ActionKind = 'enrollment' | 'glossary' | 'correction' | 'protocol'

export type ActionStatus = 'running' | 'done' | 'error'

/** Одно событие этапа от сервера (см. web/actions.py). */
export type ActionStageEvent = {
  action: string
  stage: string
  message: string
  fraction: number | null
  elapsed: number
  status: ActionStatus
}

/** Текущее состояние выполняемого/завершённого действия для UI. */
export type ActionRun = {
  kind: ActionKind
  actionId: string
  /** Локальная метка старта — основа секундомера (не зависит от часов сервера). */
  startedAt: number
  /** Метка завершения (для остановки секундомера) или null, пока идёт. */
  finishedAt: number | null
  stage: string
  message: string
  fraction: number | null
  status: ActionStatus
  /** Итоговый текст (успех/ошибка) после завершения. */
  result: string | null
}

/** Человекочитаемые названия действий. */
export const ACTION_TITLES: Record<ActionKind, string> = {
  enrollment: 'Переопределение говорящих',
  glossary: 'Применение глоссария',
  correction: 'Проверка текста',
  protocol: 'Формирование протокола',
}

/** Этапы действий — должны совпадать со stage-идентификаторами сервера. */
export const ACTION_STEPS: Record<ActionKind, { id: string; label: string }[]> = {
  enrollment: [
    { id: 'samples', label: 'Загрузка образцов и аудио' },
    { id: 'embeddings', label: 'Расчёт эмбеддингов голосов' },
    { id: 'matching', label: 'Сопоставление говорящих и имён' },
    { id: 'apply', label: 'Применение имён' },
  ],
  glossary: [
    { id: 'prepare', label: 'Подготовка глоссария' },
    { id: 'process', label: 'Обработка реплик' },
  ],
  correction: [
    { id: 'analyze', label: 'Поиск правок' },
    { id: 'apply', label: 'Применение правок' },
  ],
  protocol: [
    { id: 'prepare', label: 'Подготовка стенограммы' },
    { id: 'llm', label: 'Резюме встречи (LLM)' },
    { id: 'export', label: 'Экспорт документов' },
  ],
}

/** Результат-описание завершённого действия для карточки прогресса. */
export type ActionDescription = { text: string; error?: boolean }

/** Генерирует уникальный идентификатор действия (UUID или запасной вариант). */
export function newActionId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }
  const random = Math.random().toString(36).slice(2)
  return `action-${Date.now().toString(36)}-${random}`
}

/**
 * Хук прогресса действий: запускает задачу с SSE-каналом этапов, хранит
 * состояние для карточки и завершает его по результату/ошибке задачи.
 *
 * `runTask` создаёт `action_id`, открывает поток, выполняет `task(actionId)` и
 * по её завершении (успех/исключение) фиксирует итог. Статус с сервера
 * (`running`/`done`/`error`) используется только для этапов; финальный статус
 * определяет сама задача — так корректно обрабатываются «мягкие» ошибки,
 * которые сервер отдаёт с кодом 200 и полем `error`.
 */
export function useActionProgress() {
  const [run, setRun] = useState<ActionRun | null>(null)
  const sourceRef = useRef<EventSource | null>(null)

  const closeSource = useCallback(() => {
    sourceRef.current?.close()
    sourceRef.current = null
  }, [])

  useEffect(() => closeSource, [closeSource])

  const start = useCallback(
    (kind: ActionKind, actionId: string) => {
      closeSource()
      setRun({
        kind,
        actionId,
        startedAt: Date.now(),
        finishedAt: null,
        stage: '',
        message: 'Запуск…',
        fraction: null,
        status: 'running',
        result: null,
      })
      if (typeof EventSource === 'undefined') return
      const stream = new EventSource(
        `/api/actions/${encodeURIComponent(actionId)}/events`,
      )
      sourceRef.current = stream
      stream.onmessage = (message) => {
        let event: ActionStageEvent
        try {
          event = JSON.parse(message.data) as ActionStageEvent
        } catch {
          return
        }
        setRun((current) => {
          if (!current || current.actionId !== actionId) return current
          return {
            ...current,
            stage: event.stage ?? current.stage,
            message: event.message || current.message,
            fraction: event.fraction ?? null,
          }
        })
        // Конечное событие закрывает серверный поток — не даём EventSource
        // переподключаться; финальный статус зафиксирует сама задача.
        if (event.status !== 'running') closeSource()
      }
    },
    [closeSource],
  )

  const finish = useCallback(
    (actionId: string, status: ActionStatus, result: string | null) => {
      closeSource()
      setRun((current) =>
        current && current.actionId === actionId
          ? { ...current, status, result, finishedAt: Date.now() }
          : current,
      )
    },
    [closeSource],
  )

  const reset = useCallback(() => {
    closeSource()
    setRun(null)
  }, [closeSource])

  const runTask = useCallback(
    async <T,>(
      kind: ActionKind,
      task: (actionId: string) => Promise<T>,
      describe: (value: T) => ActionDescription,
    ): Promise<T> => {
      const actionId = newActionId()
      start(kind, actionId)
      try {
        const value = await task(actionId)
        const description = describe(value)
        finish(actionId, description.error ? 'error' : 'done', description.text)
        return value
      } catch (cause) {
        finish(actionId, 'error', errorMessage(cause))
        throw cause
      }
    },
    [start, finish],
  )

  return { run, runTask, reset }
}
