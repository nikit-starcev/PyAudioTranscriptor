import { useEffect, useState } from 'react'

import { formatStageTime, type StageTime } from '../api'

const STAGE_LABELS: Record<string, string> = {
  denoise: 'Шумоподавление',
  asr: 'Распознавание речи',
  diarization: 'Определение говорящих',
  merge: 'Объединение сегментов',
  clean: 'Очистка артефактов',
  correction: 'Автоисправление',
  llm: 'LLM-постобработка',
  export: 'Экспорт',
}

export function stageLabel(stage: string): string {
  return STAGE_LABELS[stage] ?? stage
}

/** Ключи известных стадий в порядке выполнения (запасной список). */
export const STAGE_KEYS: string[] = Object.keys(STAGE_LABELS)

type Props = {
  // Завершённые стадии (с сервера), в порядке выполнения.
  times: StageTime[]
  // Планируемые стадии в порядке выполнения (с сервера; пусто — запасной список).
  plannedStages: string[]
  // Идёт ли обработка прямо сейчас.
  running: boolean
  // Статус задачи (``running``/``done``/``error``/``cancelled``/``queued``).
  status: string
  // Момент старта задачи (мс epoch) — для живого счётчика общего времени.
  totalStartedAt: number | null
  // Итоговое время обработки (когда задача завершена).
  finalTotalSeconds: number | null
  // Текущая стадия и момент её старта (мс epoch) — для живого счётчика.
  currentStage: string | null
  currentStartedAt: number | null
  // Стадия, на которой произошёл сбой (при статусе ``error``).
  failedStage: string | null
}

/**
 * Блок «Стадии и время»: единственное место, где показан перечень стадий.
 *
 * Выводит сразу **весь** план (в порядке выполнения): пройденные — с галочкой и
 * временем, активная — выделена с живым секундомером, ожидающие — приглушены,
 * упавшая — со значком ошибки. План приходит с сервера (``planned_stages``).
 */
export default function StageTimes({
  times,
  plannedStages,
  running,
  status,
  totalStartedAt,
  finalTotalSeconds,
  currentStage,
  currentStartedAt,
  failedStage,
}: Props) {
  const [now, setNow] = useState(() => Date.now())

  useEffect(() => {
    if (!running) return
    const id = window.setInterval(() => setNow(Date.now()), 250)
    return () => window.clearInterval(id)
  }, [running])

  const liveSeconds =
    currentStartedAt != null ? Math.max((now - currentStartedAt) / 1000, 0) : null
  const totalSeconds =
    running && totalStartedAt != null
      ? Math.max((now - totalStartedAt) / 1000, 0)
      : finalTotalSeconds

  const byStage = new Map(times.map((item) => [item.stage, item]))
  // Упавшую стадию всегда показываем, даже если её нет в плане (защита от
  // рассинхрона: например, конфигурация изменилась после старта).
  const stages =
    failedStage && !plannedStages.includes(failedStage)
      ? [...plannedStages, failedStage]
      : plannedStages
  const hasData = stages.length > 0 || totalSeconds != null

  return (
    <div className="mt-4 rounded-md border border-slate-200 bg-slate-50 p-3 dark:border-slate-800 dark:bg-slate-800/40">
      <div className="mb-2 flex items-center justify-between gap-3">
        <h3 className="text-xs font-medium uppercase tracking-wide text-slate-500 dark:text-slate-400">
          Стадии и время
        </h3>
        <span className="text-sm tabular-nums text-slate-700 dark:text-slate-200">
          Итого:{' '}
          <span className="font-medium">
            {totalSeconds != null ? formatStageTime(totalSeconds) : '—'}
          </span>
        </span>
      </div>

      {!hasData ? (
        <p className="py-1 text-xs text-slate-400 dark:text-slate-500">Нет данных о времени</p>
      ) : (
        <ul className="space-y-1 text-sm">
          {stages.map((stage) => {
            const timing = byStage.get(stage)
            const isFailed = failedStage === stage
            const isActive = running && currentStage === stage
            const isDone = !isFailed && !isActive && timing != null
            const title = isFailed
              ? `Сбой на стадии «${stageLabel(stage)}»`
              : stageLabel(stage)
            return (
              <li
                key={stage}
                className={`flex items-center gap-2 rounded px-1 ${
                  isActive
                    ? 'bg-blue-50 dark:bg-blue-950/40'
                    : isFailed
                      ? 'bg-red-50 dark:bg-red-950/40'
                      : ''
                }`}
              >
                <span
                  className={
                    isFailed
                      ? 'text-red-600 dark:text-red-400'
                      : isDone
                        ? 'text-emerald-600 dark:text-emerald-400'
                        : isActive
                          ? 'animate-pulse text-blue-600 dark:text-blue-400'
                          : 'text-slate-300 dark:text-slate-600'
                  }
                  title={title}
                >
                  {isFailed ? '⛔' : isDone ? '✓' : isActive ? '●' : '·'}
                </span>
                <span
                  className={`flex-1 truncate ${
                    isFailed
                      ? 'font-medium text-red-700 dark:text-red-300'
                      : isActive
                        ? 'font-medium text-blue-700 dark:text-blue-300'
                        : isDone
                          ? ''
                          : 'text-slate-400 dark:text-slate-500'
                  }`}
                >
                  {stageLabel(stage)}
                </span>
                {timing?.cached && <CachedBadge />}
                <span
                  className={`tabular-nums ${
                    isFailed
                      ? 'text-red-600 dark:text-red-400'
                      : isActive
                        ? 'text-blue-600 dark:text-blue-400'
                        : isDone
                          ? 'text-slate-600 dark:text-slate-300'
                          : 'text-slate-400 dark:text-slate-500'
                  }`}
                >
                  {isFailed
                    ? 'сбой'
                    : isActive
                      ? running && liveSeconds != null
                        ? `${formatStageTime(liveSeconds)}…`
                        : '—'
                      : timing
                        ? formatStageTime(timing.seconds)
                        : '—'}
                </span>
              </li>
            )
          })}
        </ul>
      )}

      {status === 'error' && failedStage == null && (
        <p className="mt-2 text-xs text-red-600 dark:text-red-400">
          Задача завершилась ошибкой
        </p>
      )}
    </div>
  )
}

function CachedBadge() {
  return (
    <span
      className="rounded-full bg-amber-100 px-2 py-0.5 text-xs text-amber-700 dark:bg-amber-950/60 dark:text-amber-300"
      title="Стадия не пересчитывалась — результат взят из кэша"
    >
      из кэша
    </span>
  )
}
