import { useEffect, useState } from 'react'
import { Ban, Circle, Check } from 'lucide-react'

import { formatStageTime, type StageTime } from '../api'
import { Badge, Card, cn } from './ui'

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
    <Card>
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-4 py-3">
        <h2 className="text-sm font-semibold text-text">Стадии и время</h2>
        <span className="text-sm tabular-nums text-text">
          Итого:{' '}
          <span className="font-medium">
            {totalSeconds != null ? formatStageTime(totalSeconds) : '—'}
          </span>
        </span>
      </div>
      <div className="p-4">
        {!hasData ? (
          <p className="py-1 text-xs text-muted">Нет данных о времени</p>
        ) : (
          <ul className="space-y-1 text-sm">
            {stages.map((stage) => {
              const timing = byStage.get(stage)
              const isFailed = failedStage === stage
              const isActive = running && currentStage === stage
              const isDone = !isFailed && !isActive && timing != null
              return (
                <li
                  key={stage}
                  className={cn(
                    'flex items-center gap-2 rounded px-1',
                    isActive && 'bg-info-soft',
                    isFailed && 'bg-danger-soft',
                  )}
                >
                  <span
                    className={cn(
                      'shrink-0',
                      isFailed
                        ? 'text-danger'
                        : isDone
                          ? 'text-success'
                          : isActive
                            ? 'text-primary'
                            : 'text-muted',
                    )}
                    title={isFailed ? `Сбой на стадии «${stageLabel(stage)}»` : stageLabel(stage)}
                  >
                    {isFailed ? (
                      <Ban aria-hidden className="h-3.5 w-3.5" />
                    ) : isDone ? (
                      <Check aria-hidden className="h-3.5 w-3.5" />
                    ) : isActive ? (
                      <Circle aria-hidden className="h-2 w-2 animate-pulse fill-current" />
                    ) : (
                      <Circle aria-hidden className="h-2 w-2" />
                    )}
                  </span>
                  <span
                    className={cn(
                      'min-w-0 flex-1 truncate',
                      isFailed
                        ? 'font-medium text-danger-soft-fg'
                        : isActive
                          ? 'font-medium text-text'
                          : isDone
                            ? 'text-text'
                            : 'text-muted',
                    )}
                  >
                    {stageLabel(stage)}
                  </span>
                  {timing?.cached && (
                    <span className="shrink-0" title="Стадия не пересчитывалась — результат взят из кэша">
                      <Badge tone="warn">из кэша</Badge>
                    </span>
                  )}
                  <span
                    className={cn(
                      'tabular-nums',
                      isFailed ? 'text-danger' : isActive ? 'text-primary' : 'text-muted',
                      isDone && 'text-text',
                    )}
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
          <p className="mt-2 text-xs text-danger">Задача завершилась ошибкой</p>
        )}
      </div>
    </Card>
  )
}
