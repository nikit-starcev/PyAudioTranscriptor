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

type Props = {
  // Завершённые стадии (с сервера), в порядке выполнения.
  times: StageTime[]
  // Идёт ли обработка прямо сейчас.
  running: boolean
  // Момент старта задачи (мс epoch) — для живого счётчика общего времени.
  totalStartedAt: number | null
  // Итоговое время обработки (когда задача завершена).
  finalTotalSeconds: number | null
  // Текущая стадия и момент её старта (мс epoch) — для живого счётчика.
  currentStage: string | null
  currentStartedAt: number | null
}

/**
 * Блок «Стадии и время»: длительности этапов, пометка «из кэша» и итоговое
 * время обработки. Во время прогона ведёт локальный секундомер по таймеру.
 */
export default function StageTimes({
  times,
  running,
  totalStartedAt,
  finalTotalSeconds,
  currentStage,
  currentStartedAt,
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

  // Текущая стадия показывается отдельной строкой; из списка завершённых её
  // исключаем на случай совпадения ключа. Служебные стадии (``queued``) не
  // показываем — для них есть статус в блоке прогресса.
  const showCurrent = currentStage != null && currentStage in STAGE_LABELS
  const completed = currentStage ? times.filter((item) => item.stage !== currentStage) : times
  const hasData = completed.length > 0 || showCurrent || totalSeconds != null

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
          {completed.map((item) => (
            <li key={item.stage} className="flex items-center gap-2">
              <span className="text-emerald-600 dark:text-emerald-400">✓</span>
              <span className="flex-1 truncate">{stageLabel(item.stage)}</span>
              {item.cached && <CachedBadge />}
              <span className="tabular-nums text-slate-600 dark:text-slate-300">
                {formatStageTime(item.seconds)}
              </span>
            </li>
          ))}
          {showCurrent && currentStage && (
            <li className="flex items-center gap-2">
              <span className="animate-pulse text-blue-600 dark:text-blue-400">●</span>
              <span className="flex-1 truncate">{stageLabel(currentStage)}</span>
              <span className="tabular-nums text-blue-600 dark:text-blue-400">
                {running && liveSeconds != null ? `${formatStageTime(liveSeconds)}…` : '—'}
              </span>
            </li>
          )}
        </ul>
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
