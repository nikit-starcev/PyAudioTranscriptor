import { useCallback, useEffect, useRef, useState } from 'react'

import {
  api,
  errorMessage,
  type BinaryRequirement,
  type DependencyEvent,
  type DependencyInfo,
  type DepsResponse,
} from '../api'

type Props = {
  /** Требования шага «Бинарники/Пакеты» из плана мастера. */
  requirements: BinaryRequirement[]
  /** Вызывается после успешной установки — чтобы пересобрать план мастера. */
  onChanged?: () => void
}

/**
 * Список внешних компонентов мастера. Бинарники (whisper.cpp/llama.cpp)
 * собираются вручную и показываются подсказкой; опциональные пакеты из
 * allowlist (#66) ставятся кнопкой «Установить» с прогрессом по SSE
 * ``/api/deps/events``.
 */
function BinaryRequirements({ requirements, onChanged }: Props) {
  const [info, setInfo] = useState<Record<string, DependencyInfo>>({})
  const [installer, setInstaller] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState<string | null>(null)

  // Колбэк в ref: его идентичность не должна пересоздавать EventSource (#65).
  const onChangedRef = useRef(onChanged)
  useEffect(() => {
    onChangedRef.current = onChanged
  }, [onChanged])

  // Номер последнего обработанного SSE-события: историю при переподключении
  // не применяем повторно.
  const lastSeenSeq = useRef(-1)

  const refresh = useCallback(async () => {
    try {
      const next = await api<DepsResponse>('/api/deps')
      setInfo(Object.fromEntries(next.deps.map((dep) => [dep.key, dep])))
      setInstaller(next.installer)
      setError(null)
    } catch (cause) {
      setError(errorMessage(cause))
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  // Соединение одно на время показа шага; зависимость только от стабильного
  // `refresh`. Иначе inline-колбэк родителя пересоздавал бы EventSource (#65).
  useEffect(() => {
    const source = new EventSource('/api/deps/events')
    source.onmessage = (message) => {
      const event = JSON.parse(message.data) as DependencyEvent
      const seq = event.seq
      if (typeof seq === 'number') {
        if (seq <= lastSeenSeq.current) return
        lastSeenSeq.current = seq
      }
      setInfo((current) => {
        const existing = current[event.key]
        if (!existing) return current
        return {
          ...current,
          [event.key]: {
            ...existing,
            status: event.status,
            message: event.message,
            error: event.error,
          },
        }
      })
      if (event.status === 'done' || event.status === 'error') {
        void refresh()
        if (event.status === 'done') onChangedRef.current?.()
      }
    }
    source.onerror = () => {
      // Соединение переподключится само; ошибку не показываем.
    }
    return () => source.close()
  }, [refresh])

  const install = async (key: string) => {
    setBusy(key)
    setError(null)
    try {
      await api(`/api/deps/${key}/install`, { method: 'POST' })
      setInfo((current) => {
        const existing = current[key]
        if (!existing) return current
        return { ...current, [key]: { ...existing, status: 'running', message: 'Установка…', error: null } }
      })
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setBusy(null)
    }
  }

  return (
    <div className="space-y-3">
      <p className="text-sm text-slate-600 dark:text-slate-300">
        Бинарники whisper.cpp/llama.cpp собираются под ОС и GPU вручную (пути — в
        настройках). Опциональные пакеты ставятся кнопкой «Установить»
        {installer ? ` через ${installer}` : ''} — прогресс виден здесь же.
      </p>
      {error && (
        <p className="rounded-md bg-red-50 px-3 py-1.5 text-xs text-red-700 dark:bg-red-950/50 dark:text-red-300">
          {error}
        </p>
      )}
      {requirements.length === 0 ? (
        <p className="rounded-md border border-emerald-200 bg-emerald-50 px-3 py-2 text-sm text-emerald-700 dark:border-emerald-900 dark:bg-emerald-950/40 dark:text-emerald-300">
          Для выбранного режима отдельные компоненты не требуются.
        </p>
      ) : (
        <ul className="space-y-2">
          {requirements.map((requirement) => {
            const dep = requirement.dep_key ? info[requirement.dep_key] : undefined
            const installed = dep?.installed ?? requirement.available
            const running = dep?.status === 'running'
            const installable = dep?.installable ?? requirement.installable ?? false
            return (
              <li
                key={requirement.key}
                className="rounded-md border border-slate-200 p-3 text-sm dark:border-slate-800"
              >
                <p className="font-medium">
                  {requirement.label}{' '}
                  <span
                    className={
                      installed
                        ? 'text-emerald-600 dark:text-emerald-400'
                        : running
                          ? 'text-blue-600 dark:text-blue-400'
                          : 'text-amber-600 dark:text-amber-400'
                    }
                  >
                    {installed ? '✓ установлено' : running ? '⏳ установка…' : '✗ не найдено'}
                  </span>
                </p>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  {requirement.instructions}
                </p>
                {running && (
                  <div className="mt-1 space-y-1">
                    <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
                      <div className="h-full w-1/3 animate-pulse rounded-full bg-blue-500" />
                    </div>
                    {dep?.message && (
                      <p className="truncate font-mono text-[11px] text-slate-500 dark:text-slate-400">
                        {dep.message}
                      </p>
                    )}
                  </div>
                )}
                {dep?.error && !running && (
                  <p className="mt-0.5 text-xs text-red-600 dark:text-red-400">{dep.error}</p>
                )}
                <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
                  {requirement.dep_key && !installed && (
                    <button
                      type="button"
                      onClick={() => void install(requirement.dep_key as string)}
                      disabled={busy !== null || running || !installable}
                      className="rounded-md bg-slate-800 px-3 py-1 text-xs text-white hover:bg-slate-700 disabled:opacity-40 dark:bg-slate-200 dark:text-slate-900 dark:hover:bg-white"
                    >
                      {running ? 'Установка…' : 'Установить'}
                    </button>
                  )}
                  {requirement.dep_key && !installable && (
                    <span className="text-amber-700 dark:text-amber-400">
                      Не найден установщик (uv/pip)
                    </span>
                  )}
                  {requirement.links.map((link) => (
                    <a
                      key={link}
                      href={link}
                      target="_blank"
                      rel="noreferrer noopener"
                      className="text-blue-600 underline hover:text-blue-500 dark:text-blue-400 dark:hover:text-blue-300"
                    >
                      {link}
                    </a>
                  ))}
                </div>
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}

export default BinaryRequirements
