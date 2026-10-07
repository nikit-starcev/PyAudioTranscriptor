import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import {
  api,
  errorMessage,
  formatBytes,
  type AssetInfo,
  type AssetsResponse,
  type BinaryRequirement,
  type DependencyEvent,
} from '../api'

type Props = {
  /** Требования шага «Бинарники/Пакеты» из плана мастера. */
  requirements: BinaryRequirement[]
  /** Вызывается после успешной установки — чтобы пересобрать план мастера. */
  onChanged?: () => void
}

/**
 * Список внешних компонентов мастера (#98). Бинарники (whisper-cli,
 * llama-server, deep-filter) скачиваются кнопкой «Скачать» — по allowlist
 * фиксированных URL с проверкой sha256; pip-пакеты (#66) ставятся кнопкой
 * «Установить». Прогресс и статусы приходят по SSE ``/api/assets/events``.
 */
function BinaryRequirements({ requirements, onChanged }: Props) {
  const [assets, setAssets] = useState<Record<string, AssetInfo>>({})
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
      const next = await api<AssetsResponse>('/api/assets')
      setAssets(Object.fromEntries(next.assets.map((asset) => [asset.key, asset])))
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
    const source = new EventSource('/api/assets/events')
    source.onmessage = (message) => {
      const event = JSON.parse(message.data) as DependencyEvent
      const seq = event.seq
      if (typeof seq === 'number') {
        if (seq <= lastSeenSeq.current) return
        lastSeenSeq.current = seq
      }
      setAssets((current) => {
        const existing = current[event.key]
        if (!existing) return current
        return {
          ...current,
          [event.key]: {
            ...existing,
            status: event.status,
            message: event.message,
            error: event.error,
            bytes_done: event.bytes_done ?? existing.bytes_done,
            total: event.total ?? existing.total,
            fraction: event.fraction ?? existing.fraction,
            path: event.path || existing.path,
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

  const install = async (asset: AssetInfo) => {
    setBusy(asset.key)
    setError(null)
    try {
      await api(`/api/assets/${asset.key}/install`, { method: 'POST' })
      setAssets((current) => ({
        ...current,
        [asset.key]: { ...asset, status: 'running', message: 'Запуск…', error: null },
      }))
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setBusy(null)
    }
  }

  const canInstallPip = installer !== null
  const summary = useMemo(
    () =>
      requirements.map((requirement) => ({
        requirement,
        assetKey: requirement.asset_key ?? requirement.dep_key ?? requirement.key,
      })),
    [requirements],
  )

  return (
    <div className="space-y-3">
      <p className="text-sm text-slate-600 dark:text-slate-300">
        Внешние бинарники скачиваются кнопкой «Скачать» (фиксированные версии с проверкой
        контрольной суммы), пакеты ставятся кнопкой «Установить»
        {installer ? ` через ${installer}` : ''}. Прогресс виден здесь же.
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
          {summary.map(({ requirement, assetKey }) => {
            const asset = assets[assetKey]
            const isBinary = (asset?.kind ?? requirement.kind) === 'binary'
            const installed = asset?.installed ?? requirement.available
            const running = asset?.status === 'running'
            const downloadable = asset?.downloadable ?? requirement.downloadable ?? false
            const size = asset?.artifact?.size ?? requirement.artifact?.size ?? 0
            const fraction = running && asset?.fraction != null ? Math.round(asset.fraction * 100) : null
            const installedPath = asset?.path || requirement.installed_path || ''
            const canAct = asset != null && (isBinary ? downloadable : canInstallPip)
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
                  {requirement.optional && !installed && (
                    <span className="ml-2 text-xs text-slate-400 dark:text-slate-500">необязательно</span>
                  )}
                </p>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  {requirement.instructions}
                </p>
                {size > 0 && (
                  <p className="text-xs text-slate-400 dark:text-slate-500">
                    Размер: ~{formatBytes(size)}
                    {requirement.platform ? ` · ${requirement.platform}` : ''}
                    {asset?.artifact?.variant ? ` · ${asset.artifact.variant}` : ''}
                  </p>
                )}
                {running && (
                  <div className="mt-1 space-y-1">
                    <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
                      <div
                        className={`h-full rounded-full bg-blue-500 ${fraction == null ? 'w-1/3 animate-pulse' : ''}`}
                        style={fraction == null ? undefined : { width: `${fraction}%` }}
                      />
                    </div>
                    {asset?.message && (
                      <p className="truncate font-mono text-[11px] text-slate-500 dark:text-slate-400">
                        {asset.message}
                      </p>
                    )}
                  </div>
                )}
                {asset?.error && !running && (
                  <p className="mt-0.5 text-xs text-red-600 dark:text-red-400">{asset.error}</p>
                )}
                {installedPath && installed && (
                  <p className="mt-0.5 truncate font-mono text-[11px] text-slate-400 dark:text-slate-500" title={installedPath}>
                    {installedPath}
                  </p>
                )}
                <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
                  {!installed && (
                    <button
                      type="button"
                      onClick={() => asset && void install(asset)}
                      disabled={busy !== null || running || !canAct}
                      className="rounded-md bg-slate-800 px-3 py-1 text-xs text-white hover:bg-slate-700 disabled:opacity-40 dark:bg-slate-200 dark:text-slate-900 dark:hover:bg-white"
                    >
                      {running ? 'Установка…' : isBinary ? 'Скачать' : 'Установить'}
                    </button>
                  )}
                  {!installed && isBinary && !downloadable && (
                    <span className="text-amber-700 dark:text-amber-400">
                      Для вашей ОС/архитектуры готового артефакта нет — нужна ручная сборка
                    </span>
                  )}
                  {!installed && !isBinary && !canInstallPip && (
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
