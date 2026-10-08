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
import { Alert, Badge, Button, ProgressBar } from './ui'

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
      <p className="text-sm text-muted">
        Внешние бинарники скачиваются кнопкой «Скачать» (фиксированные версии с проверкой
        контрольной суммы), пакеты ставятся кнопкой «Установить»
        {installer ? ` через ${installer}` : ''}. Прогресс виден здесь же.
      </p>
      {error && (
        <Alert tone="danger" live>
          {error}
        </Alert>
      )}
      {requirements.length === 0 ? (
        <Alert tone="success">Для выбранного режима отдельные компоненты не требуются.</Alert>
      ) : (
        <ul className="space-y-2">
          {summary.map(({ requirement, assetKey }) => {
            const asset = assets[assetKey]
            const isBinary = (asset?.kind ?? requirement.kind) === 'binary'
            const installed = asset?.installed ?? requirement.available
            const running = asset?.status === 'running'
            const downloadable = asset?.downloadable ?? requirement.downloadable ?? false
            const size = asset?.artifact?.size ?? requirement.artifact?.size ?? 0
            const fraction =
              running && asset?.fraction != null ? Math.round(asset.fraction * 100) : null
            const installedPath = asset?.path || requirement.installed_path || ''
            const canAct = asset != null && (isBinary ? downloadable : canInstallPip)
            return (
              <li key={requirement.key} className="rounded-md border border-border p-3 text-sm">
                <p className="flex flex-wrap items-center gap-2 font-medium">
                  {requirement.label}
                  <Badge tone={installed ? 'success' : running ? 'info' : 'warn'}>
                    {installed ? 'установлено' : running ? 'установка…' : 'не найдено'}
                  </Badge>
                  {requirement.optional && !installed && (
                    <span className="text-xs text-muted">необязательно</span>
                  )}
                </p>
                <p className="text-xs text-muted">{requirement.instructions}</p>
                {size > 0 && (
                  <p className="text-xs text-muted">
                    Размер: ~{formatBytes(size)}
                    {requirement.platform ? ` · ${requirement.platform}` : ''}
                    {asset?.artifact?.variant ? ` · ${asset.artifact.variant}` : ''}
                  </p>
                )}
                {running && (
                  <div className="mt-1 space-y-1">
                    <ProgressBar
                      size="sm"
                      value={fraction ?? 0}
                      indeterminate={fraction == null}
                      label={`Загрузка: ${requirement.label}`}
                    />
                    {asset?.message && (
                      <p className="truncate font-mono text-[11px] text-muted">{asset.message}</p>
                    )}
                  </div>
                )}
                {asset?.error && !running && <p className="mt-0.5 text-xs text-danger">{asset.error}</p>}
                {installedPath && installed && (
                  <p className="mt-0.5 truncate font-mono text-[11px] text-muted" title={installedPath}>
                    {installedPath}
                  </p>
                )}
                <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
                  {!installed && (
                    <Button
                      variant="primary"
                      size="sm"
                      loading={busy === assetKey || running}
                      disabled={busy !== null || !canAct}
                      onClick={() => asset && void install(asset)}
                    >
                      {running ? 'Установка…' : isBinary ? 'Скачать' : 'Установить'}
                    </Button>
                  )}
                  {!installed && isBinary && !downloadable && (
                    <span className="text-warn">
                      Для вашей ОС/архитектуры готового артефакта нет — нужна ручная сборка
                    </span>
                  )}
                  {!installed && !isBinary && !canInstallPip && (
                    <span className="text-warn">Не найден установщик (uv/pip)</span>
                  )}
                  {requirement.links.map((link) => (
                    <a
                      key={link}
                      href={link}
                      target="_blank"
                      rel="noreferrer noopener"
                      className="font-medium text-primary underline hover:opacity-80"
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
