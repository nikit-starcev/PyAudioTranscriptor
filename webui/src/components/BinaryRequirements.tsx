import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import {
  api,
  errorMessage,
  formatBytes,
  type AssetDeleteAllResponse,
  type AssetInfo,
  type AssetInstallAllResponse,
  type AssetsResponse,
  type BinaryRequirement,
  type DependencyEvent,
} from '../api'
import { Alert, Badge, Button, Checkbox, ProgressBar, Spinner } from './ui'

type Props = {
  /**
   * Требования шага «Бинарники/Пакеты» из плана мастера. Если не передан —
   * компонент работает автономно (страница «Бинарные пакеты», #105): список
   * строится из единого реестра ``/api/assets``.
   */
  requirements?: BinaryRequirement[]
  /** Вызывается после успешной установки — чтобы пересобрать план мастера. */
  onChanged?: () => void
}

/** Строит запись требования из ресурса реестра (автономный режим, #105). */
function requirementFromAsset(asset: AssetInfo): BinaryRequirement {
  return {
    key: asset.key,
    label: asset.label,
    needed: !asset.optional,
    available: asset.installed,
    status: asset.installed ? 'ok' : 'fail',
    instructions: asset.note,
    links: [],
    asset_key: asset.key,
    kind: asset.kind,
    downloadable: asset.downloadable,
    platform: asset.platform,
    artifact: asset.artifact,
    setting_key: asset.settings_field,
    installed_path: asset.path,
    optional: asset.optional,
  }
}

/**
 * Список внешних компонентов мастера (#98) и страницы «Бинарные пакеты»
 * (#105). Бинарники (whisper-cli, llama-server, deep-filter) скачиваются
 * кнопкой «Скачать» — по allowlist фиксированных URL с проверкой sha256;
 * pip-пакеты (#66) ставятся кнопкой «Установить». Прогресс и статусы приходят
 * по SSE ``/api/assets/events``.
 *
 * Массовые операции (#115): «Скачать все» / «Установить все» ставят все
 * отсутствующие ресурсы, «Удалить все» (или выбранные чекбоксами) удаляет
 * установленные бинарники. Установка идёт последовательно на сервере, прогресс
 * по каждому компоненту — тот же SSE-поток.
 */
function BinaryRequirements({ requirements, onChanged }: Props) {
  const [assets, setAssets] = useState<Record<string, AssetInfo>>({})
  const [loaded, setLoaded] = useState(false)
  const [installer, setInstaller] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [batchBusy, setBatchBusy] = useState(false)
  const [selected, setSelected] = useState<Set<string>>(new Set())

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
      const map = Object.fromEntries(next.assets.map((asset) => [asset.key, asset]))
      setAssets(map)
      setInstaller(next.installer)
      setSelected((current) => {
        const pruned = [...current].filter((key) => map[key]?.installed)
        return pruned.length === current.size ? current : new Set(pruned)
      })
      setError(null)
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setLoaded(true)
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

  const canInstallPip = installer !== null
  // Массовые операции показываем только в автономном разделе «Бинарные пакеты»
  // (в мастере список ограничен его планом).
  const auto = requirements == null
  const assetList = useMemo(() => Object.values(assets), [assets])
  const missingBinaries = useMemo(
    () =>
      assetList.filter(
        (asset) => asset.kind === 'binary' && !asset.installed && asset.downloadable,
      ),
    [assetList],
  )
  const missingPip = useMemo(
    () => assetList.filter((asset) => asset.kind === 'pip' && !asset.installed),
    [assetList],
  )
  const installedBinaries = useMemo(
    () => assetList.filter((asset) => asset.kind === 'binary' && asset.installed),
    [assetList],
  )
  const anyRunning = assetList.some((asset) => asset.status === 'running')
  const actionsDisabled = busy !== null || batchBusy || anyRunning

  const install = async (asset: AssetInfo) => {
    setBusy(asset.key)
    setError(null)
    setNotice(null)
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

  const remove = async (asset: AssetInfo) => {
    if (
      !window.confirm(
        `Удалить установленный ресурс «${asset.label}»? Файлы будут удалены из служебного каталога.`,
      )
    ) {
      return
    }
    setBusy(asset.key)
    setError(null)
    setNotice(null)
    try {
      await api(`/api/assets/${asset.key}`, { method: 'DELETE' })
      setSelected((current) => {
        if (!current.has(asset.key)) return current
        const next = new Set(current)
        next.delete(asset.key)
        return next
      })
      await refresh()
      onChangedRef.current?.()
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setBusy(null)
    }
  }

  const installAll = async (kind: 'binary' | 'pip') => {
    setBatchBusy(true)
    setError(null)
    setNotice(null)
    try {
      const result = await api<AssetInstallAllResponse>('/api/assets/install-all', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ kind }),
      })
      if (result.started.length > 0) {
        setAssets((current) => {
          const next = { ...current }
          for (const key of result.started) {
            const existing = next[key]
            if (existing) {
              next[key] = { ...existing, status: 'running', message: 'Запуск…', error: null }
            }
          }
          return next
        })
        setNotice(`Установка запущена: ${result.started.length}. Идёт последовательно.`)
      } else {
        setNotice('Нет отсутствующих компонентов этого вида.')
      }
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setBatchBusy(false)
    }
  }

  const removeAll = async (keys?: string[]) => {
    const count = keys ? keys.length : installedBinaries.length
    if (count === 0) return
    const what = keys ? `выбранные компоненты (${count})` : `все установленные бинарники (${count})`
    if (
      !window.confirm(
        `Удалить ${what}? Файлы будут удалены из служебного каталога.`,
      )
    ) {
      return
    }
    setBatchBusy(true)
    setError(null)
    setNotice(null)
    try {
      const result = await api<AssetDeleteAllResponse>('/api/assets/delete-all', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(keys ? { keys } : {}),
      })
      setSelected(new Set())
      await refresh()
      onChangedRef.current?.()
      if (result.deleted.length > 0) {
        setNotice(`Удалено: ${result.deleted.length}.`)
      }
      if (result.failed.length > 0) {
        setError(
          `Не удалось удалить: ${result.failed
            .map((item) => `${item.key} — ${item.reason}`)
            .join('; ')}`,
        )
      }
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setBatchBusy(false)
    }
  }

  const toggleSelected = (key: string) => {
    setSelected((current) => {
      const next = new Set(current)
      if (next.has(key)) {
        next.delete(key)
      } else {
        next.add(key)
      }
      return next
    })
  }

  // Требования: либо из плана мастера, либо — автономно — из реестра ресурсов.
  const summary = useMemo(() => {
    if (requirements) {
      return requirements.map((requirement) => ({
        requirement,
        assetKey: requirement.asset_key ?? requirement.dep_key ?? requirement.key,
      }))
    }
    return assetList
      .map((asset) => ({ requirement: requirementFromAsset(asset), assetKey: asset.key }))
      .sort((a, b) => a.requirement.label.localeCompare(b.requirement.label, 'ru'))
  }, [requirements, assetList])

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
      {notice && (
        <Alert tone="info" onDismiss={() => setNotice(null)}>
          {notice}
        </Alert>
      )}
      {auto && (missingBinaries.length > 0 || missingPip.length > 0 || installedBinaries.length > 0) && (
        <div className="flex flex-wrap items-center gap-2 rounded-md border border-border bg-surface-2 p-2">
          {missingBinaries.length > 0 && (
            <Button
              variant="primary"
              size="sm"
              loading={batchBusy}
              disabled={actionsDisabled}
              onClick={() => void installAll('binary')}
            >
              {`Скачать все (${missingBinaries.length})`}
            </Button>
          )}
          {missingPip.length > 0 && canInstallPip && (
            <Button
              variant="primary"
              size="sm"
              loading={batchBusy}
              disabled={actionsDisabled}
              onClick={() => void installAll('pip')}
            >
              {`Установить все (${missingPip.length})`}
            </Button>
          )}
          {installedBinaries.length > 0 && (
            <Button
              variant={selected.size > 0 ? 'danger' : 'secondary'}
              size="sm"
              disabled={actionsDisabled}
              onClick={() => void removeAll(selected.size > 0 ? [...selected] : undefined)}
            >
              {selected.size > 0
                ? `Удалить выбранные (${selected.size})`
                : `Удалить все (${installedBinaries.length})`}
            </Button>
          )}
          <span className="text-xs text-muted">
            Установка идёт последовательно; удаление — только для бинарников в служебном каталоге.
          </span>
        </div>
      )}
      {requirements != null && requirements.length === 0 ? (
        <Alert tone="success">Для выбранного режима отдельные компоненты не требуются.</Alert>
      ) : requirements == null && !loaded ? (
        <div className="flex items-center justify-center py-6">
          <Spinner size={20} label="Загрузка списка компонентов" />
        </div>
      ) : summary.length === 0 ? (
        <Alert tone="info">Список внешних компонентов пуст.</Alert>
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
            const managed = asset?.managed ?? false
            return (
              <li key={requirement.key} className="rounded-md border border-border p-3 text-sm">
                <p className="flex flex-wrap items-center gap-2 font-medium">
                  {auto && installed && isBinary && (
                    <Checkbox
                      checked={selected.has(assetKey)}
                      onChange={() => toggleSelected(assetKey)}
                      aria-label={`Выбрать «${requirement.label}»`}
                    />
                  )}
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
                      disabled={actionsDisabled || !canAct}
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
                  {installed && canAct && (
                    <Button
                      variant="secondary"
                      size="sm"
                      loading={busy === assetKey || running}
                      disabled={actionsDisabled}
                      onClick={() => asset && void install(asset)}
                    >
                      {running ? 'Обновление…' : 'Обновить'}
                    </Button>
                  )}
                  {installed && isBinary && (
                    <Button
                      variant="danger"
                      size="sm"
                      loading={busy === assetKey}
                      disabled={actionsDisabled || !managed}
                      title={
                        managed
                          ? undefined
                          : 'Установлен вне служебного каталога — удалите файл вручную'
                      }
                      onClick={() => asset && void remove(asset)}
                    >
                      Удалить
                    </Button>
                  )}
                  {installed && isBinary && !managed && (
                    <span className="text-warn">
                      Вне служебного каталога — удаление вручную
                    </span>
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
