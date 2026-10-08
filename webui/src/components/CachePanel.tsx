import { useCallback, useEffect, useState } from 'react'
import { Trash2 } from 'lucide-react'

import {
  clearCache,
  errorMessage,
  fetchCacheInfo,
  formatBytes,
  type CacheInfo,
} from '../api'
import { Alert, Button, Card, CardContent, CardHeader, Checkbox, Modal, Spinner } from './ui'

/**
 * Карточка «Кэш обработки» (#100): показывает размер постадийного кэша и
 * позволяет очистить его вручную. Очистка не трогает идущий прогон без явного
 * подтверждения — по умолчанию сервер отвечает 409, а тут появляется флажок
 * «всё равно очистить».
 */
function CachePanel() {
  const [info, setInfo] = useState<CacheInfo | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [open, setOpen] = useState(false)
  const [force, setForce] = useState(false)
  const [busy, setBusy] = useState(false)

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      setInfo(await fetchCacheInfo())
      setError(null)
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const activeJobs = info?.active_jobs.length ?? 0

  const doClear = async () => {
    setBusy(true)
    setError(null)
    try {
      const result = await clearCache(force)
      setNotice(
        result.removed > 0
          ? `Кэш очищен: удалено файлов — ${result.removed}.`
          : 'Кэш уже пуст — удалять нечего.',
      )
      setOpen(false)
      setForce(false)
      await refresh()
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <Card>
        <CardHeader
          title="Кэш обработки"
          description="Промежуточные результаты стадий (USE_CACHE). Очистка заставит пересчитать стадии заново"
          actions={
            <Button
              variant="secondary"
              size="sm"
              icon={<Trash2 aria-hidden className="h-4 w-4" />}
              disabled={loading}
              onClick={() => {
                setNotice(null)
                setForce(false)
                setOpen(true)
              }}
            >
              Очистить кэш
            </Button>
          }
        />
        <CardContent className="space-y-3">
          {error && (
            <Alert tone="danger" live onDismiss={() => setError(null)}>
              {error}
            </Alert>
          )}
          {notice && (
            <Alert tone="success" live onDismiss={() => setNotice(null)}>
              {notice}
            </Alert>
          )}
          {loading ? (
            <div className="flex items-center justify-center py-2">
              <Spinner size={16} label="Чтение состояния кэша" />
            </div>
          ) : (
            info && (
              <dl className="grid grid-cols-1 gap-x-6 gap-y-1 text-sm sm:grid-cols-2">
                <div className="flex items-baseline justify-between gap-3">
                  <dt className="text-muted">Файлов</dt>
                  <dd className="tabular-nums">{info.files}</dd>
                </div>
                <div className="flex items-baseline justify-between gap-3">
                  <dt className="text-muted">Размер</dt>
                  <dd className="tabular-nums">{formatBytes(info.bytes)}</dd>
                </div>
                <div className="flex min-w-0 items-baseline justify-between gap-3 sm:col-span-2">
                  <dt className="shrink-0 text-muted">Каталог</dt>
                  <dd className="truncate font-mono text-[11px]" title={info.directory}>
                    {info.directory}
                  </dd>
                </div>
                {activeJobs > 0 && (
                  <div className="flex items-baseline justify-between gap-3 sm:col-span-2">
                    <dt className="text-muted">Активных прогонов</dt>
                    <dd className="tabular-nums text-warn">{activeJobs}</dd>
                  </div>
                )}
              </dl>
            )
          )}
        </CardContent>
      </Card>

      <Modal
        open={open}
        onClose={() => {
          if (!busy) setOpen(false)
        }}
        title="Очистить кэш?"
        size="sm"
        footer={
          <>
            <Button variant="ghost" onClick={() => setOpen(false)} disabled={busy}>
              Отмена
            </Button>
            <Button
              variant="danger"
              loading={busy}
              disabled={activeJobs > 0 && !force}
              onClick={() => void doClear()}
            >
              Очистить
            </Button>
          </>
        }
      >
        <div className="space-y-3 text-sm">
          <p>
            Будет удалено файлов кэша:{' '}
            <span className="font-medium tabular-nums">{info?.files ?? 0}</span> (
            {formatBytes(info?.bytes ?? 0)}). Следующий прогон пересчитает очищенные стадии.
          </p>
          {activeJobs > 0 && (
            <>
              <Alert tone="warn" title="Идёт обработка">
                Сейчас задач в работе: {activeJobs}. Очистка сбросит промежуточные результаты
                активного прогона — он продолжит работу, но пересчитает стадии заново.
              </Alert>
              <Checkbox
                checked={force}
                onChange={(event) => setForce(event.target.checked)}
                label="Всё равно очистить (принудительно)"
              />
            </>
          )}
        </div>
      </Modal>
    </>
  )
}

export default CachePanel
