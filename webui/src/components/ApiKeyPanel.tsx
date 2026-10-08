import { useCallback, useEffect, useState } from 'react'
import { Check, Copy, Eye, EyeOff, RefreshCw, Trash2 } from 'lucide-react'

import {
  clearApiKey,
  errorMessage,
  fetchApiKey,
  generateApiKey,
  type ApiKeyStatus,
} from '../api'
import {
  Alert,
  Badge,
  Button,
  Card,
  CardContent,
  CardHeader,
  Field,
  Input,
  Spinner,
  Tooltip,
} from './ui'

/**
 * Карточка «API-ключ» (#110): генерация, показ/копирование и очистка ключа
 * OpenAI-совместимого API `/v1`. Ключ хранится локально в
 * `web-data/secrets.json` (права 0600) и имеет приоритет над `config.env`.
 * Сгенерированный ключ действует сразу — сервер перечитывает его на каждом
 * запросе, перезапуск не нужен.
 */
function ApiKeyPanel() {
  const [status, setStatus] = useState<ApiKeyStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState<'generate' | 'clear' | null>(null)
  const [revealed, setRevealed] = useState(false)
  const [copied, setCopied] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)

  const apply = useCallback((next: ApiKeyStatus) => {
    setStatus(next)
    setRevealed(false)
    setCopied(false)
  }, [])

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      apply(await fetchApiKey())
      setError(null)
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setLoading(false)
    }
  }, [apply])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const generate = async () => {
    if (
      status?.set &&
      !window.confirm(
        'Перегенерировать ключ? Старый ключ перестанет работать, и подключённым клиентам ' +
          'нужно будет указать новый.',
      )
    ) {
      return
    }
    setBusy('generate')
    setError(null)
    setNotice(null)
    try {
      apply(await generateApiKey())
      setNotice('Ключ сгенерирован. Он уже действует — перезапуск не нужен.')
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setBusy(null)
    }
  }

  const clear = async () => {
    if (
      !window.confirm(
        'Очистить ключ? OpenAI-совместимый API перестанет принимать запросы ' +
          '(если ключ не задан также в config.env).',
      )
    ) {
      return
    }
    setBusy('clear')
    setError(null)
    setNotice(null)
    try {
      apply(await clearApiKey())
      setNotice('Ключ очищен.')
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setBusy(null)
    }
  }

  const copy = async () => {
    if (!status?.key) return
    try {
      await navigator.clipboard.writeText(status.key)
      setCopied(true)
      setError(null)
    } catch {
      setError('Не удалось скопировать ключ в буфер обмена.')
    }
  }

  const set = status?.set ?? false
  const fromEnv = status?.source === 'env'

  return (
    <Card>
      <CardHeader
        title="API-ключ (OpenAI-совместимый API)"
        description="Ключ для клиентов /v1 (Open-WebUI, LM Studio, скрипты). Хранится локально в web-data/secrets.json"
        actions={loading ? <Spinner size={16} label="Чтение статуса ключа" /> : undefined}
      />
      <CardContent className="space-y-4">
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

        {loading ? null : (
          <>
            <div className="flex flex-wrap items-center gap-2">
              <Badge tone={set ? 'success' : 'danger'}>{set ? 'Задан' : 'Не задан'}</Badge>
              {fromEnv && <Badge tone="info">из config.env</Badge>}
            </div>

            {fromEnv ? (
              <p className="max-w-prose text-pretty rounded-md bg-surface-2 px-3 py-1.5 text-xs text-muted">
                Действующий ключ берётся из <code>config.env</code> (<code>API_KEY</code>).
                Сгенерированный здесь ключ запишется в <code>secrets.json</code> и перекроет
                значение из <code>config.env</code>.
              </p>
            ) : (
              <p className="max-w-prose text-pretty rounded-md bg-surface-2 px-3 py-1.5 text-xs text-muted">
                Если ключ не задан, эндпоинты <code>/v1</code> отвечают <code>401</code>. Ключ
                передаётся заголовком <code>Authorization: Bearer &lt;ключ&gt;</code>.
              </p>
            )}

            {set && status?.key && (
              <>
                <Field label="Действующий ключ" htmlFor="api-key-value">
                  <Input
                    id="api-key-value"
                    className="font-mono"
                    readOnly
                    type={revealed ? 'text' : 'password'}
                    value={status.key}
                    aria-label="Действующий API-ключ"
                    onFocus={(event) => event.currentTarget.select()}
                  />
                </Field>
                <div className="flex flex-wrap items-center gap-2">
                  <Button
                    variant="secondary"
                    size="sm"
                    icon={
                      revealed ? (
                        <EyeOff aria-hidden className="h-4 w-4" />
                      ) : (
                        <Eye aria-hidden className="h-4 w-4" />
                      )
                    }
                    onClick={() => setRevealed((value) => !value)}
                  >
                    {revealed ? 'Скрыть' : 'Показать'}
                  </Button>
                  <Tooltip label="Копировать ключ">
                    <Button
                      variant="secondary"
                      size="sm"
                      icon={
                        copied ? (
                          <Check aria-hidden className="h-4 w-4" />
                        ) : (
                          <Copy aria-hidden className="h-4 w-4" />
                        )
                      }
                      aria-label="Копировать API-ключ"
                      onClick={() => void copy()}
                    >
                      {copied ? 'Скопировано' : 'Копировать'}
                    </Button>
                  </Tooltip>
                </div>
              </>
            )}

            <div className="flex flex-wrap items-center gap-2">
              <Button
                variant={set ? 'secondary' : 'primary'}
                icon={<RefreshCw aria-hidden className="h-4 w-4" />}
                loading={busy === 'generate'}
                disabled={busy !== null}
                onClick={() => void generate()}
              >
                {set ? 'Перегенерировать' : 'Сгенерировать'}
              </Button>
              <Tooltip label={status?.secret_set ? 'Очистить ключ' : 'Ключ в секретах не задан'}>
                <Button
                  variant="danger"
                  icon={<Trash2 aria-hidden className="h-4 w-4" />}
                  loading={busy === 'clear'}
                  disabled={busy !== null || !status?.secret_set}
                  onClick={() => void clear()}
                >
                  Очистить
                </Button>
              </Tooltip>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  )
}

export default ApiKeyPanel
