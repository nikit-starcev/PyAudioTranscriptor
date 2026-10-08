import { useCallback, useEffect, useRef, useState } from 'react'
import type { KeyboardEvent as ReactKeyboardEvent } from 'react'
import { Send, Trash2 } from 'lucide-react'

import {
  clearChatHistory,
  errorMessage,
  fetchChatHistory,
  formatClock,
  streamChat,
  type ChatCitation,
  type ChatMessage,
} from '../api'
import { Alert, Button, Card, CardHeader, Spinner, Textarea, cn } from './ui'

type Props = {
  jobId: string
}

type DisplayMessage = {
  key: string
  role: 'user' | 'assistant'
  content: string
  citations: ChatCitation[]
  pending?: boolean
  failed?: boolean
}

const CITATION_MARKER = /^\[\s*\d+(?:\s*,\s*\d+)*\s*\]$/

function toDisplay(message: ChatMessage): DisplayMessage {
  return {
    key: `m-${message.id}`,
    role: message.role,
    content: message.content,
    citations: message.citations ?? [],
  }
}

/**
 * Тело ответа с кликабельными ссылками на реплики: номера ``[N]`` в тексте
 * превращаются в кнопки, включающие воспроизведение реплики-источника.
 */
function MessageContent({
  content,
  citations,
  onCite,
}: {
  content: string
  citations: ChatCitation[]
  onCite: (citation: ChatCitation) => void
}) {
  const byIndex = new Map(citations.map((citation) => [citation.index, citation]))
  const parts = content.split(/(\[\s*\d+(?:\s*,\s*\d+)*\s*\])/g)
  return (
    <p className="whitespace-pre-wrap break-words">
      {parts.map((part, partIndex) => {
        if (!CITATION_MARKER.test(part)) return <span key={partIndex}>{part}</span>
        const indexes = part
          .slice(1, -1)
          .split(',')
          .map((token) => Number(token.trim()))
          .filter((index) => byIndex.has(index))
        if (indexes.length === 0) return <span key={partIndex}>{part}</span>
        return (
          <span key={partIndex} className="whitespace-nowrap">
            [
            {indexes.map((index, position) => (
              <span key={index}>
                {position > 0 && ', '}
                <Button
                  variant="secondary"
                  size="sm"
                  className="align-middle"
                  aria-label={`Источник ${index}`}
                  onClick={() => onCite(byIndex.get(index)!)}
                >
                  {index}
                </Button>
              </span>
            ))}
            ]
          </span>
        )
      })}
    </p>
  )
}

/**
 * Чат по стенограмме активной задачи (#54/#96): вопрос → потоковый ответ
 * локальной LLM с цитатами-ссылками. Клик по цитате проигрывает фрагмент
 * записи и подсвечивает активную ссылку; история сохраняется на сервере.
 */
function ChatPanel({ jobId }: Props) {
  const [messages, setMessages] = useState<DisplayMessage[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [playingIndex, setPlayingIndex] = useState<number | null>(null)

  const audioRef = useRef<HTMLAudioElement | null>(null)
  const rafRef = useRef<number | null>(null)
  const stopAtRef = useRef<number | null>(null)
  const endRef = useRef<HTMLDivElement | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const cancelLoop = useCallback(() => {
    if (rafRef.current != null) {
      cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }
  }, [])

  const stopAudio = useCallback(() => {
    cancelLoop()
    stopAtRef.current = null
    audioRef.current?.pause()
    setPlayingIndex(null)
  }, [cancelLoop])

  const playCitation = useCallback(
    (citation: ChatCitation) => {
      const audio = audioRef.current
      if (!audio) return
      cancelLoop()
      stopAtRef.current = citation.end
      setPlayingIndex(citation.index)
      const begin = () => {
        audio.currentTime = Math.max(0, citation.start)
        void audio
          .play()
          .then(() => {
            const tick = () => {
              const element = audioRef.current
              const stopAt = stopAtRef.current
              if (!element || stopAt == null) return
              if (element.currentTime >= stopAt - 0.005 || element.ended) {
                stopAudio()
                return
              }
              rafRef.current = requestAnimationFrame(tick)
            }
            rafRef.current = requestAnimationFrame(tick)
          })
          .catch(() => stopAudio())
      }
      if (audio.readyState >= 1) begin()
      else {
        const onReady = () => {
          audio.removeEventListener('loadedmetadata', onReady)
          begin()
        }
        audio.addEventListener('loadedmetadata', onReady)
        audio.load()
      }
    },
    [cancelLoop, stopAudio],
  )

  useEffect(() => stopAudio, [jobId, stopAudio])

  useEffect(() => {
    let cancelled = false
    setMessages([])
    setInput('')
    setError(null)
    setLoading(true)
    stopAudio()
    fetchChatHistory(jobId)
      .then((data) => {
        if (!cancelled) setMessages(data.messages.map(toDisplay))
      })
      .catch((cause) => {
        if (!cancelled) setError(errorMessage(cause))
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [jobId, stopAudio])

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: 'end' })
  }, [messages])

  const send = useCallback(async () => {
    const text = input.trim()
    if (!text || busy) return
    setError(null)
    setInput('')
    setBusy(true)
    const assistantKey = `stream-${Date.now()}`
    setMessages((current) => [
      ...current,
      { key: `user-${Date.now()}`, role: 'user', content: text, citations: [] },
      { key: assistantKey, role: 'assistant', content: '', citations: [], pending: true },
    ])
    const controller = new AbortController()
    abortRef.current = controller
    try {
      await streamChat(
        jobId,
        text,
        (event) => {
          if (event.type === 'token') {
            setMessages((current) =>
              current.map((message) =>
                message.key === assistantKey
                  ? { ...message, content: message.content + event.text }
                  : message,
              ),
            )
          } else if (event.type === 'done') {
            setMessages((current) =>
              current.map((message) =>
                message.key === assistantKey
                  ? {
                      ...message,
                      content: event.content,
                      citations: event.citations,
                      pending: false,
                    }
                  : message,
              ),
            )
          } else if (event.type === 'error') {
            setError(event.message)
            setMessages((current) =>
              current.map((message) =>
                message.key === assistantKey
                  ? { ...message, content: event.message, failed: true, pending: false }
                  : message,
              ),
            )
          }
        },
        controller.signal,
      )
    } catch (cause) {
      const message = errorMessage(cause)
      setError(message)
      setMessages((current) =>
        current.map((item) =>
          item.key === assistantKey && item.pending
            ? { ...item, content: item.content || message, failed: true, pending: false }
            : item,
        ),
      )
    } finally {
      abortRef.current = null
      setBusy(false)
    }
  }, [busy, input, jobId])

  const clear = useCallback(async () => {
    if (!window.confirm('Очистить историю чата по этой записи?')) return
    setBusy(true)
    setError(null)
    stopAudio()
    try {
      await clearChatHistory(jobId)
      setMessages([])
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setBusy(false)
    }
  }, [jobId, stopAudio])

  const onKeyDown = (event: ReactKeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      void send()
    }
  }

  return (
    <Card>
      <CardHeader
        title="Чат по стенограмме"
        description="Вопрос → ответ локальной LLM с ссылками на реплики"
        actions={
          <Button
            variant="secondary"
            size="sm"
            icon={<Trash2 aria-hidden className="h-4 w-4" />}
            disabled={busy || messages.length === 0}
            title="Удалить историю чата по этой записи"
            onClick={() => void clear()}
          >
            Очистить
          </Button>
        }
      />
      <div className="space-y-3 p-4">
        {error && (
          <Alert tone="danger" live onDismiss={() => setError(null)}>
            {error}
          </Alert>
        )}

        <div className="max-h-96 space-y-2 overflow-auto rounded-md border border-border bg-surface-2/40 p-3">
          {loading ? (
            <div className="flex items-center justify-center gap-2 py-4 text-sm text-muted">
              <Spinner size={16} /> Загрузка истории…
            </div>
          ) : messages.length === 0 ? (
            <p className="py-4 text-center text-sm text-muted">
              Задайте вопрос по стенограмме: «о чём договорились?», «какие сроки назвали?»
            </p>
          ) : (
            messages.map((message) => (
              <div
                key={message.key}
                className={cn(
                  'max-w-[85%] rounded-md px-3 py-2 text-sm',
                  message.role === 'user'
                    ? 'ml-auto bg-primary-soft text-primary-soft-fg'
                    : 'mr-auto bg-surface-3 text-text',
                )}
              >
                <p className="mb-0.5 text-[10px] uppercase text-muted">
                  {message.role === 'user' ? 'Вы' : 'Ассистент'}
                  {message.pending ? ' · печатает…' : ''}
                </p>
                {message.failed ? (
                  <p className="break-words text-danger">{message.content}</p>
                ) : (
                  <MessageContent
                    content={message.content}
                    citations={message.citations}
                    onCite={playCitation}
                  />
                )}
                {message.citations.length > 0 && (
                  <div className="mt-2 flex flex-wrap items-center gap-1.5 border-t border-border pt-2 text-xs">
                    <span className="text-muted">Источники:</span>
                    {message.citations.map((citation) => (
                      <Button
                        key={citation.index}
                        variant={playingIndex === citation.index ? 'primary' : 'secondary'}
                        size="sm"
                        aria-pressed={playingIndex === citation.index}
                        title={`${citation.speaker}: ${citation.text}`}
                        onClick={() => playCitation(citation)}
                      >
                        [{citation.index}] {formatClock(citation.start)} · {citation.speaker}
                      </Button>
                    ))}
                  </div>
                )}
              </div>
            ))
          )}
          <div ref={endRef} />
        </div>

        <div className="flex items-end gap-2">
          <Textarea
            aria-label="Вопрос по стенограмме"
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={onKeyDown}
            rows={2}
            placeholder="Вопрос по стенограмме… (Enter — отправить, Shift+Enter — новая строка)"
            disabled={busy}
            className="flex-1"
          />
          <Button
            variant="primary"
            icon={<Send aria-hidden className="h-4 w-4" />}
            loading={busy}
            disabled={input.trim().length === 0}
            onClick={() => void send()}
          >
            {busy ? 'Отвечаю…' : 'Отправить'}
          </Button>
        </div>

        <audio ref={audioRef} preload="metadata" src={`/api/jobs/${jobId}/audio`} className="hidden" />
      </div>
    </Card>
  )
}

export default ChatPanel
