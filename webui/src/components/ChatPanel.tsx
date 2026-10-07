import { useCallback, useEffect, useRef, useState } from 'react'
import type { KeyboardEvent as ReactKeyboardEvent } from 'react'

import {
  clearChatHistory,
  errorMessage,
  fetchChatHistory,
  formatClock,
  streamChat,
  type ChatCitation,
  type ChatMessage,
} from '../api'

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
                <button
                  type="button"
                  onClick={() => onCite(byIndex.get(index)!)}
                  className="rounded bg-blue-100 px-1 font-medium text-blue-700 hover:bg-blue-200 dark:bg-blue-950/60 dark:text-blue-300 dark:hover:bg-blue-900"
                >
                  {index}
                </button>
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
    <div className="space-y-3 rounded-md border border-slate-200 bg-slate-50 p-3 dark:border-slate-800 dark:bg-slate-800/50">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="font-medium">Чат по стенограмме</h3>
        <button
          type="button"
          onClick={() => void clear()}
          disabled={busy || messages.length === 0}
          title="Удалить историю чата по этой записи"
          className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-xs hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-40 dark:border-slate-600 dark:bg-slate-900 dark:hover:bg-slate-800"
        >
          Очистить
        </button>
      </div>

      {error && (
        <p
          role="alert"
          className="rounded-md bg-red-50 px-3 py-1.5 text-xs text-red-700 dark:bg-red-950/50 dark:text-red-300"
        >
          {error}
        </p>
      )}

      <div className="max-h-[24rem] space-y-2 overflow-auto rounded-md border border-slate-200 bg-white p-3 dark:border-slate-700 dark:bg-slate-900">
        {loading ? (
          <p className="py-4 text-center text-sm text-slate-400 dark:text-slate-500">
            Загрузка истории…
          </p>
        ) : messages.length === 0 ? (
          <p className="py-4 text-center text-sm text-slate-400 dark:text-slate-500">
            Задайте вопрос по стенограмме: «о чём договорились?», «какие сроки назвали?»
          </p>
        ) : (
          messages.map((message) => (
            <div
              key={message.key}
              className={
                message.role === 'user'
                  ? 'ml-auto max-w-[85%] rounded-md bg-blue-50 px-3 py-2 text-sm dark:bg-blue-950/40'
                  : 'mr-auto max-w-[85%] rounded-md bg-slate-100 px-3 py-2 text-sm dark:bg-slate-800'
              }
            >
              <p className="mb-0.5 text-[10px] uppercase text-slate-400 dark:text-slate-500">
                {message.role === 'user' ? 'Вы' : 'Ассистент'}
                {message.pending ? ' · печатает…' : ''}
              </p>
              {message.failed ? (
                <p className="break-words text-red-700 dark:text-red-300">{message.content}</p>
              ) : (
                <MessageContent
                  content={message.content}
                  citations={message.citations}
                  onCite={playCitation}
                />
              )}
              {message.citations.length > 0 && (
                <div className="mt-2 flex flex-wrap items-center gap-1.5 border-t border-slate-200 pt-2 text-xs dark:border-slate-700">
                  <span className="text-slate-400 dark:text-slate-500">Источники:</span>
                  {message.citations.map((citation) => (
                    <button
                      key={citation.index}
                      type="button"
                      onClick={() => playCitation(citation)}
                      title={`${citation.speaker}: ${citation.text}`}
                      aria-pressed={playingIndex === citation.index}
                      className={
                        playingIndex === citation.index
                          ? 'rounded border border-blue-400 bg-blue-100 px-1.5 py-0.5 text-blue-700 dark:border-blue-700 dark:bg-blue-950/60 dark:text-blue-300'
                          : 'rounded border border-slate-300 px-1.5 py-0.5 text-slate-600 hover:bg-slate-100 dark:border-slate-600 dark:text-slate-300 dark:hover:bg-slate-800'
                      }
                    >
                      [{citation.index}] {formatClock(citation.start)} · {citation.speaker}
                    </button>
                  ))}
                </div>
              )}
            </div>
          ))
        )}
        <div ref={endRef} />
      </div>

      <div className="flex items-end gap-2">
        <textarea
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={onKeyDown}
          rows={2}
          placeholder="Вопрос по стенограмме… (Enter — отправить, Shift+Enter — новая строка)"
          disabled={busy}
          className="min-h-[2.5rem] flex-1 resize-y rounded-md border border-slate-300 px-3 py-1.5 text-sm focus:border-blue-400 focus:outline-none disabled:opacity-60 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100 dark:placeholder-slate-500"
        />
        <button
          type="button"
          onClick={() => void send()}
          disabled={busy || input.trim().length === 0}
          className="rounded-md bg-slate-800 px-3 py-1.5 text-sm text-white hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-40 dark:bg-slate-200 dark:text-slate-900 dark:hover:bg-white"
        >
          {busy ? 'Отвечаю…' : 'Отправить'}
        </button>
      </div>

      <audio
        ref={audioRef}
        preload="metadata"
        src={`/api/jobs/${jobId}/audio`}
        className="hidden"
      />
    </div>
  )
}

export default ChatPanel
