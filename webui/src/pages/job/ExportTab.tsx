import { useState } from 'react'
import { Download } from 'lucide-react'

import { EXPORT_FORMATS } from '../../api'
import { useApp } from '../../app/useApp'
import {
  Alert,
  Button,
  Card,
  CardContent,
  CardHeader,
  Checkbox,
  EmptyState,
  Field,
  Select,
  Spinner,
} from '../../components/ui'
import SummaryPromptsPanel from '../../components/SummaryPromptsPanel'

const LINK_BUTTON =
  'inline-flex h-9 items-center justify-center gap-2 rounded-md border border-border-strong ' +
  'bg-surface px-4 text-sm font-medium text-text transition-colors hover:bg-surface-2 ' +
  'focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring'

function ExportTab({ jobId }: { jobId: string }) {
  const app = useApp()
  const [promptsOpen, setPromptsOpen] = useState(false)
  const [highlightWords, setHighlightWords] = useState(false)

  const highlightSupported =
    app.exportFormat === 'srt' || app.exportFormat === 'vtt'

  if (!app.result) {
    return (
      <EmptyState
        title="Экспорт пока недоступен"
        description="Результат появится после завершения обработки"
      />
    )
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader title="Протокол и резюме" description="LLM-обработка стенограммы" />
        <CardContent className="space-y-3">
          <div className="flex flex-wrap items-end gap-3">
            <Button
              variant="primary"
              loading={app.protocolBusy}
              onClick={() => void app.generateProtocol(jobId)}
            >
              Сформировать протокол
            </Button>
            {app.summaryPrompts.length > 0 && (
              <Field label="Промпт резюме" htmlFor="export-summary-prompt" className="min-w-44">
                <Select
                  id="export-summary-prompt"
                  value={app.protocolPromptId}
                  title="Шаблон промпта резюме для этого протокола"
                  onChange={(event) =>
                    app.setProtocolPromptId(event.target.value ? Number(event.target.value) : '')
                  }
                >
                  {app.summaryPrompts.map((prompt) => (
                    <option key={prompt.id} value={prompt.id}>
                      {prompt.name}
                      {prompt.builtin ? ' (встроенный)' : ''}
                    </option>
                  ))}
                </Select>
              </Field>
            )}
            <Button
              variant="secondary"
              aria-expanded={promptsOpen}
              onClick={() => setPromptsOpen((value) => !value)}
            >
              Промпты резюме
            </Button>
          </div>

          {app.protocolBusy && (
            <p className="flex items-center gap-2 text-xs text-muted">
              <Spinner size={14} />
              Считается резюме и экспорт — это может занять время
            </p>
          )}

          {app.protocolError && (
            <Alert tone="danger" live>
              {app.protocolError}
            </Alert>
          )}

          {app.summary && (
            <div className="space-y-1">
              <p className="text-xs font-medium uppercase text-muted">Резюме встречи</p>
              <p className="whitespace-pre-wrap text-sm text-text">{app.summary}</p>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader title="Скачать" description="Стенограмма и протокол в файле" />
        <CardContent className="flex flex-wrap items-end gap-3">
          <Field label="Формат" htmlFor="export-format" className="min-w-24">
            <Select
              id="export-format"
              value={app.exportFormat}
              onChange={(event) => app.setExportFormat(event.target.value)}
            >
              {EXPORT_FORMATS.map((fmt) => (
                <option key={fmt} value={fmt}>
                  {fmt.toUpperCase()}
                </option>
              ))}
            </Select>
          </Field>
          {highlightSupported && (
            <Checkbox
              label="Подсветка слов"
              checked={highlightWords}
              onChange={(event) => setHighlightWords(event.target.checked)}
            />
          )}
          <a
            className={LINK_BUTTON}
            href={`/api/jobs/${jobId}/export?fmt=${app.exportFormat}${
              highlightSupported && highlightWords ? '&highlight=1' : ''
            }`}
            download
          >
            <Download aria-hidden className="h-4 w-4" />
            Скачать расшифровку
          </a>
          {app.protocol && (
            <>
              <a className={LINK_BUTTON} href={`/api/jobs/${jobId}/protocol/download?fmt=txt`}>
                Скачать .txt
              </a>
              <a className={LINK_BUTTON} href={`/api/jobs/${jobId}/protocol/download?fmt=docx`}>
                Скачать .docx
              </a>
            </>
          )}
        </CardContent>
      </Card>

      {promptsOpen && (
        <SummaryPromptsPanel onChanged={() => void app.refreshSummaryPrompts()} />
      )}
    </div>
  )
}

export default ExportTab
