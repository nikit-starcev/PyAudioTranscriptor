import { useRef } from 'react'
import { FileAudio, RotateCcw, Trash2, Upload } from 'lucide-react'

import { formatBytes, formatDuration } from '../../api'
import { useApp } from '../../app/useApp'
import { navigateTo } from '../../app/routes'
import {
  Badge,
  Button,
  Card,
  CardContent,
  CardHeader,
  Checkbox,
  EmptyState,
  IconButton,
  Input,
} from '../../components/ui'

function FilesPanel() {
  const app = useApp()
  const fileInput = useRef<HTMLInputElement>(null)

  const onPick = () => fileInput.current?.click()

  return (
    <Card className="h-fit">
      <CardHeader
        title="Файлы"
        description="Загрузка аудио и постановка в очередь"
        actions={
          <Button
            variant="primary"
            size="sm"
            icon={<Upload aria-hidden className="h-4 w-4" />}
            onClick={onPick}
          >
            Загрузить
          </Button>
        }
      />
      <CardContent className="space-y-3">
        <div className="flex items-center justify-between gap-2">
          <Checkbox
            label="Обработанные"
            checked={app.showProcessed}
            onChange={(event) => app.toggleProcessed(event.target.checked)}
          />
          <input
            ref={fileInput}
            type="file"
            aria-label="Загрузить аудиофайл"
            className="hidden"
            onChange={(event) => {
              const file = event.target.files?.[0]
              if (file) void app.upload(file)
              event.target.value = ''
            }}
          />
        </div>

        {app.files.length === 0 ? (
          <EmptyState
            icon={<FileAudio aria-hidden className="h-7 w-7" />}
            title="Файлов пока нет"
            description="Загрузите аудио, чтобы поставить его в очередь"
          />
        ) : (
          <ul className="space-y-2">
            {app.files.map((file, index) => (
              <li key={file.path} className="rounded-md border border-border p-3">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <p className="line-clamp-2 break-words text-sm" title={file.name}>
                      {file.name}
                    </p>
                    <p className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-muted">
                      <span className="tabular-nums">{formatBytes(file.size)}</span>
                      <span aria-hidden>·</span>
                      <span className="tabular-nums">{formatDuration(file.duration)}</span>
                      <Badge tone={file.processed ? 'success' : 'neutral'}>
                        {file.processed ? 'Обработан' : 'Не обработан'}
                      </Badge>
                    </p>
                  </div>
                  <div className="flex shrink-0 items-center gap-1.5">
                    {file.processed ? (
                      <Button
                        variant="secondary"
                        size="sm"
                        icon={<RotateCcw aria-hidden className="h-3.5 w-3.5" />}
                        onClick={() => void app.restoreFile(file.name)}
                      >
                        Вернуть
                      </Button>
                    ) : (
                      <Button
                        variant="primary"
                        size="sm"
                        disabled={app.readinessBlocked}
                        title={app.blockedHint}
                        onClick={() => {
                          void app.enqueue(file.path).then((id) => {
                            if (id) navigateTo(`/jobs/${encodeURIComponent(id)}`)
                          })
                        }}
                      >
                        В очередь
                      </Button>
                    )}
                    <IconButton
                      aria-label={`Удалить файл ${file.name}`}
                      size="sm"
                      onClick={() => void app.deleteFile(file.name)}
                    >
                      <Trash2 aria-hidden className="h-4 w-4" />
                    </IconButton>
                  </div>
                </div>

                {!file.processed && (
                  <div className="mt-3 flex flex-wrap items-end gap-3">
                    <label className="flex flex-col gap-1 text-xs text-muted">
                      Точно
                      <Input
                        id={`exact-${index}`}
                        type="number"
                        min={1}
                        step={1}
                        placeholder="авто"
                        title="Точное число говорящих: пусто — автоопределение"
                        value={app.speakerCounts[file.path] ?? ''}
                        onChange={(event) =>
                          app.setSpeakerCounts((prev) => ({
                            ...prev,
                            [file.path]: event.target.value,
                          }))
                        }
                        className="w-20"
                      />
                    </label>
                    <label className="flex flex-col gap-1 text-xs text-muted">
                      Мин
                      <Input
                        id={`min-${index}`}
                        type="number"
                        min={1}
                        step={1}
                        placeholder="—"
                        title="Нижняя граница числа говорящих: пусто — без ограничения"
                        value={app.speakerMins[file.path] ?? ''}
                        onChange={(event) =>
                          app.setSpeakerMins((prev) => ({
                            ...prev,
                            [file.path]: event.target.value,
                          }))
                        }
                        className="w-20"
                      />
                    </label>
                    <label className="flex flex-col gap-1 text-xs text-muted">
                      Макс
                      <Input
                        id={`max-${index}`}
                        type="number"
                        min={1}
                        step={1}
                        placeholder="—"
                        title="Верхняя граница числа говорящих: пусто — без ограничения"
                        value={app.speakerMaxs[file.path] ?? ''}
                        onChange={(event) =>
                          app.setSpeakerMaxs((prev) => ({
                            ...prev,
                            [file.path]: event.target.value,
                          }))
                        }
                        className="w-20"
                      />
                    </label>
                  </div>
                )}
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  )
}

export default FilesPanel
