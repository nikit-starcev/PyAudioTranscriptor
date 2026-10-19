export const REPOSITORY_URL = 'https://github.com/nikit-starcev/PyAudioTranscriptor'
export const BUG_REPORT_URL = `${REPOSITORY_URL}/issues/new?template=bug_report.yml`
export const FEATURE_REQUEST_URL = `${REPOSITORY_URL}/issues/new?template=feature_request.yml`
export const ARCHITECTURE_DOCS_URL = `${REPOSITORY_URL}/tree/main/docs/architecture`

export type BugReportContext = {
  version?: string | null
  os?: string | null
  asrBackend?: string | null
  asrDevice?: string | null
  asrLabel?: string | null
  diarizationEngine?: string | null
  cpu?: string | null
  cores?: string | null
  ram?: string | null
  gpu?: string | null
  notes?: string[]
}

function valueOrUnknown(value: string | null | undefined): string {
  const trimmed = value?.trim()
  return trimmed ? trimmed : 'неизвестно'
}

function environmentLines(context: BugReportContext): string[] {
  const asrDevice = [context.asrDevice, context.asrLabel]
    .map((value) => value?.trim())
    .filter((value): value is string => Boolean(value))
    .join(' — ')

  const lines = [
    `- Версия: ${valueOrUnknown(context.version)}`,
    `- ОС: ${valueOrUnknown(context.os)}`,
    `- ASR-бэкенд: ${valueOrUnknown(context.asrBackend)}`,
    `- ASR-устройство: ${valueOrUnknown(asrDevice)}`,
    `- Движок диаризации: ${valueOrUnknown(context.diarizationEngine)}`,
    `- CPU: ${valueOrUnknown(context.cpu)}`,
    `- Ядра: ${valueOrUnknown(context.cores)}`,
    `- RAM: ${valueOrUnknown(context.ram)}`,
    `- GPU: ${valueOrUnknown(context.gpu)}`,
  ]

  const notes = context.notes?.map((note) => note.trim()).filter(Boolean) ?? []
  if (notes.length > 0) lines.push(`- Оговорки: ${notes.join('; ')}`)
  return lines
}

export function buildBugReportBody(context: BugReportContext): string {
  return [
    '### Что произошло',
    '',
    '<!-- Опишите проблему: что делали, что ожидали и что получилось -->',
    '',
    '### Окружение',
    '',
    ...environmentLines(context),
    '',
    '### Логи',
    '',
    'Приложите лог/вывод `audio-transcriber doctor` (уберите приватные данные).',
  ].join('\n')
}

export function buildBugReportUrl(context: BugReportContext): string {
  const version = context.version?.trim()
  const title = version ? `Баг: (v${version})` : 'Баг: '
  const body = buildBugReportBody(context)
  return `${BUG_REPORT_URL}&title=${encodeURIComponent(title)}&body=${encodeURIComponent(body)}`
}
