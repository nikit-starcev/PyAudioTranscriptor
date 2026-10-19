import { useEffect, useState } from 'react'
import { BookOpen, Bug, Lightbulb } from 'lucide-react'

import {
  api,
  errorMessage,
  type AsrDeviceInfo,
  type HardwareInfo,
  type HardwareReport,
  type WebSettings,
} from '../api'
import {
  ARCHITECTURE_DOCS_URL,
  FEATURE_REQUEST_URL,
  buildBugReportUrl,
  type BugReportContext,
} from '../feedback'
import { Card, CardContent, CardHeader } from './ui'

type Props = {
  version: string
  asrDevice: AsrDeviceInfo | null
}

const ACTION_LINK =
  'inline-flex h-9 items-center justify-center gap-2 rounded-md px-4 text-sm font-medium ' +
  'transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 ' +
  'focus-visible:outline-ring'

function cpuLabel(hardware: HardwareInfo): string | null {
  const name = hardware.cpu.name?.trim()
  const freq = hardware.cpu.max_freq_mhz
  if (!name) return freq ? `${freq} МГц` : null
  return freq ? `${name}, до ${freq} МГц` : name
}

function coresLabel(hardware: HardwareInfo): string | null {
  const logical = hardware.cpu.logical_cores
  const physical = hardware.cpu.physical_cores
  if (logical == null && physical == null) return null
  if (logical != null && physical != null) {
    return `${physical} физических / ${logical} логических`
  }
  return `${logical ?? physical}`
}

function gpuLabel(hardware: HardwareInfo): string {
  const gpu = hardware.gpu
  if (!gpu.present) return 'не обнаружен'
  const name = [gpu.vendor, gpu.name].filter(Boolean).join(' ')
  const vram = gpu.vram_gb != null ? `${gpu.vram_gb} ГБ` : null
  const accelerators = gpu.accelerators.length > 0 ? gpu.accelerators.join(', ') : null
  return [name || 'обнаружен', vram, accelerators].filter(Boolean).join(', ')
}

function ramLabel(hardware: HardwareInfo): string | null {
  if (hardware.total_ram_gb == null) return null
  return `${hardware.total_ram_gb} ГБ`
}

function FeedbackPanel({ version, asrDevice }: Props) {
  const [hardware, setHardware] = useState<HardwareInfo | null>(null)
  const [settings, setSettings] = useState<WebSettings | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false

    const load = async () => {
      const [hardwareResult, settingsResult] = await Promise.allSettled([
        api<HardwareReport>('/api/hardware'),
        api<WebSettings>('/api/settings'),
      ])
      if (cancelled) return

      const failures: string[] = []
      if (hardwareResult.status === 'fulfilled') {
        setHardware(hardwareResult.value.hardware)
      } else {
        failures.push(errorMessage(hardwareResult.reason))
      }
      if (settingsResult.status === 'fulfilled') {
        setSettings(settingsResult.value)
      } else {
        failures.push(errorMessage(settingsResult.reason))
      }
      setError(failures.length > 0 ? failures.join('; ') : null)
    }

    void load()
    return () => {
      cancelled = true
    }
  }, [])

  const context: BugReportContext = {
    version,
    os: hardware?.os ?? null,
    asrBackend: settings?.asr_backend ?? asrDevice?.backend ?? null,
    asrDevice: settings?.device ?? asrDevice?.device ?? null,
    asrLabel: asrDevice?.label ?? null,
    diarizationEngine: settings?.diarization_engine ?? null,
    cpu: hardware ? cpuLabel(hardware) : null,
    cores: hardware ? coresLabel(hardware) : null,
    ram: hardware ? ramLabel(hardware) : null,
    gpu: hardware ? gpuLabel(hardware) : null,
    notes: hardware?.notes,
  }
  const bugReportUrl = buildBugReportUrl(context)

  return (
    <Card>
      <CardHeader
        title="Помощь и обратная связь"
        description="Ссылки открывают форму на GitHub с уже заполненным окружением. Данные никуда не отправляются автоматически."
      />
      <CardContent className="space-y-3">
        <div className="flex flex-wrap gap-2">
          <a
            href={bugReportUrl}
            target="_blank"
            rel="noreferrer noopener"
            aria-label="Сообщить о проблеме на GitHub — форма откроется с заполненным окружением"
            className={`${ACTION_LINK} bg-primary text-primary-fg shadow-sm hover:bg-primary-hover`}
          >
            <Bug aria-hidden className="h-4 w-4" />
            Сообщить о проблеме
          </a>
          <a
            href={FEATURE_REQUEST_URL}
            target="_blank"
            rel="noreferrer noopener"
            aria-label="Предложить идею на GitHub"
            className={`${ACTION_LINK} border border-border-strong bg-surface text-text hover:bg-surface-2`}
          >
            <Lightbulb aria-hidden className="h-4 w-4" />
            Предложить идею
          </a>
          <a
            href={ARCHITECTURE_DOCS_URL}
            target="_blank"
            rel="noreferrer noopener"
            aria-label="Открыть документацию по архитектуре на GitHub"
            className={`${ACTION_LINK} text-text hover:bg-surface-2`}
          >
            <BookOpen aria-hidden className="h-4 w-4" />
            Документация
          </a>
        </div>
        <p className="max-w-prose text-pretty text-xs text-muted">
          В форму для бага подставляются версия, ОС, ASR-бэкенд и устройство, движок
          диаризации и основные параметры железа. Перед отправкой проверьте текст: при
          необходимости уберите приватные данные.
        </p>
        {error && (
          <p className="max-w-prose text-pretty text-xs text-warn">
            Часть данных окружения недоступна: {error}. Ссылка всё равно откроется, поля
            можно заполнить вручную.
          </p>
        )}
      </CardContent>
    </Card>
  )
}

export default FeedbackPanel
