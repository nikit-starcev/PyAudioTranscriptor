import { Menu, Settings, Waves } from 'lucide-react'

import type { AsrDeviceInfo } from '../../api'
import { Badge, Button, IconButton, Tooltip, type BadgeTone } from '../ui'
import ThemeToggle from './ThemeToggle'

type Props = {
  version: string
  asrDevice: AsrDeviceInfo | null
  onOpenNav: () => void
}

function asrTone(device: AsrDeviceInfo | null): BadgeTone {
  if (device?.device === 'gpu') return 'success'
  if (device?.device === 'unknown') return 'warn'
  return 'neutral'
}

function Topbar({ version, asrDevice, onOpenNav }: Props) {
  const asrTitle = asrDevice
    ? `${asrDevice.label}${asrDevice.details.length ? ` · ${asrDevice.details.join('; ')}` : ''} · ${asrDevice.note}`
    : ''

  return (
    <header className="sticky top-0 z-30 border-b border-border bg-surface/90 backdrop-blur">
      <div className="flex h-14 items-center gap-3 px-3 sm:px-6">
        <IconButton
          aria-label="Открыть навигацию"
          className="lg:hidden"
          onClick={onOpenNav}
        >
          <Menu aria-hidden className="h-5 w-5" />
        </IconButton>

        <a
          href="#/jobs"
          className="flex min-w-0 items-center gap-2 rounded-md focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"
        >
          <span
            aria-hidden
            className="flex h-8 w-8 items-center justify-center rounded-md bg-primary text-primary-fg"
          >
            <Waves className="h-4 w-4" />
          </span>
          <span className="truncate text-sm font-semibold">AudioTranscriber</span>
        </a>

        <div className="ml-auto flex min-w-0 items-center gap-2">
          {asrDevice && (
            <div className="hidden min-w-0 lg:block">
              <Tooltip label={asrTitle} className="min-w-0">
                <Badge
                  tone={asrTone(asrDevice)}
                  title={asrTitle}
                  className="max-w-[16rem] min-w-0"
                >
                  <span className="min-w-0 truncate">ASR: {asrDevice.label}</span>
                </Badge>
              </Tooltip>
            </div>
          )}
          {version && (
            <span className="hidden shrink-0 sm:inline-flex">
              <Badge tone="neutral" title={`Версия ${version}`}>
                v{version}
              </Badge>
            </span>
          )}
          <ThemeToggle />
          <Button
            variant="secondary"
            size="sm"
            aria-label="Настройки"
            icon={<Settings aria-hidden className="h-4 w-4" />}
            onClick={() => {
              window.location.hash = '#/settings'
            }}
          >
            <span className="hidden sm:inline">Настройки</span>
          </Button>
        </div>
      </div>
    </header>
  )
}

export default Topbar
