import { Monitor, Moon, Sun } from 'lucide-react'

import { useThemeMode, type ThemeMode } from '../../theme'
import { Button } from '../ui'

const OPTIONS: { value: ThemeMode; label: string; Icon: typeof Sun }[] = [
  { value: 'light', label: 'Светлая тема', Icon: Sun },
  { value: 'dark', label: 'Тёмная тема', Icon: Moon },
  { value: 'auto', label: 'Как в системе', Icon: Monitor },
]

function ThemeToggle() {
  const [mode, setMode] = useThemeMode()

  return (
    <div
      role="group"
      aria-label="Тема оформления"
      className="inline-flex items-center gap-0.5 rounded-md border border-border bg-surface-2 p-0.5"
    >
      {OPTIONS.map(({ value, label, Icon }) => {
        const active = mode === value
        return (
          <Button
            key={value}
            variant={active ? 'secondary' : 'ghost'}
            size="sm"
            iconOnly
            aria-pressed={active}
            aria-label={label}
            title={label}
            onClick={() => setMode(value)}
            className={active ? 'shadow-sm' : 'text-muted'}
          >
            <Icon aria-hidden className="h-4 w-4" />
          </Button>
        )
      })}
    </div>
  )
}

export default ThemeToggle
