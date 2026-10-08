import {
  BookText,
  Cpu,
  ListChecks,
  Package,
  Settings,
  Volume2,
  Wand2,
  X,
} from 'lucide-react'

import { cn, IconButton } from '../ui'
import type { Route } from '../../app/routes'

type NavMatch = Route['name']

const NAV: {
  id: string
  label: string
  hash: string
  Icon: typeof ListChecks
  match: NavMatch[]
}[] = [
  { id: 'jobs', label: 'Задачи', hash: '#/jobs', Icon: ListChecks, match: ['jobs', 'job'] },
  { id: 'models', label: 'Модели', hash: '#/models', Icon: Cpu, match: ['models'] },
  {
    id: 'binaries',
    label: 'Бинарные пакеты',
    hash: '#/binaries',
    Icon: Package,
    match: ['binaries'],
  },
  { id: 'voices', label: 'Голоса', hash: '#/voices', Icon: Volume2, match: ['voices'] },
  { id: 'glossary', label: 'Глоссарий', hash: '#/glossary', Icon: BookText, match: ['glossary'] },
  { id: 'wizard', label: 'Мастер', hash: '#/wizard', Icon: Wand2, match: ['wizard'] },
  { id: 'settings', label: 'Настройки', hash: '#/settings', Icon: Settings, match: ['settings'] },
]

type Props = {
  route: Route
  open: boolean
  onClose: () => void
}

function Sidebar({ route, open, onClose }: Props) {
  return (
    <>
      {open && (
        <div
          className="fixed inset-0 z-30 bg-overlay lg:hidden"
          onClick={onClose}
          aria-hidden
        />
      )}
      <aside
        className={cn(
          'fixed inset-y-0 left-0 z-40 flex w-64 flex-col border-r border-border bg-surface transition-transform',
          'lg:sticky lg:top-14 lg:z-0 lg:h-[calc(100dvh-3.5rem)] lg:translate-x-0',
          open ? 'translate-x-0' : '-translate-x-full',
        )}
      >
        <div className="flex items-center justify-between px-3 py-2 lg:hidden">
          <span className="text-sm font-semibold">Навигация</span>
          <IconButton aria-label="Закрыть навигацию" onClick={onClose}>
            <X aria-hidden className="h-4 w-4" />
          </IconButton>
        </div>
        <nav aria-label="Основная навигация" className="flex flex-col gap-0.5 p-2">
          {NAV.map(({ id, label, hash, Icon, match }) => {
            const active = match.includes(route.name)
            return (
              <a
                key={id}
                href={hash}
                aria-current={active ? 'page' : undefined}
                onClick={onClose}
                className={cn(
                  'flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors',
                  'focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring',
                  active
                    ? 'bg-primary-soft text-primary-soft-fg'
                    : 'text-muted hover:bg-surface-2 hover:text-text',
                )}
              >
                <Icon aria-hidden className="h-4 w-4 shrink-0" />
                <span className="truncate">{label}</span>
              </a>
            )
          })}
        </nav>
      </aside>
    </>
  )
}

export default Sidebar
