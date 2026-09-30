import { useThemeMode, type ThemeMode } from '../theme'

const OPTIONS: { value: ThemeMode; label: string; title: string }[] = [
  { value: 'light', label: 'Светлая', title: 'Светлая тема' },
  { value: 'dark', label: 'Тёмная', title: 'Тёмная тема' },
  { value: 'auto', label: 'Авто', title: 'Как в системе (prefers-color-scheme)' },
]

function ThemeToggle() {
  const [mode, setMode] = useThemeMode()

  return (
    <div
      role="group"
      aria-label="Тема оформления"
      className="inline-flex overflow-hidden rounded-md border border-slate-300 text-sm dark:border-slate-700"
    >
      {OPTIONS.map((option) => {
        const active = mode === option.value
        return (
          <button
            key={option.value}
            type="button"
            title={option.title}
            aria-pressed={active}
            onClick={() => setMode(option.value)}
            className={
              active
                ? 'bg-slate-800 px-2.5 py-1.5 text-white dark:bg-slate-200 dark:text-slate-900'
                : 'px-2.5 py-1.5 text-slate-700 hover:bg-slate-100 dark:text-slate-200 dark:hover:bg-slate-800'
            }
          >
            {option.label}
          </button>
        )
      })}
    </div>
  )
}

export default ThemeToggle
