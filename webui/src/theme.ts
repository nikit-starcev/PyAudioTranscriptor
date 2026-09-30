// Управление темой оформления: Светлая / Тёмная / Авто.
//
// Выбор хранится в localStorage под ключом ``at-theme`` и применяется классом
// ``dark`` на <html>. Inline-скрипт в index.html делает то же самое до первой
// отрисовки, поэтому мигания светлой темы при загрузке нет.

import { useCallback, useEffect, useState } from 'react'

export type ThemeMode = 'light' | 'dark' | 'auto'

const STORAGE_KEY = 'at-theme'
const MEDIA_QUERY = '(prefers-color-scheme: dark)'

export function readThemeMode(): ThemeMode {
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored === 'light' || stored === 'dark' || stored === 'auto') return stored
  } catch {
    // приватный режим / отключённое хранилище — используем «Авто»
  }
  return 'auto'
}

function prefersDark(): boolean {
  return typeof window !== 'undefined' && window.matchMedia(MEDIA_QUERY).matches
}

export function isDarkMode(mode: ThemeMode): boolean {
  return mode === 'dark' || (mode === 'auto' && prefersDark())
}

export function applyThemeMode(mode: ThemeMode): void {
  const dark = isDarkMode(mode)
  const root = document.documentElement
  root.classList.toggle('dark', dark)
  root.style.colorScheme = dark ? 'dark' : 'light'
  try {
    localStorage.setItem(STORAGE_KEY, mode)
  } catch {
    // сохранение не критично
  }
}

export function useThemeMode(): [ThemeMode, (mode: ThemeMode) => void] {
  const [mode, setMode] = useState<ThemeMode>(() => readThemeMode())

  useEffect(() => {
    applyThemeMode(mode)
  }, [mode])

  // В режиме «Авто» реагируем на смену системной темы на лету.
  useEffect(() => {
    if (mode !== 'auto') return
    const media = window.matchMedia(MEDIA_QUERY)
    const onChange = () => applyThemeMode('auto')
    media.addEventListener('change', onChange)
    return () => media.removeEventListener('change', onChange)
  }, [mode])

  const update = useCallback((next: ThemeMode) => setMode(next), [])
  return [mode, update]
}
