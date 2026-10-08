import { useContext } from 'react'

import { AppContext } from './appContext'
import type { AppController } from './appContext'

export function useApp(): AppController {
  const ctx = useContext(AppContext)
  if (!ctx) throw new Error('useApp must be used within <AppProvider>')
  return ctx
}
