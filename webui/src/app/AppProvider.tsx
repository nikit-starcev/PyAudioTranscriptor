import type { ReactNode } from 'react'

import { AppContext } from './appContext'
import { useAppController } from './useAppController'

export function AppProvider({ children }: { children: ReactNode }) {
  const value = useAppController()
  return <AppContext.Provider value={value}>{children}</AppContext.Provider>
}
