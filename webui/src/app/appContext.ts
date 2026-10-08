import { createContext } from 'react'

import type { useAppController } from './useAppController'

export type AppController = ReturnType<typeof useAppController>

export const AppContext = createContext<AppController | null>(null)
