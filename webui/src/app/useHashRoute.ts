import { useCallback, useEffect, useState } from 'react'

import { DEFAULT_ROUTE, parseHash, type Route } from './routes'

export function useHashRoute(): { route: Route; navigate: (to: string) => void } {
  const [route, setRoute] = useState<Route>(() =>
    window.location.hash ? parseHash(window.location.hash) : parseHash(DEFAULT_ROUTE),
  )

  useEffect(() => {
    const onHashChange = () => setRoute(parseHash(window.location.hash))
    window.addEventListener('hashchange', onHashChange)
    if (!window.location.hash) {
      window.history.replaceState(null, '', DEFAULT_ROUTE)
    }
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])

  const navigate = useCallback((to: string) => {
    const next = to.startsWith('#') ? to : `#${to.startsWith('/') ? '' : '/'}${to}`
    if (window.location.hash === next) {
      setRoute(parseHash(next))
      return
    }
    window.location.hash = next
  }, [])

  return { route, navigate }
}
