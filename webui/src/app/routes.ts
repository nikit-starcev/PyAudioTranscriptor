export type Route =
  | { name: 'jobs' }
  | { name: 'job'; id: string }
  | { name: 'models' }
  | { name: 'glossary' }
  | { name: 'voices' }
  | { name: 'settings' }
  | { name: 'wizard' }

export const DEFAULT_ROUTE = '#/jobs'

export function navigateTo(to: string): void {
  const next = to.startsWith('#') ? to : `#/${to.replace(/^\/+/, '')}`
  if (window.location.hash === next) return
  window.location.hash = next
}

export function parseHash(hash: string): Route {
  const path = hash.replace(/^#/, '').replace(/^\/+/, '')
  const [head, ...rest] = path.split('/')
  switch (head) {
    case 'jobs':
      return rest[0] ? { name: 'job', id: decodeURIComponent(rest[0]) } : { name: 'jobs' }
    case 'models':
      return { name: 'models' }
    case 'glossary':
      return { name: 'glossary' }
    case 'voices':
      return { name: 'voices' }
    case 'settings':
      return { name: 'settings' }
    case 'wizard':
      return { name: 'wizard' }
    default:
      return { name: 'jobs' }
  }
}
