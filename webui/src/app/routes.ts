export type Route =
  | { name: 'jobs' }
  | { name: 'job'; id: string; entry?: number }
  | { name: 'archive' }
  | { name: 'models' }
  | { name: 'binaries' }
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
  const raw = hash.replace(/^#/, '').replace(/^\/+/, '')
  const [path, queryString] = raw.split('?')
  const [head, ...rest] = path.split('/')
  const query = new URLSearchParams(queryString ?? '')
  switch (head) {
    case 'jobs': {
      if (!rest[0]) return { name: 'jobs' }
      const entryRaw = query.get('entry')
      const entry = entryRaw != null && /^\d+$/.test(entryRaw) ? Number(entryRaw) : undefined
      return entry == null
        ? { name: 'job', id: decodeURIComponent(rest[0]) }
        : { name: 'job', id: decodeURIComponent(rest[0]), entry }
    }
    case 'archive':
      return { name: 'archive' }
    case 'models':
      return { name: 'models' }
    case 'binaries':
      return { name: 'binaries' }
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
