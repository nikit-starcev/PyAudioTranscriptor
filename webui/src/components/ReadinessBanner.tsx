import { AlertCircle, Ban, TriangleAlert } from 'lucide-react'

import { type DoctorCheck, type DoctorReport } from '../api'
import { Button, Card, cn } from './ui'

type Props = {
  report: DoctorReport | null
  loading: boolean
  error: string | null
  onRecheck: () => void
}

function linkLabel(url: string): string {
  try {
    const parsed = new URL(url)
    return `${parsed.hostname}${parsed.pathname}`.replace(/\/$/, '')
  } catch {
    return url
  }
}

function CheckRow({ check }: { check: DoctorCheck }) {
  const isFail = check.status === 'fail'
  return (
    <li className="py-1.5">
      <div className="flex items-start gap-2">
        <span aria-hidden className={cn('mt-0.5 shrink-0', isFail ? 'text-danger' : 'text-warn')}>
          {isFail ? <Ban className="h-4 w-4" /> : <TriangleAlert className="h-4 w-4" />}
        </span>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium">
            {check.label}
            {check.critical ? '' : ' — некритично'}
          </p>
          {check.detail && (
            <p className="text-xs text-muted">
              <code>{check.detail}</code>
            </p>
          )}
          {check.hint && <p className="text-xs text-muted">{check.hint}</p>}
          {check.links.length > 0 && (
            <p className="mt-0.5 flex flex-wrap gap-x-3 gap-y-0.5 text-xs">
              {check.links.map((link) => (
                <a
                  key={link}
                  href={link}
                  target="_blank"
                  rel="noreferrer noopener"
                  className="font-medium text-primary underline hover:opacity-80"
                >
                  {linkLabel(link)}
                </a>
              ))}
            </p>
          )}
        </div>
      </div>
    </li>
  )
}

function ReadinessBanner({ report, loading, error, onRecheck }: Props) {
  const problems = report?.checks.filter((check) => check.status !== 'ok') ?? []
  const failures = problems.filter((check) => check.status === 'fail')
  const warnings = problems.filter((check) => check.status === 'warn')
  const blocked = failures.length > 0

  const recheckButton = (
    <Button variant="secondary" size="sm" loading={loading} onClick={onRecheck}>
      Проверить снова
    </Button>
  )

  if (!report) {
    return (
      <Card padded className="flex flex-wrap items-center gap-3 text-sm text-muted">
        <AlertCircle aria-hidden className="h-4 w-4 shrink-0" />
        <span>{error ? `Не удалось проверить готовность: ${error}` : 'Проверка готовности…'}</span>
        {error && <span className="ml-auto">{recheckButton}</span>}
      </Card>
    )
  }

  if (problems.length === 0) {
    return (
      <div className="flex flex-wrap items-center gap-3 rounded-lg border border-success/40 bg-success-soft px-4 py-3 text-sm text-success-soft-fg">
        <span className="font-medium">Окружение готово</span>
        <span className="text-xs">
          критичных проблем нет · проверок {report.summary.ok} в порядке
        </span>
        <span className="ml-auto">{recheckButton}</span>
      </div>
    )
  }

  return (
    <section
      role="alert"
      className={cn(
        'rounded-lg border px-4 py-3',
        blocked
          ? 'border-danger/40 bg-danger-soft text-danger-soft-fg'
          : 'border-warn/40 bg-warn-soft text-warn-soft-fg',
      )}
    >
      <div className="flex flex-wrap items-center gap-3">
        <h2 className="font-semibold">
          {blocked
            ? `Не готово к запуску — критичных проблем: ${failures.length}`
            : `Есть предупреждения: ${warnings.length}`}
        </h2>
        <span className="text-xs">
          {blocked
            ? 'Запуск обработки заблокирован, пока критичные проблемы не устранены.'
            : 'Запуск возможен, но часть функций может быть недоступна.'}
        </span>
        <span className="ml-auto">{recheckButton}</span>
      </div>
      <ul className="mt-2 divide-y divide-border/60">
        {failures.map((check) => (
          <CheckRow key={check.id} check={check} />
        ))}
        {warnings.map((check) => (
          <CheckRow key={check.id} check={check} />
        ))}
      </ul>
    </section>
  )
}

export default ReadinessBanner
