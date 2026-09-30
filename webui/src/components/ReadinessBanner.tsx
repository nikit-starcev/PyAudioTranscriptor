import { type DoctorCheck, type DoctorReport } from '../api'

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
        <span
          aria-hidden
          className={
            isFail
              ? 'mt-0.5 text-red-600 dark:text-red-400'
              : 'mt-0.5 text-amber-600 dark:text-amber-400'
          }
        >
          {isFail ? '✗' : '!'}
        </span>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium">
            {check.label}
            {check.critical ? '' : ' — некритично'}
          </p>
          {check.detail && (
            <p className="text-xs text-slate-600 dark:text-slate-300">
              <code>{check.detail}</code>
            </p>
          )}
          {check.hint && (
            <p className="text-xs text-slate-600 dark:text-slate-300">{check.hint}</p>
          )}
          {check.links.length > 0 && (
            <p className="mt-0.5 flex flex-wrap gap-x-3 gap-y-0.5 text-xs">
              {check.links.map((link) => (
                <a
                  key={link}
                  href={link}
                  target="_blank"
                  rel="noreferrer noopener"
                  className="text-blue-600 underline hover:text-blue-500 dark:text-blue-400 dark:hover:text-blue-300"
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
    <button
      type="button"
      onClick={onRecheck}
      disabled={loading}
      className="rounded-md border border-slate-300 bg-white px-3 py-1 text-xs hover:bg-slate-100 disabled:opacity-40 dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100 dark:hover:bg-slate-800"
    >
      {loading ? 'Проверка…' : 'Проверить снова'}
    </button>
  )

  if (!report) {
    return (
      <div className="flex flex-wrap items-center gap-3 rounded-md border border-slate-200 bg-white px-4 py-2 text-sm text-slate-500 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-400">
        <span>{error ? `Не удалось проверить готовность: ${error}` : 'Проверка готовности…'}</span>
        {error && recheckButton}
      </div>
    )
  }

  if (problems.length === 0) {
    return (
      <div className="flex flex-wrap items-center gap-3 rounded-md border border-emerald-200 bg-emerald-50 px-4 py-2 text-sm text-emerald-700 dark:border-emerald-900 dark:bg-emerald-950/40 dark:text-emerald-300">
        <span className="font-medium">✓ Окружение готово</span>
        <span className="text-xs">
          критичных проблем нет · проверок {report.summary.ok} в порядке
        </span>
        <span className="ml-auto">{recheckButton}</span>
      </div>
    )
  }

  const palette = blocked
    ? 'border-red-300 bg-red-50 text-red-800 dark:border-red-900 dark:bg-red-950/50 dark:text-red-200'
    : 'border-amber-300 bg-amber-50 text-amber-800 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-200'

  return (
    <section
      role="alert"
      className={`rounded-md border px-4 py-3 ${palette}`}
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
      <ul className="mt-2 divide-y divide-black/5 dark:divide-white/10">
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
