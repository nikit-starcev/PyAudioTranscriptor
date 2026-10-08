import { useEffect, useRef, useState } from 'react'

import { navigateTo, type Route } from '../../app/routes'
import { useApp } from '../../app/useApp'
import { useHashRoute } from '../../app/useHashRoute'
import { Alert } from '../ui'
import ActionProgressCard from '../ActionProgressCard'
import GlossaryPage from '../../pages/GlossaryPage'
import JobPage from '../../pages/JobPage'
import JobsPage from '../../pages/JobsPage'
import ModelsPage from '../../pages/ModelsPage'
import SettingsPage from '../../pages/SettingsPage'
import VoicesPage from '../../pages/VoicesPage'
import WizardPage from '../../pages/WizardPage'
import Sidebar from './Sidebar'
import Topbar from './Topbar'

function RouteContent({ route }: { route: Route }) {
  switch (route.name) {
    case 'jobs':
      return <JobsPage />
    case 'job':
      return <JobPage jobId={route.id} />
    case 'models':
      return <ModelsPage />
    case 'glossary':
      return <GlossaryPage />
    case 'voices':
      return <VoicesPage />
    case 'settings':
      return <SettingsPage />
    case 'wizard':
      return <WizardPage />
  }
}

function AppShell() {
  const app = useApp()
  const { route } = useHashRoute()
  const [navOpen, setNavOpen] = useState(false)
  const wizardAutoShown = useRef(false)

  useEffect(() => {
    if (!app.doctor) return
    if (app.doctor.summary.critical_failures > 0 && !wizardAutoShown.current) {
      wizardAutoShown.current = true
      navigateTo('/wizard')
    }
  }, [app.doctor])

  useEffect(() => {
    const close = () => setNavOpen(false)
    window.addEventListener('hashchange', close)
    return () => window.removeEventListener('hashchange', close)
  }, [])

  return (
    <div className="min-h-dvh bg-bg text-text">
      <Topbar
        version={app.version}
        asrDevice={app.asrDevice}
        onOpenNav={() => setNavOpen(true)}
      />
      <div className="flex">
        <Sidebar route={route} open={navOpen} onClose={() => setNavOpen(false)} />
        <main className="min-w-0 flex-1">
          <div className="mx-auto w-full max-w-6xl space-y-5 px-4 py-6 sm:px-6 lg:px-8">
            {app.error && (
              <Alert tone="danger" live onDismiss={() => app.setError(null)}>
                {app.error}
              </Alert>
            )}
            {app.applyNotice && (
              <Alert
                tone={app.applyNotice.kind === 'error' ? 'danger' : 'info'}
                live
                onDismiss={() => app.setApplyNotice(null)}
              >
                {app.applyNotice.text}
              </Alert>
            )}
            <RouteContent route={route} />
          </div>
        </main>
      </div>
      <ActionProgressCard run={app.actionRun} onClose={app.resetActionProgress} />
    </div>
  )
}

export default AppShell
