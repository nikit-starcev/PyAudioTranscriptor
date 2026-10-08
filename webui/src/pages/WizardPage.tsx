import { navigateTo } from '../app/routes'
import { useApp } from '../app/useApp'
import SetupWizard from '../components/SetupWizard'

function WizardPage() {
  const app = useApp()
  return (
    <SetupWizard
      open
      onClose={() => navigateTo('/jobs')}
      report={app.doctor}
      onRecheck={() => void app.recheckDoctor()}
      onChanged={app.handleWizardChanged}
    />
  )
}

export default WizardPage
