import { navigateTo } from '../app/routes'
import { useApp } from '../app/useApp'
import SetupWizard from '../components/SetupWizard'

function WizardPage() {
  const app = useApp()
  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-lg font-semibold">Мастер первого запуска</h1>
        <p className="text-sm text-muted">Пошаговая настройка окружения и моделей</p>
      </div>
      <SetupWizard
        report={app.doctor}
        onRecheck={() => void app.recheckDoctor()}
        onDone={() => navigateTo('/jobs')}
        onChanged={app.handleWizardChanged}
      />
    </div>
  )
}

export default WizardPage
