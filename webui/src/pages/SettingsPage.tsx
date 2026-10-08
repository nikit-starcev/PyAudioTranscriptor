import { navigateTo } from '../app/routes'
import { useApp } from '../app/useApp'
import SettingsModal from '../components/SettingsModal'

function SettingsPage() {
  const app = useApp()
  return (
    <SettingsModal
      open
      onClose={() => navigateTo('/jobs')}
      onSaved={app.handleSettingsSaved}
    />
  )
}

export default SettingsPage
