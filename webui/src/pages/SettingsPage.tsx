import { useApp } from '../app/useApp'
import ApiKeyPanel from '../components/ApiKeyPanel'
import CachePanel from '../components/CachePanel'
import SettingsPanel from '../components/SettingsPanel'

function SettingsPage() {
  const app = useApp()
  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-lg font-semibold">Настройки</h1>
        <p className="text-sm text-muted">Режимы обработки, модели, диаризация и пути</p>
      </div>
      <SettingsPanel onSaved={app.handleSettingsSaved} />
      <ApiKeyPanel />
      <CachePanel />
    </div>
  )
}

export default SettingsPage
