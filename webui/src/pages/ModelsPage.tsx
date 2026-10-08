import { navigateTo } from '../app/routes'
import { useApp } from '../app/useApp'
import ModelsModal from '../components/ModelsModal'

function ModelsPage() {
  const app = useApp()
  return (
    <ModelsModal
      open
      onClose={() => navigateTo('/jobs')}
      onChanged={app.handleModelsChanged}
    />
  )
}

export default ModelsPage
