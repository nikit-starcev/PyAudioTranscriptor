import { useApp } from '../app/useApp'
import ModelsPanel from '../components/ModelsPanel'
import { Card, CardContent } from '../components/ui'

function ModelsPage() {
  const app = useApp()
  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-lg font-semibold">Модели</h1>
        <p className="text-sm text-muted">Авто-скачивание с Hugging Face и управление файлами</p>
      </div>
      <Card>
        <CardContent>
          <ModelsPanel onChanged={app.handleModelsChanged} />
        </CardContent>
      </Card>
    </div>
  )
}

export default ModelsPage
