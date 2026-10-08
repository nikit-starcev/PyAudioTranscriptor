import { navigateTo } from '../app/routes'
import { useApp } from '../app/useApp'
import BinaryRequirements from '../components/BinaryRequirements'
import { Button, Card, CardContent, CardHeader } from '../components/ui'

/**
 * Страница «Бинарные пакеты» (#105): самостоятельный раздел поверх того же
 * компонента и эндпоинтов, что и шаг мастера (#98). Внешние бинарники
 * (whisper-cli, llama-server, deep-filter) и Python-пакеты можно скачать,
 * установить/обновить и посмотреть их статус.
 */
function BinariesPage() {
  const app = useApp()
  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-lg font-semibold">Бинарные пакеты</h1>
        <p className="text-sm text-muted">
          Внешние компоненты и Python-пакеты: статус, скачивание, установка и обновление
        </p>
      </div>
      <Card>
        <CardHeader
          title="Внешние компоненты"
          description="Скачанные бинарники кладутся в служебный каталог и прописываются в настройках автоматически"
          actions={
            <Button variant="secondary" size="sm" onClick={() => navigateTo('/wizard')}>
              Мастер настройки
            </Button>
          }
        />
        <CardContent>
          <BinaryRequirements onChanged={app.handleWizardChanged} />
        </CardContent>
      </Card>
    </div>
  )
}

export default BinariesPage
