import { useApp } from '../app/useApp'
import ReadinessBanner from '../components/ReadinessBanner'
import FilesPanel from './jobs/FilesPanel'
import JobList from './jobs/JobList'

function JobsPage() {
  const app = useApp()

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-lg font-semibold">Задачи</h1>
        <p className="text-sm text-muted">
          Загрузка файлов, очередь обработки и готовность окружения
        </p>
      </div>

      <ReadinessBanner
        report={app.doctor}
        loading={app.doctorLoading}
        error={app.doctorError}
        onRecheck={() => void app.recheckDoctor()}
      />

      <div className="grid items-start gap-5 lg:grid-cols-[minmax(0,1fr)_20rem]">
        <JobList />
        <div className="lg:sticky lg:top-20">
          <FilesPanel />
        </div>
      </div>
    </div>
  )
}

export default JobsPage
