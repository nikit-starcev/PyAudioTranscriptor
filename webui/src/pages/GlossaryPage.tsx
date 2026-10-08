import GlossaryPanel from '../components/GlossaryPanel'

function GlossaryPage() {
  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-lg font-semibold">Глоссарий</h1>
        <p className="text-sm text-muted">
          Термины и их ошибочные формы, применяемые при распознавании
        </p>
      </div>
      <GlossaryPanel />
    </div>
  )
}

export default GlossaryPage
