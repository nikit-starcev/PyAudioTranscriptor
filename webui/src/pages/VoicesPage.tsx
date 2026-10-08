import VoicesPanel from '../components/VoicesPanel'

function VoicesPage() {
  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-lg font-semibold">Библиотека голосов</h1>
        <p className="text-sm text-muted">
          Образцы голоса для сопоставления говорящих с именами
        </p>
      </div>
      <VoicesPanel />
    </div>
  )
}

export default VoicesPage
