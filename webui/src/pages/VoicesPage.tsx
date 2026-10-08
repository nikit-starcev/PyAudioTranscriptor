import { navigateTo } from '../app/routes'
import VoicesModal from '../components/VoicesModal'

function VoicesPage() {
  return <VoicesModal open onClose={() => navigateTo('/jobs')} />
}

export default VoicesPage
