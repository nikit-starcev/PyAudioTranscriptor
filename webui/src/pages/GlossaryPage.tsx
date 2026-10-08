import { navigateTo } from '../app/routes'
import GlossaryModal from '../components/GlossaryModal'

function GlossaryPage() {
  return <GlossaryModal open onClose={() => navigateTo('/jobs')} />
}

export default GlossaryPage
