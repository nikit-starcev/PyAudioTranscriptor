import { AppProvider } from './app/AppProvider'
import AppShell from './components/layout/AppShell'

function App() {
  return (
    <AppProvider>
      <AppShell />
    </AppProvider>
  )
}

export default App
