import { Routes, Route, Navigate } from 'react-router-dom'
import { ThemeProvider } from './context/ThemeContext'
import Navbar from './components/Navbar'
import Footer from './components/Footer'
import Home from './pages/Home'
import Docs from './pages/Docs'
import Demo from './pages/Demo'
import Dashboard from './pages/Dashboard'
import StatsPage from './pages/dashboard/StatsPage'
import ChatPage from './pages/dashboard/ChatPage'
import GraphPage from './pages/dashboard/GraphPage'
import MemoriesPage from './pages/dashboard/MemoriesPage'
import LifecyclePage from './pages/dashboard/LifecyclePage'

export default function App() {
  return (
    <ThemeProvider>
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/docs" element={<Docs />} />
        <Route path="/demo" element={<Demo />} />
        <Route path="/dashboard" element={<Dashboard />}>
          <Route index element={<Navigate to="stats" replace />} />
          <Route path="stats" element={<StatsPage />} />
          <Route path="chat" element={<ChatPage />} />
          <Route path="graph" element={<GraphPage />} />
          <Route path="memories" element={<MemoriesPage />} />
          <Route path="lifecycle" element={<LifecyclePage />} />
        </Route>
      </Routes>
      <Footer />
    </ThemeProvider>
  )
}
