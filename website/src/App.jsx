import { Routes, Route, Navigate, useLocation } from 'react-router-dom'
import { ThemeProvider } from './context/ThemeContext'
import Navbar from './components/Navbar'
import Footer from './components/Footer'
import ProtectedRoute from './components/ProtectedRoute'
import Home from './pages/Home'
import Docs from './pages/Docs'
import Demo from './pages/Demo'
import Dashboard from './pages/Dashboard'
import Login from './pages/Login'
import Settings from './pages/Settings'
import StatsPage from './pages/dashboard/StatsPage'
import ChatPage from './pages/dashboard/ChatPage'
import GraphPage from './pages/dashboard/GraphPage'
import MemoriesPage from './pages/dashboard/MemoriesPage'
import LifecyclePage from './pages/dashboard/LifecyclePage'

function AppRoutes() {
  const location = useLocation()
  const isDashboard = location.pathname.startsWith('/dashboard')

  return (
    <>
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/docs" element={<Docs />} />
        <Route path="/demo" element={<Demo />} />
        <Route path="/login" element={<Login />} />
        <Route
          path="/dashboard"
          element={
            <ProtectedRoute>
              <Dashboard />
            </ProtectedRoute>
          }
        >
          <Route index element={<Navigate to="stats" replace />} />
          <Route path="stats" element={<StatsPage />} />
          <Route path="chat" element={<ChatPage />} />
          <Route path="graph" element={<GraphPage />} />
          <Route path="memories" element={<MemoriesPage />} />
          <Route path="lifecycle" element={<LifecyclePage />} />
          <Route path="settings" element={<Settings />} />
        </Route>
        <Route path="/settings" element={<Navigate to="/dashboard/settings" replace />} />
      </Routes>
      {!isDashboard && <Footer />}
    </>
  )
}

export default function App() {
  return (
    <ThemeProvider>
      <AppRoutes />
    </ThemeProvider>
  )
}
