import { createContext, useContext, useState, useEffect, useCallback } from 'react'
import { login as apiLogin, register as apiRegister, getProfile } from '../api/client'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  const [token, setToken] = useState(() => localStorage.getItem('dc_token'))
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  // On mount (or token change), try to load user profile
  useEffect(() => {
    if (!token) {
      setUser(null)
      setLoading(false)
      return
    }

    setLoading(true)
    getProfile()
      .then(profile => {
        setUser(profile)
        localStorage.setItem('dc_user', JSON.stringify(profile))
      })
      .catch(() => {
        // Token invalid/expired -- clear everything
        setUser(null)
        setToken(null)
        localStorage.removeItem('dc_token')
        localStorage.removeItem('dc_user')
      })
      .finally(() => setLoading(false))
  }, [token])

  // Listen for forced logout events (from API client on 401)
  useEffect(() => {
    function handleLogout() {
      setUser(null)
      setToken(null)
    }
    window.addEventListener('dc_logout', handleLogout)
    return () => window.removeEventListener('dc_logout', handleLogout)
  }, [])

  const login = useCallback(async (username, password) => {
    setError(null)
    try {
      const res = await apiLogin(username, password)
      localStorage.setItem('dc_token', res.access_token)
      setToken(res.access_token)
      return res
    } catch (e) {
      setError(e.message)
      throw e
    }
  }, [])

  const register = useCallback(async (username, password, email) => {
    setError(null)
    try {
      const res = await apiRegister(username, password, email || null)
      return res
    } catch (e) {
      setError(e.message)
      throw e
    }
  }, [])

  const logout = useCallback(() => {
    setUser(null)
    setToken(null)
    localStorage.removeItem('dc_token')
    localStorage.removeItem('dc_user')
  }, [])

  const refreshProfile = useCallback(async () => {
    if (!token) return
    try {
      const profile = await getProfile()
      setUser(profile)
      localStorage.setItem('dc_user', JSON.stringify(profile))
      return profile
    } catch (e) {
      console.error('Failed to refresh profile:', e)
    }
  }, [token])

  const value = {
    user,
    token,
    loading,
    error,
    isAuthenticated: !!user && !!token,
    hasApiKey: !!user?.has_api_key,
    login,
    register,
    logout,
    refreshProfile,
    clearError: () => setError(null),
  }

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

export default AuthContext
